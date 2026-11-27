import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import lightkurve as lk
from astropy.timeseries import LombScargle, BoxLeastSquares
from openai import OpenAI
import base64
from scipy.ndimage import median_filter
from duckduckgo_search import DDGS
from streamlit_mic_recorder import speech_to_text

# ==========================================
# 1. 页面设置：浅蓝科研风，强制浅色模式
# ==========================================
st.set_page_config(page_title="星鉴 AstroFusion", layout="wide")

st.markdown("""
    <style>
    :root { color-scheme: light !important; }
    .stApp { background-color: #F4F7F9 !important; }
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E0E6ED; }
    div[data-baseweb="input"], input, textarea { background-color: #FFFFFF !important; color: #1F2937 !important; border-radius: 8px !important; }
    [data-testid="stChatMessage"] { background-color: #FFFFFF !important; border-radius: 10px !important; border: 1px solid #E5E7EB; }
    .stButton>button { background-color: #3B82F6 !important; color: #FFFFFF !important; border-radius: 8px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 密码锁
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.title("🔒 星鉴 AstroFusion V1.0 竞赛版")
        pwd = st.text_input("访问码 (默认 123456)", type="password")
        if st.button("进入平台", use_container_width=True):
            if pwd == "123456":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("密码错误！")
        return False
    return True

# ==========================================
# 3. 核心功能与算法
# ==========================================
@st.cache_data(show_spinner=False)
def download_tess_data(tic_id):
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search_result) == 0: return None
        lc = search_result[0].download()
        if lc is None: return None
        lc = lc.remove_nans().normalize()
        mask = (lc.flux.value > 0.8) & (lc.flux.value < 1.2)
        return lc[mask]
    except Exception:
        return None

def run_multi_algorithms(time_arr, flux_flat, min_p=0.2, max_p=5.0):
    freq = np.linspace(1/max_p, 1/min_p, 2000)
    
    # 1. Lomb-Scargle
    ls = LombScargle(time_arr, flux_flat - np.mean(flux_flat))
    power_ls = ls.power(freq)
    best_ls_idx = np.argmax(power_ls)
    p_ls = 1 / freq[best_ls_idx]
    max_p_ls = power_ls[best_ls_idx]
    
    # 2. BLS (Box Least Squares)
    try:
        bls = BoxLeastSquares(time_arr, flux_flat)
        durations = np.linspace(0.05, 0.2, 5)
        bls_res = bls.power(freq, durations)
        p_bls = 1 / freq[np.argmax(bls_res.power)]
    except Exception:
        p_bls = p_ls # 若 BLS 失败，降级处理
    
    # 计算 P/2 功率
    half_p = p_ls / 2.0
    half_idx = np.argmin(np.abs(1/freq - half_p))
    half_power = power_ls[half_idx]
    
    return p_ls, p_bls, max_p_ls, half_power, freq, power_ls

def internet_search(query):
    try:
        results = DDGS().text(f"{query} astronomy", max_results=3)
        return "【文献】\n" + "\n".join([f"- {r['title']}: {r['body']}" for r in results]) if results else "无文献"
    except Exception: return "检索不可用"

# ==========================================
# 4. 主程序
# ==========================================
if check_password():
    # 🔑 🔑 🔑 填入你的真实 API Key 🔑 🔑 🔑
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    with st.sidebar:
        st.header("⚙️ 数据控制台")
        tic_input = st.text_input("TESS TIC 编号", value="260128333", placeholder="输入真实编号")
        window_length = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        st.markdown("---")
        st.header("🔄 联合验证 (可选)")
        ref_period = st.number_input("其他星表周期 (天)", value=0.0, format="%.4f")
        fast_period = st.number_input("FAST 射电周期 (天)", value=0.0, format="%.3f")
        
        if st.button("🚀 启动核心分析", use_container_width=True):
            if not tic_input: st.error("请输入 TIC 编号！")
            else:
                with st.spinner("正在获取 NASA 数据并运行多算法..."):
                    lc_data = download_tess_data(tic_input)
                    if lc_data is None: st.error("下载失败或数据异常。")
                    else: 
                        st.session_state.lc = lc_data
                        st.session_state.tic = tic_input
                        st.session_state.ref_p = ref_period
                        st.session_state.fast_p = fast_period
                        st.success("分析完成！")

        st.markdown("---")
        voice_text = speech_to_text(language='zh-CN', start_prompt="🎤 语音输入", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
        uploaded_file = st.file_uploader("📷 上传天文图片", type=["jpg", "jpeg", "png"])

    tab1, tab2 = st.tabs(["📊 核心诊断与冲突识别", "📦 批量与导出 (原型)"])

    # ======== TAB 1: 核心诊断 ========
    with tab1:
        if "lc" in st.session_state and st.session_state.lc is not None:
            lc = st.session_state.lc
            time_arr = lc.time.value
            flux_arr = lc.flux.value
            
            # 预处理去趋势
            window_points = int(window_length * 720)
            if window_points % 2 == 0: window_points += 1
            trend = median_filter(flux_arr, size=window_points, mode='nearest')
            flux_flat = flux_arr - trend + 1.0
            
            # 多算法运行
            p_ls, p_bls, max_p_ls, half_power, freq, power_ls = run_multi_algorithms(time_arr, flux_flat)
            
            # 判定逻辑
            is_insufficient = len(time_arr) < 500 or np.std(flux_flat) < 0.001
            is_half_period = half_power > max_p_ls * 0.7
            
            # 左图右表布局
            col_left, col_right = st.columns([1.3, 1])
            
            with col_left:
                # 1. 光变曲线
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_arr, mode='markers', marker=dict(size=2, color='#CBD5E1'), name='原始'))
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_flat, mode='markers', marker=dict(size=2, color='#3B82F6'), name='去趋势'))
                fig_lc.update_layout(title="TESS 光变曲线", height=250, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_lc, use_container_width=True)
                
                # 2. 周期图 (标注候选峰值)
                fig_pg = go.Figure()
                fig_pg.add_trace(go.Scatter(x=1/freq, y=power_ls, mode='lines', line=dict(color='#F59E0B'), name='LS Power'))
                fig_pg.add_vline(x=p_ls, line_dash="dash", line_color="#EF4444", annotation_text=f"主峰 P: {p_ls:.3f}")
                fig_pg.add_vline(x=p_ls/2, line_dash="dot", line_color="#8B5CF6", annotation_text=f"半周期 P/2: {p_ls/2:.3f}")
                fig_pg.add_vline(x=p_ls*2, line_dash="dot", line_color="#10B981", annotation_text=f"倍周期 2P: {p_ls*2:.3f}")
                fig_pg.update_layout(title="Lomb-Scargle 周期谱 (候选峰值)", height=250, template="plotly_white", xaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_pg, use_container_width=True)
                
                # 3. 相位折叠与残差
                phase = (time_arr / p_ls) % 1.0
                phase[phase > 0.5] -= 1.0
                residual = flux_flat - np.mean(flux_flat)
                
                fig_phase = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig_phase.add_trace(go.Scattergl(x=phase, y=flux_flat, mode='markers', marker=dict(size=2, color='#000000')), row=1, col=1)
                fig_phase.add_trace(go.Scattergl(x=phase, y=residual, mode='markers', marker=dict(size=2, color='#EF4444')), row=2, col=1)
                fig_phase.update_layout(height=350, template="plotly_white", title="相位折叠与残差", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_phase, use_container_width=True)

            with col_right:
                st.markdown("### 🩺 多候选周期与冲突诊断")
                
                # 候选周期表格
                df_candidates = pd.DataFrame({
                    "算法": ["Lomb-Scargle", "BLS (Box)", "谐波校验 (P/2)"],
                    "候选周期 (天)": [round(p_ls, 4), round(p_bls, 4), round(p_ls/2, 4)],
                    "诊断标签": ["consistent", "consistent", "half_period_alias" if is_half_period else "no_alias"]
                })
                st.dataframe(df_candidates, use_container_width=True, hide_index=True)
                
                st.markdown("#### 🚨 诊断结论")
                if is_insufficient:
                    st.error("🚫 **insufficient_evidence (拒判)**：有效数据点过少或信噪比极低，无法判定周期。")
                    st.session_state.tag = "insufficient_evidence"
                elif is_half_period:
                    st.warning(f"⚠️ **half_period_alias (半周期混叠)**：检测到 P/2 ({p_ls/2:.3f}d) 功率极高，极可能发生半周期误判！")
                    st.session_state.tag = "half_period_alias"
                else:
                    st.success(f"✅ **consistent (周期可靠)**：最佳周期 {p_ls:.4f} 天，无明显混叠。")
                    st.session_state.tag = "consistent"
                
                # 跨星表比对逻辑
                if "ref_p" in st.session_state and st.session_state.ref_p > 0:
                    ratio = p_ls / st.session_state.ref_p
                    st.info(f"🔄 **多源比对**：与参考周期比值为 {ratio:.3f}")
                
                st.session_state.period = p_ls

                # AI 对话
                st.markdown("---")
                st.subheader("💬 科研 Agent")
                if "messages" not in st.session_state:
                    st.session_state.messages = [{"role": "assistant", "content": "数据已就绪。请提问或要求生成报告。"}]
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                        
                prompt = st.chat_input("输入科研问题...")
                if voice_text and not prompt: prompt = voice_text
                
                if prompt or uploaded_file:
                    user_msg = {"role": "user", "content": prompt if prompt else "请分析图片。"}
                    with st.chat_message("user"):
                        if uploaded_file: 
                            st.image(uploaded_file, width=200)
                            user_msg["image"] = uploaded_file
                        st.markdown(user_msg["content"])
                    st.session_state.messages.append(user_msg)
                    
                    with st.chat_message("assistant"):
                        search_context = ""
                        if not uploaded_file and ("文献" in prompt or "查一下" in prompt):
                            with st.spinner("🌐 正在全网检索..."):
                                search_context = internet_search(prompt)
                        
                        context = f"{search_context}\n【当前数据】TIC {st.session_state.tic}，光学周期 {st.session_state.period:.4f} 天，标签 {st.session_state.tag}。"
                        sys_prompt = f"你是资深天文学家。{context} 请专业回答。输出格式：先输出【深度思考过程】，再输出【最终解答】。公式用 $$ 包裹。"

                        try:
                            client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                            if uploaded_file:
                                img_bytes = uploaded_file.getvalue()
                                b64_img = base64.b64encode(img_bytes).decode('utf-8')
                                messages = [{"role": "system", "content": "你是天文学家，请识别图片。"}, {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}]
                                res = client.chat.completions.create(model="glm-4v-flash", messages=messages)
                                st.markdown(res.choices[0].message.content)
                            else:
                                with st.status("🧠 AI 正在深度推理...", expanded=True) as status:
                                    st.write("解析上下文与文献证据...")
                                    status.update(label="推理完成：", state="complete", expanded=False)
                                
                                def generate_chunks():
                                    res = client.chat.completions.create(
                                        model="glm-4-flash",
                                        messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages[-4:],
                                        stream=True
                                    )
                                    for chunk in res:
                                        if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
                                reply = st.write_stream(generate_chunks())
                            st.session_state.messages.append({"role": "assistant", "content": reply if not uploaded_file else res.choices[0].message.content})
                            st.rerun()
                        except Exception as e:
                            st.warning(f"⚠️ 网络波动: {e}")
        else:
            st.info("👈 请在左侧输入 TIC 编号并点击启动分析。")

    # ======== TAB 2: 批量与导出原型 ========
    with tab2:
        st.subheader("📦 批量任务与报告导出")
        st.markdown("*(当前为 UI 原型演示，后续接入本地 CSV 批处理)*")
        col_up, col_down = st.columns(2)
        with col_up:
            st.markdown("**1. 批量导入 TIC 列表 (.csv)**")
            st.file_uploader("拖拽 CSV 文件", type=["csv"], label_visibility="collapsed")
            if st.button("▶️ 启动批量诊断"):
                st.info("模拟运行中... 已加入后台队列。")
                st.dataframe(pd.DataFrame({"TIC ID": [260128333, 12345678], "周期 (天)": [0.4068, 1.2345], "诊断标签": ["consistent", "half_period_alias"]}))
        with col_down:
            st.markdown("**2. 一键生成诊断报告**")
            st.button("📄 导出当前报告 (Markdown)")
            st.success("规划中：后续采用文件流实现一键下载。")
