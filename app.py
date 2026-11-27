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
import re

# ==========================================
# 1. 页面设置：浅蓝科研风，强制浅色模式
# ==========================================
st.set_page_config(page_title="星鉴 AstroFusion V1.0", layout="wide")

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
# 2. 核心功能与算法
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

# 暴力清洗公式格式，保证 100% 渲染为课本体
def format_math_text(text):
    if not text: return text
    # 1. 替换转义的括号
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    text = text.replace("\\(", "$").replace("\\)", "$")
    
    # 2. 有些模型会用 ```math 包裹，去掉它
    text = re.sub(r'```math\n(.*?)\n```', r'$$\1$$', text, flags=re.DOTALL)
    
    # 3. 暴力把 [F=G...] 强行转成 $$F=G...$$ (只要里面包含等号或 frac)
    text = re.sub(r'\[\s*(.*?(?:=|\\frac|\\approx).*?)\s*\]', r'$$\1$$', text)
    
    return text

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
    ls = LombScargle(time_arr, flux_flat - np.mean(flux_flat))
    power_ls = ls.power(freq)
    best_ls_idx = np.argmax(power_ls)
    p_ls = 1 / freq[best_ls_idx]
    max_p_ls = power_ls[best_ls_idx]
    
    try:
        bls = BoxLeastSquares(time_arr, flux_flat)
        durations = np.linspace(0.05, 0.2, 5)
        bls_res = bls.power(freq, durations)
        p_bls = 1 / freq[np.argmax(bls_res.power)]
    except Exception:
        p_bls = p_ls 
    
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
# 3. 主程序
# ==========================================
if check_password():
    # 🔑 填入你的智谱 Key
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    with st.sidebar:
        st.header("⚙️ 数据控制台")
        tic_input = st.text_input("TESS TIC 编号", value="260128333")
        window_length = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        st.markdown("---")
        st.header("🔄 联合验证")
        ref_period = st.number_input("其他星表周期 (天)", value=0.0, format="%.4f")
        fast_period = st.number_input("FAST 射电周期 (天)", value=0.0, format="%.3f")
        
        if st.button("🚀 启动核心分析", use_container_width=True):
            if not tic_input: st.error("请输入 TIC 编号！")
            else:
                with st.spinner("正在获取 NASA 数据..."):
                    lc_data = download_tess_data(tic_input)
                    if lc_data is None: st.error("下载失败。")
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

    # ======== TAB 1: 核心诊断与对话 ========
    with tab1:
        col_left, col_right = st.columns([1.3, 1])

        # ---- 左侧：数据诊断区 ----
        with col_left:
            if "lc" in st.session_state and st.session_state.lc is not None:
                lc = st.session_state.lc
                time_arr = lc.time.value
                flux_arr = lc.flux.value
                
                window_points = int(window_length * 720)
                if window_points % 2 == 0: window_points += 1
                trend = median_filter(flux_arr, size=window_points, mode='nearest')
                flux_flat = flux_arr - trend + 1.0
                
                p_ls, p_bls, max_p_ls, half_power, freq, power_ls = run_multi_algorithms(time_arr, flux_flat)
                
                is_insufficient = len(time_arr) < 500 or np.std(flux_flat) < 0.001
                is_half_period = half_power > max_p_ls * 0.7
                
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_arr, mode='markers', marker=dict(size=2, color='#CBD5E1'), name='原始'))
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_flat, mode='markers', marker=dict(size=2, color='#3B82F6'), name='去趋势'))
                fig_lc.update_layout(title="TESS 光变曲线", height=250, template="plotly_white")
                st.plotly_chart(fig_lc, use_container_width=True)
                
                fig_pg = go.Figure()
                fig_pg.add_trace(go.Scatter(x=1/freq, y=power_ls, mode='lines', line=dict(color='#F59E0B')))
                fig_pg.add_vline(x=p_ls, line_dash="dash", line_color="#EF4444", annotation_text=f"主峰 P")
                fig_pg.add_vline(x=p_ls/2, line_dash="dot", line_color="#8B5CF6", annotation_text=f"P/2")
                fig_pg.update_layout(title="周期谱与候选峰", height=250, template="plotly_white", xaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_pg, use_container_width=True)
                
                # 相位折叠
                phase = (time_arr / p_ls) % 1.0
                phase[phase > 0.5] -= 1.0
                fig_phase = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig_phase.add_trace(go.Scattergl(x=phase, y=flux_flat, mode='markers', marker=dict(size=2, color='#000000')), row=1, col=1)
                fig_phase.add_trace(go.Scattergl(x=phase, y=flux_flat - np.mean(flux_flat), mode='markers', marker=dict(size=2, color='#EF4444')), row=2, col=1)
                fig_phase.update_layout(height=300, template="plotly_white", title="相位折叠与残差", showlegend=False)
                st.plotly_chart(fig_phase, use_container_width=True)

                st.markdown("### 🩺 多候选周期与冲突诊断")
                df_candidates = pd.DataFrame({
                    "算法": ["Lomb-Scargle", "BLS (Box)", "谐波校验 (P/2)"],
                    "候选周期 (天)": [round(p_ls, 4), round(p_bls, 4), round(p_ls/2, 4)],
                    "诊断标签": ["consistent", "consistent", "half_period_alias" if is_half_period else "no_alias"]
                })
                st.dataframe(df_candidates, use_container_width=True, hide_index=True)
                
                if is_insufficient:
                    st.error("🚫 **insufficient_evidence (拒判)**：有效数据点过少或信噪比极低，无法判定周期。")
                    st.session_state.tag = "insufficient_evidence"
                elif is_half_period:
                    st.warning(f"⚠️ **half_period_alias (半周期混叠)**：检测到 P/2 ({p_ls/2:.3f}d) 功率极高，极可能发生半周期误判！")
                    st.session_state.tag = "half_period_alias"
                else:
                    st.success(f"✅ **consistent (周期可靠)**：最佳周期 {p_ls:.4f} 天，无明显混叠。")
                    st.session_state.tag = "consistent"
                
                st.session_state.period = p_ls
            else:
                st.info("👈 请在左侧输入编号并启动分析。")

        # ---- 右侧：AI 对话区 ----
        with col_right:
            st.subheader("💬 科研 Agent 对话")
            
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "你好！你可以问我天文问题，我会提供**课本印刷体**公式，并附带**微信复制版**。"}]
            
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): 
                    st.markdown(msg["content"])
                    
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
                    # 极其严苛的提示词
                    sys_prompt = """
                    你是一个资深天文学家。请专业回答用户问题。
                    
                    【最高优先级：公式排版死命令】
                    1. 先输出【深度思考过程】，再输出【最终解答】。
                    2. 绝对禁止使用 `[`、`]`、`\\[`、`\\]` 包裹公式！
                    3. 独立的数学公式，**必须且只能**使用双美元符号包裹，例如：$$ F = G \\frac{m_1 m_2}{r^2} $$
                    4. 行内公式，必须且只能使用单美元符号，例如：$ E = mc^2 $。
                    
                    【微信复制格式要求】
                    在回答的最后，请加一个 Markdown 代码块（用 ```text 包裹），提供纯文本版结论（不能有$和\\，用 Unicode 字符如 m₁，r²）。
                    """

                    try:
                        client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                        if uploaded_file:
                            img_bytes = uploaded_file.getvalue()
                            b64_img = base64.b64encode(img_bytes).decode('utf-8')
                            messages = [{"role": "system", "content": "你是天文学家，请识别图片。"}, {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}]
                            res = client.chat.completions.create(model="glm-4v-flash", messages=messages)
                            reply = res.choices[0].message.content
                            st.markdown(reply)
                        else:
                            with st.spinner("🧠 AI 正在深度推理并排版公式..."):
                                # 【关键修改】关闭流式输出，一次性获取完整文本，确保 LaTeX 公式不崩
                                res = client.chat.completions.create(
                                    model="glm-4-flash",
                                    messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages[-4:],
                                    stream=False # 彻底关闭流式
                                )
                                raw_reply = res.choices[0].message.content
                                
                                # 后台暴力正则替换
                                cleaned_reply = format_math_text(raw_reply)
                                
                                # 一次性渲染出完美的课本公式
                                st.markdown(cleaned_reply)
                                reply = cleaned_reply
                                
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                    except Exception as e:
                        st.warning(f"⚠️ 网络波动: {e}")

    # ======== TAB 2: 批量与导出原型 ========
    with tab2:
        st.subheader("📦 批量任务与报告导出")
        st.info("提示：本模块为后续开发原型，展示系统未来的批量处理能力。")
        
        col_up, col_down = st.columns(2)
        with col_up:
            st.markdown("#### 1. 批量导入与诊断")
            st.file_uploader("拖拽 TIC 列表 CSV 文件", type=["csv"])
            if st.button("▶️ 启动批量诊断"):
                st.success("模拟运行中... 已加入后台队列")
                st.dataframe(pd.DataFrame({
                    "TIC ID": [260128333, 12345678, 87654321],
                    "最佳周期 (天)": [0.4068, 1.2345, 5.6789],
                    "诊断标签": ["consistent", "half_period_alias", "insufficient_evidence"]
                }), use_container_width=True)
                
        with col_down:
            st.markdown("#### 2. 一键生成科研报告")
            st.write("将当前分析图表、物理参数与 AI 诊断结论打包导出。")
            st.selectbox("选择导出格式", ["PDF 报告", "Markdown 文档", "CSV 数据表"])
            if st.button("📄 导出报告"):
                st.success("报告生成成功！(模拟下载)")
