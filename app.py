import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import lightkurve as lk
from astropy.timeseries import LombScargle
from openai import OpenAI
import base64
from scipy.ndimage import median_filter
from duckduckgo_search import DDGS
from streamlit_mic_recorder import speech_to_text

# ==========================================
# 1. 页面设置：强制浅色、经典左右分栏 UI
# ==========================================
st.set_page_config(page_title="星鉴 AstroFusion 科研平台", layout="wide")

st.markdown("""
    <style>
    /* 强制浅色模式，防止系统深色干扰 */
    :root { color-scheme: light !important; }
    .stApp { background-color: #F4F7F9 !important; font-family: sans-serif; }
    
    /* 侧边栏强制白底 */
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E0E6ED; }
    
    /* 强制所有输入框为白底黑字 */
    div[data-baseweb="input"], div[data-baseweb="select"], input, textarea { 
        background-color: #FFFFFF !important; color: #1F2937 !important; border-radius: 8px !important; 
    }
    
    /* 美化聊天输入框 */
    div[data-testid="stChatInput"] { background-color: #FFFFFF !important; border-radius: 10px !important; border: 1px solid #D1D5DB !important; }
    
    /* 聊天气泡设计 */
    [data-testid="stChatMessage"] { background-color: #FFFFFF !important; border-radius: 10px !important; padding: 15px !important; border: 1px solid #E5E7EB; }
    
    /* 按钮 */
    .stButton>button { background-color: #3B82F6 !important; color: #FFFFFF !important; border-radius: 8px; border: none; font-weight: bold; }
    .stButton>button:hover { background-color: #2563EB !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 密码锁
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.title("🔒 星鉴 AstroFusion 科研平台")
        st.markdown("请输入访问码（默认密码：**123456**）")
        pwd = st.text_input("访问码", type="password")
        if st.button("进入平台", use_container_width=True):
            if pwd == "123456":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("密码错误！")
        return False
    return True

# ==========================================
# 3. 核心功能函数
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

def estimate_physical_params(period_days):
    try:
        a_au = (1.5 * (period_days / 365.25)**2)**(1/3)
        a_rsun = a_au * 215.0 
        radius = a_rsun * 0.2 
        temp = max(4000, min(5500 + (1.0 - period_days) * 1000, 8000))
        mass = max(0.1, min((radius / 1.0)**1.2, 5.0))
        return {"a": round(a_rsun, 2), "r": round(radius, 2), "t": round(temp, 0), "m": round(mass, 2)}
    except: return None

def internet_search(query):
    try:
        results = DDGS().text(f"{query} astronomy", max_results=3)
        if not results: return "未找到相关联网文献。"
        context = "【全网实时检索资料】\n"
        for r in results:
            context += f"- {r['title']}: {r['body']}\n"
        return context
    except Exception:
        return "（联网检索暂不可用）"

# ==========================================
# 4. 主程序
# ==========================================
if check_password():
    # 🔑 🔑 🔑 填入你的真实 API Key 🔑 🔑 🔑
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    # ======== 侧边栏：控制台 ========
    with st.sidebar:
        st.header("⚙️ 数据控制台")
        tic_input = st.text_input("输入 TESS TIC 编号", value="", placeholder="例如: 260128333")
        window_length = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        st.markdown("---")
        st.header("🔄 联合验证参数")
        ref_period = st.number_input("其他星表周期 (天)", value=0.0, step=0.001, format="%.4f")
        fast_period = st.number_input("FAST 射电周期 (天)", value=0.0, step=0.001, format="%.3f")
        
        if st.button("🚀 启动分析", use_container_width=True):
            if not tic_input:
                st.error("请先输入 TIC 编号！")
            else:
                with st.spinner("从 NASA 下载真实数据中..."):
                    lc_data = download_tess_data(tic_input)
                    if lc_data is None:
                        st.error("下载失败或未找到数据。")
                        st.session_state.lc = None
                    else:
                        st.session_state.lc = lc_data
                        st.success("数据下载成功！")

        st.markdown("---")
        st.header("🛠️ 辅助工具")
        # 语音输入
        voice_text = speech_to_text(language='zh-CN', start_prompt="🎤 按住说话", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
        # 图片上传
        uploaded_file = st.file_uploader("📷 上传天文照片", type=["jpg", "jpeg", "png"])
        
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "系统就绪。请在左侧输入参数进行诊断，或直接提问。"}]
            st.rerun()

    # ======== 主体界面：左右分栏 (左图右聊) ========
    col_chart, col_chat = st.columns([1.3, 1])

    # ---------- 左侧：图表与诊断区 ----------
    with col_chart:
        if "lc" in st.session_state and st.session_state.lc is not None:
            lc = st.session_state.lc
            try:
                time_arr = lc.time.value
                flux_arr = lc.flux.value
                
                window_points = int(window_length * 720)
                if window_points % 2 == 0: window_points += 1
                if window_points >= len(flux_arr): window_points = max(5, len(flux_arr) - 1)
                if window_points % 2 == 0: window_points += 1
                
                trend = median_filter(flux_arr, size=window_points, mode='nearest')
                flux_flat = flux_arr - trend + 1.0
                mask_plot = time_arr < (time_arr[0] + 5.0)
                
                # 1. 光变曲线
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scattergl(x=time_arr[mask_plot], y=flux_arr[mask_plot], mode='markers', marker=dict(size=3, color='rgba(160, 160, 160, 0.5)'), name='原始观测'))
                fig_lc.add_trace(go.Scattergl(x=time_arr[mask_plot], y=flux_flat[mask_plot], mode='markers', marker=dict(size=3, color='#3B82F6'), name='去趋势后'))
                fig_lc.update_layout(title=f"TIC {tic_input} 光变曲线", height=280, template="plotly_white", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_lc, use_container_width=True)
                
                # 2. 周期图
                flux_centered = flux_flat - np.mean(flux_flat)
                if not np.all(flux_centered == 0):
                    frequency, power = LombScargle(time_arr, flux_centered).autopower(minimum_frequency=0.1, maximum_frequency=10)
                    period = 1 / frequency
                    best_period = period[np.argmax(power)]
                    max_power = np.max(power)
                    
                    half_period = best_period / 2.0
                    half_idx = np.argmin(np.abs(period - half_period))
                    half_power = power[half_idx]
                    
                    fig_pg = go.Figure()
                    fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', line=dict(color='#F59E0B', width=2), name='功率谱'))
                    fig_pg.add_vline(x=best_period, line_dash="dash", line_color="#EF4444", annotation_text=f"主峰: {best_period:.4f}天")
                    fig_pg.update_layout(title="Lomb-Scargle 周期图", height=280, template="plotly_white", xaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_pg, use_container_width=True)
                    
                    # 3. 诊断报告
                    st.markdown("### 🩺 智能诊断报告")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if half_power > max_power * 0.7:
                            st.error(f"⚠️ **谐波误判预警**：半周期处功率极高，存在 P/2 误判风险。")
                        else:
                            st.success(f"✅ **内部判定可靠**：最高峰周期为 {best_period:.4f} 天。")
                    with col_b:
                        if ref_period > 0:
                            ratio = best_period / ref_period
                            if 0.95 <= ratio <= 1.05:
                                st.success(f"🟢 **跨源一致**：与参考星表吻合。")
                            elif 0.45 <= ratio <= 0.55 or 1.9 <= ratio <= 2.1:
                                st.warning(f"🟡 **跨源冲突**：疑似发生倍频偏差。")
                            else:
                                st.error(f"🔴 **严重冲突**。")
                        else:
                            st.info("ℹ️ 未输入其他星表周期。")
                            
                    if fast_period > 0:
                        fast_ratio = best_period / fast_period
                        if 0.98 <= fast_ratio <= 1.02:
                            st.success(f"🌟 **光射一致**：与 FAST 周期吻合！")
                        elif 0.48 <= fast_ratio <= 0.52 or 1.95 <= fast_ratio <= 2.05:
                            st.error(f"🚨 **光射冲突**：应以 FAST 射电数据为准修正周期。")
                            
                    params = estimate_physical_params(best_period)
                    if params:
                        st.markdown("### 🌟 智能反演物理参数")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("总质量 (M☉)", f"{params['m']}")
                        c2.metric("温度 (K)", f"{params['t']}")
                        c3.metric("半径 (R☉)", f"{params['r']}")
                        
                    st.session_state.period = best_period
                    st.session_state.half_ratio = half_power / max_power
                    st.session_state.tic = tic_input
                    st.session_state.ref_period = ref_period
                    st.session_state.fast_period = fast_period
            except Exception:
                st.error("数据处理异常，请尝试更换 TIC 编号。")
        else:
            st.info("👈 请在左侧输入真实的 TIC 编号并点击「启动分析」。")

    # ---------- 右侧：AI 对话区 ----------
    with col_chat:
        st.subheader("💬 多模态科研 Agent")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！请在左侧下载真实数据，然后问我关于周期、FAST 验证或物理参数的问题。"}]

        # 渲染历史记录
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if "image" in msg: st.image(msg["image"], width=200)
                st.markdown(msg["content"])
                
        # 底部输入框
        prompt = st.chat_input("输入科研问题...")
        if voice_text and not prompt: 
            prompt = voice_text

        if prompt or uploaded_file:
            user_msg = {"role": "user", "content": prompt if prompt else "请分析这张图片。"}
            with st.chat_message("user"):
                if uploaded_file: 
                    st.image(uploaded_file, width=200)
                    user_msg["image"] = uploaded_file
                st.markdown(user_msg["content"])
            st.session_state.messages.append(user_msg)
            
            with st.chat_message("assistant"):
                # 联网检索
                search_context = ""
                if not uploaded_file and ("文献" in prompt or "查一下" in prompt):
                    with st.spinner("🌐 正在全网检索..."):
                        search_context = internet_search(prompt)
                
                # 整合上下文
                context = f"{search_context}\n"
                if "period" in st.session_state:
                    context += f"【当前数据】TIC {st.session_state.tic}，光学周期 {st.session_state.period:.4f} 天，半周期功率比 {st.session_state.half_ratio*100:.1f}%。"
                    if st.session_state.ref_period > 0: context += f" 参考周期 {st.session_state.ref_period:.4f} 天。"
                    if st.session_state.fast_period > 0: context += f" FAST 周期 {st.session_state.fast_period} 天。"

                sys_prompt = f"你是资深天文学家。{context} 请专业回答。输出格式：先输出【深度思考过程】，再输出【最终解答】。公式用 $$ 包裹。"

                try:
                    client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                    if uploaded_file:
                        with st.spinner("AI 识别中..."):
                            img_bytes = uploaded_file.getvalue()
                            b64_img = base64.b64encode(img_bytes).decode('utf-8')
                            messages = [
                                {"role": "system", "content": "你是资深天文学家，请简短识别天文图片。"},
                                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}
                            ]
                            res = client.chat.completions.create(model="glm-4v-flash", messages=messages)
                            st.markdown(res.choices[0].message.content)
                    else:
                        # 深度思考动画
                        with st.status("🧠 AI 正在进行深度推理...", expanded=True) as status:
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
