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
# 1. 页面设置：浅色、高效双栏
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
        st.title("🔒 星鉴 AstroFusion 科研平台")
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
# 3. 核心功能与图像搜索
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
    except Exception: return None

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
    """全网文献搜索"""
    try:
        results = DDGS().text(f"{query} astronomy", max_results=3)
        return "【文献】\n" + "\n".join([f"- {r['title']}: {r['body']}" for r in results]) if results else "无文献"
    except Exception: return "检索不可用"

def image_search(query):
    """全网真实图像检索"""
    try:
        # 增加 astronomy 关键词确保图片相关性
        results = DDGS().images(f"{query} astronomy", max_results=2)
        return [r['image'] for r in results] if results else []
    except Exception:
        return []

# ==========================================
# 4. 主程序
# ==========================================
if check_password():
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    with st.sidebar:
        st.header("⚙️ 数据控制台")
        tic_input = st.text_input("TESS TIC 编号", value="", placeholder="例如: 260128333")
        window_length = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        st.markdown("---")
        st.header("🔄 联合验证")
        ref_period = st.number_input("其他星表周期 (天)", value=0.0, format="%.4f")
        fast_period = st.number_input("FAST 射电周期 (天)", value=0.0, format="%.3f")
        
        if st.button("🚀 启动核心分析", use_container_width=True):
            if not tic_input:
                st.error("请输入 TIC 编号！")
            else:
                with st.spinner("从 NASA 获取数据..."):
                    lc_data = download_tess_data(tic_input)
                    if lc_data is None: st.error("下载失败。")
                    else: st.session_state.lc = lc_data; st.success("数据就绪！")

        st.markdown("---")
        voice_text = speech_to_text(language='zh-CN', start_prompt="🎤 语音输入", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
        uploaded_file = st.file_uploader("📷 上传天文图片", type=["jpg", "jpeg", "png"])

    # ======== 左右分屏 ========
    col_chart, col_chat = st.columns([1.2, 1])

    # ---------- 左侧：科研诊断区 ----------
    with col_chart:
        if "lc" in st.session_state and st.session_state.lc is not None:
            lc = st.session_state.lc
            time_arr = lc.time.value
            flux_arr = lc.flux.value
            
            window_points = int(window_length * 720)
            if window_points % 2 == 0: window_points += 1
            trend = median_filter(flux_arr, size=window_points, mode='nearest')
            flux_flat = flux_arr - trend + 1.0
            
            frequency, power = LombScargle(time_arr, flux_flat - np.mean(flux_flat)).autopower(minimum_frequency=0.1, maximum_frequency=10)
            period = 1 / frequency
            best_period = period[np.argmax(power)]
            max_power = np.max(power)
            
            half_p = best_period / 2.0
            half_power = power[np.argmin(np.abs(period - half_p))]
            
            # 绘图
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_arr, mode='markers', marker=dict(size=2, color='gray'), name='原始'))
            fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_flat, mode='markers', marker=dict(size=2, color='#3B82F6'), name='去趋势'))
            fig_lc.update_layout(title=f"光变曲线", height=250, template="plotly_white")
            st.plotly_chart(fig_lc, use_container_width=True)
            
            fig_pg = go.Figure()
            fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', line=dict(color='#F59E0B')))
            fig_pg.update_layout(title="周期图", height=250, template="plotly_white", xaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_pg, use_container_width=True)
            
            # 诊断标签
            st.markdown("### 🩺 诊断报告")
            if len(time_arr) < 500:
                st.error("🚫 **insufficient_evidence**：数据点过少，拒判。")
                st.session_state.tag = "insufficient_evidence"
            elif half_power > max_power * 0.75:
                st.warning(f"⚠️ **half_period**：半周期 ({half_p:.3f}d) 处功率过高，混叠风险大！")
                st.session_state.tag = "half_period"
            else:
                st.success(f"✅ **consistent**：最佳周期 {best_period:.4f} 天，无异常。")
                st.session_state.tag = "consistent"
                
            # 相位折叠图
            st.markdown("### 🔭 相位折叠图")
            phase = (time_arr / best_period) % 1.0
            phase[phase > 0.5] -= 1.0 
            fig_phase = go.Figure()
            fig_phase.add_trace(go.Scattergl(x=phase, y=flux_arr, mode='markers', marker=dict(size=2, color='#000000')))
            fig_phase.update_layout(height=250, template="plotly_white", xaxis_title="相位", yaxis_title="流量")
            st.plotly_chart(fig_phase, use_container_width=True)
            
            st.session_state.period = best_period
        else:
            st.info("👈 请在左侧输入参数并启动分析。")

    # ---------- 右侧：AI 视觉对话区 ----------
    with col_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！我是星鉴 AI。你可以问我任何天文问题，**如果需要图片，请直接告诉我（例如：给我一张太阳光谱图）**。"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                # 如果历史记录里包含图片 URL 列表，渲染出来
                if "image_urls" in msg and msg["image_urls"]:
                    cols = st.columns(len(msg["image_urls"]))
                    for i, url in enumerate(msg["image_urls"]):
                        with cols[i]: st.image(url, use_column_width=True)
                st.markdown(msg["content"])
                
        prompt = st.chat_input("输入科研问题或索要图片...")
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
                img_urls = []
                
                # 1. 智能判断：用户是不是在要图片？
                img_keywords = ["图", "图片", "照片", "影像", "画"]
                if not uploaded_file and any(k in prompt for k in img_keywords):
                    with st.spinner("🌐 正在全网抓取天文图像..."):
                        img_urls = image_search(prompt)
                        if img_urls:
                            cols = st.columns(len(img_urls))
                            for i, url in enumerate(img_urls):
                                with cols[i]: st.image(url, use_column_width=True)
                            search_context += "\n【系统提示：已经为用户实时展示相关图片】"
                
                # 2. 联网文献检索
                elif not uploaded_file and ("文献" in prompt or "查一下" in prompt):
                    with st.spinner("🌐 正在检索全网文献..."):
                        search_context = internet_search(prompt)
                
                context = f"{search_context}\n"
                if "period" in st.session_state:
                    context += f"【当前TIC数据】光学周期 {st.session_state.period:.4f} 天，标签 {st.session_state.tag}。"

                sys_prompt = f"你是资深天文学家。{context} 请专业回答。输出格式：先输出【深度思考过程】，再输出【最终解答】。公式用 $$ 包裹。"

                try:
                    client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                    if uploaded_file:
                        with st.spinner("识别中..."):
                            img_bytes = uploaded_file.getvalue()
                            b64_img = base64.b64encode(img_bytes).decode('utf-8')
                            messages = [
                                {"role": "system", "content": "你是资深天文学家，请识别图片。"},
                                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}
                            ]
                            res = client.chat.completions.create(model="glm-4v-flash", messages=messages)
                            reply = res.choices[0].message.content
                            st.markdown(reply)
                    else:
                        with st.status("🧠 AI 正在深度推理...", expanded=True) as status:
                            st.write("解析上下文与图像证据...")
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
                        
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reply if not uploaded_file else res.choices[0].message.content,
                        "image_urls": img_urls  # 保存图片URL以便历史记录回显
                    })
                    st.rerun()
                except Exception as e:
                    st.warning(f"⚠️ 网络波动: {e}")
