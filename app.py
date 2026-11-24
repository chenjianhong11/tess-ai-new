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
# 1. 页面设置与浅蓝色 UI 定制
# ==========================================
st.set_page_config(page_title="星鉴 AstroFusion 科研平台", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F0F8FF; }
    [data-testid="stSidebar"] { background-color: #E6F2FF; }
    div[data-testid="stExpander"] { background-color: #FFFFFF; border-radius: 10px; }
    .stButton>button { background-color: #4A90E2; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #357ABD; color: white; }
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

    with st.sidebar:
        st.header("🔭 数据控制台 (空白待命)")
        # 默认为空，不再预设 260128333
        tic_input = st.text_input("输入 TESS TIC 编号", value="", placeholder="例如: 260128333")
        window_length = st.slider("去趋势滑动窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        st.markdown("---")
        st.header("🔄 多源周期比对")
        # 默认 0.0
        ref_period = st.number_input("输入其他星表周期 (天)", value=0.0, step=0.001, format="%.4f")
        
        st.markdown("---")
        st.header("📡 天眼 (FAST) 联合验证")
        # 默认 0.0
        fast_period = st.number_input("输入 FAST 射电周期 (天)", value=0.0, step=0.001, format="%.3f")
        
        if st.button("🚀 获取 NASA 数据并分析", use_container_width=True):
            if not tic_input:
                st.error("请先输入 TESS TIC 编号！")
            else:
                with st.spinner("正在从 NASA 下载真实数据..."):
                    lc_data = download_tess_data(tic_input)
                    if lc_data is None:
                        st.error("下载失败或未找到数据，请检查编号。")
                        st.session_state.lc = None
                    else:
                        st.session_state.lc = lc_data
                        st.success("数据下载成功！")

    tab1, tab2 = st.tabs(["📊 真实观测与智能诊断", "💬 AI Agent 深度对话"])

    # ==== 标签页 1：科研绘图区 ====
    with tab1:
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
                
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scattergl(x=time_arr[mask_plot], y=flux_arr[mask_plot], mode='markers', marker=dict(size=3, color='rgba(160, 160, 160, 0.5)'), name='原始观测'))
                fig_lc.add_trace(go.Scattergl(x=time_arr[mask_plot], y=flux_flat[mask_plot], mode='markers', marker=dict(size=3, color='#4A90E2'), name='去趋势后'))
                fig_lc.update_layout(title=f"TIC {st.session_state.tic_id if 'tic_id' in st.session_state else ''} 光变曲线", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.6)')
                st.plotly_chart(fig_lc, use_container_width=True)
                
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
                    fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', line=dict(color='#FF7F0E', width=2), name='功率谱'))
                    fig_pg.add_vline(x=best_period, line_dash="dash", line_color="#EF553B", annotation_text=f"主峰: {best_period:.4f}天")
                    fig_pg.update_layout(title="Lomb-Scargle 周期图", height=300, xaxis=dict(autorange="reversed"), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(255,255,255,0.6)')
                    st.plotly_chart(fig_pg, use_container_width=True)
                    
                    st.markdown("### 🩺 智能诊断报告")
                    if half_power > max_power * 0.7:
                        st.error(f"⚠️ **谐波误判预警**：半周期处功率极高，存在 P/2 误判风险。")
                    else:
                        st.success(f"✅ **内部判定可靠**：最高峰周期为 {best_period:.4f} 天。")
                        
                    st.markdown("### 🔄 多源与 FAST 比对")
                    if ref_period > 0:
                        ratio = best_period / ref_period
                        if 0.95 <= ratio <= 1.05:
                            st.success(f"🟢 **跨源一致**：与参考星表吻合。")
                        elif 0.45 <= ratio <= 0.55 or 1.9 <= ratio <= 2.1:
                            st.warning(f"🟡 **跨源冲突**：疑似发生倍频偏差。")
                        else:
                            st.error(f"🔴 **严重冲突**。")
                    if fast_period > 0:
                        fast_ratio = best_period / fast_period
                        if 0.98 <= fast_ratio <= 1.02:
                            st.success(f"🌟 **光射一致**：与 FAST 周期吻合！")
                        elif 0.48 <= fast_ratio <= 0.52 or 1.95 <= fast_ratio <= 2.05:
                            st.error(f"🚨 **光射冲突**：应以 FAST 射电数据为准。")
                            
                    params = estimate_physical_params(best_period)
                    if params:
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
                st.error("数据处理异常。")
        else:
            st.info("👈 请在左侧输入真实的 TIC 编号并点击「获取 NASA 数据并分析」。")

    # ==== 标签页 2：AI 对话与语音 ====
    with tab2:
        st.subheader("🎙️ 多模态科研 Agent")
        
        voice_text = speech_to_text(language='zh-CN', start_prompt="🎤 按住说话", stop_prompt="⏹️ 停止", just_once=True, use_container_width=True)
        if voice_text:
            st.info(f"🎤 识别到语音：{voice_text}")
            
        uploaded_file = st.file_uploader("📷 上传天文图片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        prompt = st.chat_input("输入科研问题...")
        if voice_text and not prompt: prompt = voice_text
            
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！我是星鉴 AI。你可以提问，或让我检索全网文献。请在左侧先拉取真实数据。"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if "image" in msg: st.image(msg["image"], width=250)
                st.markdown(msg["content"])
                
        if prompt or uploaded_file:
            user_msg = {"role": "user", "content": prompt if prompt else "请分析这张图片。"}
            with st.chat_message("user"):
                if uploaded_file: 
                    st.image(uploaded_file, width=250)
                    user_msg["image"] = uploaded_file
                st.markdown(user_msg["content"])
            st.session_state.messages.append(user_msg)
            
            with st.chat_message("assistant"):
                search_context = ""
                if not uploaded_file and ("文献" in prompt or "查一下" in prompt):
                    with st.spinner("🌐 正在全网检索..."):
                        search_context = internet_search(prompt)
                
                context = f"{search_context}\n"
                if "period" in st.session_state:
                    context += f"【当前数据】TIC {st.session_state.tic}，光学周期 {st.session_state.period:.4f} 天，半周期功率比 {st.session_state.half_ratio*100:.1f}%。"
                    if st.session_state.ref_period > 0: context += f" 参考周期 {st.session_state.ref_period:.4f} 天。"
                    if st.session_state.fast_period > 0: context += f" FAST 周期 {st.session_state.fast_period} 天。"

                sys_prompt = f"你是资深天文学家。{context} 请专业回答。输出格式：先输出【深度思考过程】，再输出【最终解答】。公式用 $$ 包裹。"

                try:
                    client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                    if uploaded_file:
                        with st.spinner("视觉模型识别中..."):
                            img_bytes = uploaded_file.getvalue()
                            b64_img = base64.b64encode(img_bytes).decode('utf-8')
                            messages = [
                                {"role": "system", "content": "你是资深天文学家，请简短识别天文图片。"},
                                {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}
                            ]
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
