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
# 1. 极简黑白 UI (强制关闭花哨样式)
# ==========================================
st.set_page_config(page_title="星鉴 AstroFusion", layout="wide")

st.markdown("""
    <style>
    /* 强制浅色模式，纯白背景 */
    :root { color-scheme: light !important; }
    .stApp { background-color: #FFFFFF !important; font-family: 'Helvetica Neue', Arial, sans-serif; }
    
    /* 隐藏所有多余组件 */
    header { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* 侧边栏：白底，黑边 */
    [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #E5E5E5; }
    
    /* 输入框、按钮：纯黑白扁平化 */
    div[data-baseweb="input"], div[data-baseweb="select"], input, textarea { 
        background-color: #FFFFFF !important; border-radius: 0 !important; 
        border: 1px solid #000000 !important; color: #000000 !important; 
    }
    .stButton>button { 
        background-color: #000000 !important; color: #FFFFFF !important; 
        border-radius: 0 !important; border: 1px solid #000000 !important; 
    }
    .stButton>button:hover { background-color: #333333 !important; }
    
    /* 聊天消息区块：去掉背景和阴影，只用黑线分隔 */
    [data-testid="stChatMessage"] { 
        background-color: transparent !important; border-radius: 0 !important; 
        box-shadow: none !important; border-bottom: 1px dashed #CCCCCC !important;
        padding: 10px 0 !important; 
    }
    
    /* 底部输入框 */
    [data-testid="stChatInput"] { 
        background-color: #FFFFFF !important; border-radius: 0 !important; 
        border: 1px solid #000000 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 密码锁
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.title("🔒 星鉴 AstroFusion")
        st.markdown("请输入访问码（默认密码：**123456**）")
        pwd = st.text_input("访问码", type="password")
        if st.button("进入系统", use_container_width=True):
            if pwd == "123456":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("密码错误！")
        return False
    return True

# ==========================================
# 3. 核心函数
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
        context = "【联网检索资料】\n"
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

    # ======== 侧边栏：极简控制台 ========
    with st.sidebar:
        st.markdown("### ⚙️ 参数设置")
        tic_input = st.text_input("TESS TIC 编号", value="", placeholder="例如: 260128333")
        window_length = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        st.markdown("---")
        st.markdown("### 🔄 联合验证")
        ref_period = st.number_input("参考星表周期 (天)", value=0.0, step=0.001, format="%.4f")
        fast_period = st.number_input("FAST 射电周期 (天)", value=0.0, step=0.001, format="%.3f")
        
        if st.button("🚀 启动分析", use_container_width=True):
            if not tic_input:
                st.error("请输入 TIC 编号")
            else:
                with st.spinner("数据获取中..."):
                    lc_data = download_tess_data(tic_input)
                    if lc_data is None:
                        st.error("下载失败")
                    else:
                        st.session_state.lc = lc_data
                        st.session_state.current_tic = tic_input
                        st.session_state.messages.append({"role": "assistant", "content": "✅ 数据已就绪，生成图表与诊断报告："})
                        st.rerun()

        st.markdown("---")
        st.markdown("### 🛠️ 辅助工具")
        voice_text = speech_to_text(language='zh-CN', start_prompt="🎤 语音输入", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
        uploaded_file = st.file_uploader("📷 上传图片", type=["jpg", "jpeg", "png"])
        
        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "系统就绪。"}]
            st.rerun()

    # ======== 主界面：左右分屏 (左图右聊) ========
    col_chart, col_chat = st.columns([1.2, 1])

    # 左侧：图表与诊断
    with col_chart:
        if "lc" in st.session_state and st.session_state.lc is not None:
            lc = st.session_state.lc
            time_arr = lc.time.value
            flux_arr = lc.flux.value
            
            window_points = int(window_length * 720)
            if window_points % 2 == 0: window_points += 1
            if window_points >= len(flux_arr): window_points = max(5, len(flux_arr) - 1)
            if window_points % 2 == 0: window_points += 1
            
            trend = median_filter(flux_arr, size=window_points, mode='nearest')
            flux_flat = flux_arr - trend + 1.0
            mask_plot = time_arr < (time_arr[0] + 5.0)
            
            # 1. 光变曲线 (纯黑白)
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scattergl(x=time_arr[mask_plot], y=flux_arr[mask_plot], mode='markers', marker=dict(size=2, color='#CCCCCC'), name='原始观测'))
            fig_lc.add_trace(go.Scattergl(x=time_arr[mask_plot], y=flux_flat[mask_plot], mode='markers', marker=dict(size=2, color='#000000'), name='去趋势后'))
            fig_lc.update_layout(title=f"TIC {st.session_state.current_tic} 光变曲线", height=280, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_lc, use_container_width=True)
            
            # 2. 周期图
            flux_centered = flux_flat - np.mean(flux_flat)
            if not np.all(flux_centered == 0):
                frequency, power = LombScargle(time_arr, flux_centered).autopower(minimum_frequency=0.1, maximum_frequency=10)
                period = 1 / frequency
                best_period = period[np.argmax(power)]
                max_power = np.max(power)
                
                fig_pg = go.Figure()
                fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', line=dict(color='#000000', width=1)))
                fig_pg.add_vline(x=best_period, line_dash="dash", line_color="#FF0000")
                fig_pg.update_layout(title="Lomb-Scargle 周期图", height=280, template="plotly_white", xaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_pg, use_container_width=True)
                
                # 3. 诊断文字
                st.markdown(f"**最高峰周期：** {best_period:.4f} 天")
                
                if ref_period > 0:
                    ratio = best_period / ref_period
                    st.markdown(f"**多源比对：** 比值 {ratio:.2f} (参考: {ref_period})")
                if fast_period > 0:
                    fast_ratio = best_period / fast_period
                    st.markdown(f"**FAST 验证：** 比值 {fast_ratio:.2f} (射电: {fast_period})")
                    
                params = estimate_physical_params(best_period)
                if params:
                    st.markdown(f"**物理参数反演：** 质量 {params['m']} M☉ | 温度 {params['t']} K | 半径 {params['r']} R☉")
                
                st.session_state.period = best_period
        else:
            st.info("等待数据输入...")

    # 右侧：对话流
    with col_chat:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "系统就绪。请在左侧输入参数并启动分析，或直接提问。"}]

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if "image" in msg: st.image(msg["image"], width=200)
                st.markdown(msg["content"])

        prompt = st.chat_input("输入指令或问题...")
        if not prompt and voice_text:
            prompt = voice_text

        if prompt or uploaded_file:
            user_msg = {"role": "user", "content": prompt if prompt else "分析图片。"}
            with st.chat_message("user"):
                if uploaded_file: 
                    st.image(uploaded_file, width=200)
                    user_msg["image"] = uploaded_file
                st.markdown(user_msg["content"])
            st.session_state.messages.append(user_msg)
            
            with st.chat_message("assistant"):
                search_context = ""
                if not uploaded_file and ("文献" in prompt or "查一下" in prompt):
                    search_context = internet_search(prompt)
                
                context = f"{search_context}\n"
                if "period" in st.session_state:
                    context += f"当前周期 {st.session_state.period:.4f} 天。"

                sys_prompt = f"你是资深天文学家。{context} 请用最简洁、专业的语言回答。严禁废话。"

                try:
                    client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                    if uploaded_file:
                        img_bytes = uploaded_file.getvalue()
                        b64_img = base64.b64encode(img_bytes).decode('utf-8')
                        messages = [
                            {"role": "system", "content": "你是一个天文图像识别器，简短回答。"},
                            {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}]}
                        ]
                        res = client.chat.completions.create(model="glm-4v-flash", messages=messages)
                        st.markdown(res.choices[0].message.content)
                        st.session_state.messages.append({"role": "assistant", "content": res.choices[0].message.content})
                    else:
                        def generate_chunks():
                            res = client.chat.completions.create(
                                model="glm-4-flash",
                                messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages[-4:],
                                stream=True
                            )
                            for chunk in res:
                                if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content
                        reply = st.write_stream(generate_chunks())
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
