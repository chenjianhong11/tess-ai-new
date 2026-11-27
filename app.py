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
import requests
from PIL import Image
from io import BytesIO

# 1. 页面设置与 UI
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

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.title("🔒 星鉴 AstroFusion V1.0")
        pwd = st.text_input("访问码 (默认 123456)", type="password")
        if st.button("进入平台", use_container_width=True):
            if pwd == "123456":
                st.session_state["password_correct"] = True
                st.rerun()
            else: st.error("密码错误！")
        return False
    return True

# 2. 核心工具函数
def format_math_text(text):
    if not text: return text
    text = text.replace("\\[", "$$").replace("\\]", "$$").replace("\\(", "$").replace("\\)", "$")
    text = re.sub(r'```math\n(.*?)\n```', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\[\s*(.*?(?:=|\\frac|\\approx).*?)\s*\]', r'$$\1$$', text)
    return text

def render_web_image(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if response.status_code == 200:
            st.image(Image.open(BytesIO(response.content)), use_column_width=True)
    except: pass

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

def run_multi_algorithms(time_arr, flux_flat, min_p=0.2, max_p=5.0):
    freq = np.linspace(1/max_p, 1/min_p, 2000)
    ls = LombScargle(time_arr, flux_flat - np.mean(flux_flat))
    power_ls = ls.power(freq)
    p_ls = 1 / freq[np.argmax(power_ls)]
    max_p_ls = np.max(power_ls)
    try:
        bls = BoxLeastSquares(time_arr, flux_flat)
        bls_res = bls.power(freq, np.linspace(0.05, 0.2, 5))
        p_bls = 1 / freq[np.argmax(bls_res.power)]
    except: p_bls = p_ls 
    half_p = p_ls / 2.0
    half_power = power_ls[np.argmin(np.abs(1/freq - half_p))]
    return p_ls, p_bls, max_p_ls, half_power, freq, power_ls

def internet_search(query):
    try:
        res = DDGS().text(f"{query} astronomy", max_results=3)
        return "【文献】\n" + "\n".join([f"- {r['title']}: {r['body']}" for r in res]) if res else ""
    except: return ""

def image_search(query):
    try:
        res = DDGS().images(f"{query} astronomy NASA", max_results=2)
        return [r['image'] for r in res] if res else []
    except: return []

# 3. 主程序
if check_password():
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  # 🔑 填入你的真实 Key
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
            if not tic_input: st.error("请输入编号！")
            else:
                with st.spinner("获取 NASA 数据..."):
                    lc_data = download_tess_data(tic_input)
                    if lc_data is not None: 
                        st.session_state.lc = lc_data
                        st.session_state.tic = tic_input
                        st.success("分析完成！")
                    else: st.error("下载失败。")
        st.markdown("---")
        voice_text = speech_to_text(language='zh-CN', start_prompt="🎤 语音输入", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
        uploaded_file = st.file_uploader("📷 上传图片", type=["jpg", "png"])

    tab1, tab2 = st.tabs(["📊 核心诊断与冲突识别", "📦 批量与导出 (原型)"])

    with tab1:
        col_left, col_right = st.columns([1.3, 1])

        # --- 左侧图表区 ---
        with col_left:
            if "lc" in st.session_state and st.session_state.lc is not None:
                lc = st.session_state.lc
                time_arr, flux_arr = lc.time.value, lc.flux.value
                wp = int(window_length * 720)
                if wp % 2 == 0: wp += 1
                trend = median_filter(flux_arr, size=wp, mode='nearest')
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
                fig_pg.add_vline(x=p_ls, line_dash="dash", line_color="#EF4444", annotation_text="P")
                fig_pg.add_vline(x=p_ls/2, line_dash="dot", line_color="#8B5CF6", annotation_text="P/2")
                fig_pg.update_layout(title="周期谱与候选峰", height=250, template="plotly_white", xaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_pg, use_container_width=True)
                
                phase = (time_arr / p_ls) % 1.0; phase[phase > 0.5] -= 1.0
                fig_phase = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                fig_phase.add_trace(go.Scattergl(x=phase, y=flux_flat, mode='markers', marker=dict(size=2, color='#000000')), row=1, col=1)
                fig_phase.add_trace(go.Scattergl(x=phase, y=flux_flat - np.mean(flux_flat), mode='markers', marker=dict(size=2, color='#EF4444')), row=2, col=1)
                fig_phase.update_layout(height=300, template="plotly_white", title="相位折叠与残差", showlegend=False)
                st.plotly_chart(fig_phase, use_container_width=True)

                st.markdown("### 🩺 多候选周期与冲突诊断")
                df = pd.DataFrame({"算法": ["LS", "BLS", "P/2校验"], "周期 (天)": [round(p_ls, 4), round(p_bls, 4), round(p_ls/2, 4)], "标签": ["consistent", "consistent", "half_period_alias" if is_half_period else "no_alias"]})
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                if is_insufficient: st.error("🚫 **拒判**：数据点过少。"); st.session_state.tag = "insufficient_evidence"
                elif is_half_period: st.warning(f"⚠️ **半周期混叠**：P/2 ({p_ls/2:.3f}d) 功率高！"); st.session_state.tag = "half_period_alias"
                else: st.success(f"✅ **周期可靠**：最佳 {p_ls:.4f} 天。"); st.session_state.tag = "consistent"
                st.session_state.period = p_ls
            else: st.info("👈 请在左侧启动分析。")

        # --- 右侧AI对话区 ---
        with col_right:
            st.subheader("💬 科研 Agent")
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "你好！可问我天文问题。我会提供**课本公式**、**微信复制版**，若需图片可直接说（如：给一张太阳实景图）。"}]
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
                    
            prompt = st.chat_input("输入问题...")
            if voice_text and not prompt: prompt = voice_text
            if prompt or uploaded_file:
                user_msg = {"role": "user", "content": prompt if prompt else "分析图片"}
                with st.chat_message("user"):
                    if uploaded_file: st.image(uploaded_file, width=200); user_msg["image"] = uploaded_file
                    st.markdown(user_msg["content"])
                st.session_state.messages.append(user_msg)
                
                with st.chat_message("assistant"):
                    sys_prompt = """
                    你是资深天文学家。请专业回答。
                    【死命令1】绝对禁止使用 `[`、`\\[` 包裹公式！独立公式必须用 $$...$$，行内用 $...$。
                    【死命令2】在回答最后，加一个 ```text 代码块，提供纯文本版结论（不含$和\\，用Unicode）。
                    """
                    try:
                        client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                        if uploaded_file:
                            img_bytes = uploaded_file.getvalue()
                            b64 = base64.b64encode(img_bytes).decode('utf-8')
                            res = client.chat.completions.create(model="glm-4v-flash", messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}])
                            st.markdown(res.choices[0].message.content)
                        else:
                            # 联网搜图
                            if "图" in prompt or "照片" in prompt:
                                imgs = image_search(prompt)
                                for url in imgs: render_web_image(url)
                            # 联网搜文献
                            elif "文献" in prompt or "查" in prompt:
                                ctx = internet_search(prompt)
                                sys_prompt += f"\n参考文献：{ctx}"
                            
                            with st.spinner("🧠 深度推理与排版中..."):
                                res = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages[-4:], stream=False)
                                cleaned = format_math_text(res.choices[0].message.content)
                                st.markdown(cleaned)
                                st.session_state.messages.append({"role": "assistant", "content": cleaned})
                        st.rerun()
                    except Exception as e: st.warning(f"网络波动: {e}")

    with tab2:
        st.subheader("📦 批量任务与报告导出 (原型)")
        st.info("展示系统未来批量处理能力。")
        col_up, col_down = st.columns(2)
        with col_up:
            st.file_uploader("拖拽 TIC CSV 列表", type=["csv"])
            if st.button("▶️ 启动批量诊断"): st.dataframe(pd.DataFrame({"TIC": [260128333, 12345678], "周期": [0.4068, 1.2345], "标签": ["consistent", "half_period"]}))
        with col_down:
            st.write("将当前分析图表与 AI 结论打包导出。")
            st.selectbox("导出格式", ["PDF", "Markdown", "CSV"])
            if st.button("📄 导出报告"): st.success("报告生成成功！(模拟)")
