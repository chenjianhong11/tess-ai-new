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
# 1. 页面设置
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
# 2. 核心功能：暴力修复公式渲染
# ==========================================
def fix_latex_formatting(text):
    """极其暴力的公式清洗：确保所有数学公式都被包裹在 $$ 或 $ 中，以供 Streamlit 渲染成课本样式"""
    if not text: return text
    
    # 1. 替换转义的括号
    text = text.replace("\\[", "$$").replace("\\]", "$$")
    text = text.replace("\\(", "$").replace("\\)", "$")
    
    # 2. 把大模型经常错误输出的 [公式] 强行转为 $$公式$$
    # 匹配包含等号、frac、times 等数学特征的方括号
    text = re.sub(r'\[\s*(.*?(?:=|\\frac|\\times|\\cdot|\\approx|\\pm|\\Sigma|\\int).*?)\s*\]', r'$$\1$$', text)
    
    # 3. 把单行内的独立公式（如 f = 1/P）如果没加 $ 的，尝试补充（可选，防止误伤纯文本）
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
                with st.spinner("正在获取数据..."):
                    lc_data = download_tess_data(tic_input)
                    if lc_data is None: st.error("下载失败。")
                    else: 
                        st.session_state.lc = lc_data
                        st.session_state.tic = tic_input
                        st.success("分析完成！")

        st.markdown("---")
        voice_text = speech_to_text(language='zh-CN', start_prompt="🎤 语音输入", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
        uploaded_file = st.file_uploader("📷 上传天文图片", type=["jpg", "jpeg", "png"])

    tab1, tab2 = st.tabs(["📊 核心诊断与冲突识别", "📦 批量与导出"])

    with tab1:
        col_left, col_right = st.columns([1.3, 1])

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
                
                st.session_state.period = p_ls
                st.session_state.tag = "consistent" if half_power < max_p_ls * 0.7 else "half_period_alias"
            else:
                st.info("👈 请在左侧输入编号并启动分析。")

        # ==========================================
        # 右侧：AI 对话区 (彻底修复渲染)
        # ==========================================
        with col_right:
            st.subheader("💬 科研 Agent 对话")
            
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "你好！你可以问我任何天文问题。我会为你提供**课本印刷体**的公式，并在下方提供**微信一键复制**的纯文本。"}]
            
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
                    # 强制使用 $$ 渲染，并附加 text 代码块供复制
                    sys_prompt = """
                    你是一个资深天文学家。请专业回答用户问题。
                    
                    【第一要求：课本排版（绝对死命令）】
                    1. 先输出【深度思考过程】，再输出【最终解答】。
                    2. 严禁使用 `[ ]`、`\\[`、`\\]` 包裹任何公式！
                    3. 独立的数学公式，**必须且只能**使用双美元符号包裹，例如：$$ F = G \\frac{m_1 m_2}{r^2} $$
                    4. 行内公式，使用单美元符号包裹，例如：$ E = mc^2 $。
                    
                    【第二要求：微信复制友好格式】
                    在回答的最后，请加一个 Markdown 代码块（用 ```text 包裹），提供纯文本版结论（不能有$和\\符号，用 Unicode 字符如 m₁，r²，×）。
                    示例：
                    ```text
                    【天文结论】
                    万有引力公式：F = G × (m₁ × m₂) / r²
                    ```
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
                            with st.status("🧠 AI 正在深度推理...", expanded=True) as status:
                                st.write("解析上下文与渲染课本公式...")
                                status.update(label="推理完成：", state="complete", expanded=False)
                            
                            def generate_chunks():
                                res = client.chat.completions.create(
                                    model="glm-4-flash",
                                    messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages[-4:],
                                    stream=True
                                )
                                for chunk in res:
                                    if chunk.choices[0].delta.content: 
                                        raw_text = chunk.choices[0].delta.content
                                        # 核心清洗：实时将大模型犯错的方括号转化为 $$ 
                                        cleaned_text = fix_latex_formatting(raw_text)
                                        yield cleaned_text
                            
                            reply = st.write_stream(generate_chunks())
                            
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                        st.rerun()
                    except Exception as e:
                        st.warning(f"⚠️ 网络波动: {e}")

    with tab2:
        st.subheader("📦 批量任务与报告导出 (原型)")
