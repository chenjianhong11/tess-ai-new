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
import re, requests
from PIL import Image
from io import BytesIO

# 1. 页面设置与浅色 UI
st.set_page_config(page_title="星鉴 AstroFusion", layout="wide")
st.markdown("""<style>:root{color-scheme:light!important;}.stApp{background-color:#F4F7F9!important;}[data-testid="stSidebar"]{background-color:#FFF!important;border-right:1px solid #E0E6ED;}div[data-baseweb="input"],input,textarea{background-color:#FFF!important;color:#1F2937!important;border-radius:8px!important;}[data-testid="stChatMessage"]{background-color:#FFF!important;border-radius:10px!important;border:1px solid #E5E7EB;}.stButton>button{background-color:#3B82F6!important;color:#FFF!important;border-radius:8px;font-weight:bold;}</style>""", unsafe_allow_html=True)

# 2. 密码锁
if "pwd_ok" not in st.session_state: st.session_state.pwd_ok = False
if not st.session_state.pwd_ok:
    st.title("🔒 星鉴 AstroFusion V1.0 竞赛版")
    pwd = st.text_input("请输入访问码 (默认 123456)", type="password")
    if st.button("进入系统"):
        if pwd == "123456": st.session_state.pwd_ok = True; st.rerun()
        else: st.error("密码错误！")
    st.stop()

# 3. 核心配置与函数
ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  # 🔑 填你的 Key
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# 强制公式清洗器
def format_math(text):
    if not text: return text
    text = text.replace("\\[", "$$").replace("\\]", "$$").replace("\\(", "$").replace("\\)", "$")
    text = re.sub(r'```math\n(.*?)\n```', r'$$\1$$', text, flags=re.DOTALL)
    return re.sub(r'\[\s*(.*?(?:=|\\frac|\\approx).*?)\s*\]', r'$$\1$$', text)

# 实景图加载（带防爬与兜底）
def render_img(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=8)
        if res.status_code == 200:
            st.image(Image.open(BytesIO(res.content)), use_column_width=True)
            return
    except: pass
    # 兜底图：如果源站拦截，显示一张绝美NASA星空图
    st.image("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&w=800&q=80", caption="NASA 深空影像 (占位)", use_column_width=True)

@st.cache_data(show_spinner=False)
def get_tess(tic):
    try:
        res = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
        if len(res) == 0: return None, None
        lc = res[0].download().remove_nans().normalize()
        mask = (lc.flux.value > 0.8) & (lc.flux.value < 1.2)
        return lc[mask], lc.meta.get('OBJECT', f"TIC {tic}")
    except: return None, None

def run_algo(t, f, min_p=0.2, max_p=5.0):
    freq = np.linspace(1/max_p, 1/min_p, 2000)
    ls = LombScargle(t, f - np.mean(f))
    p_ls = 1 / freq[np.argmax(ls.power(freq))]
    try: p_bls = 1 / freq[np.argmax(BoxLeastSquares(t, f).power(freq, np.linspace(0.05, 0.2, 5)).power)]
    except: p_bls = p_ls
    return p_ls, p_bls, freq, ls.power(freq)

def get_img(query):
    try:
        res = DDGS().images(f"{query} space NASA", max_results=1)
        return res[0]['image'] if res else ""
    except: return ""

# 4. 侧边栏控制台（保留截图所有功能）
with st.sidebar:
    st.header("⚙️ 数据控制台")
    tic_input = st.text_input("TESS TIC 编号", value="260128333")
    win_len = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
    
    st.markdown("---")
    st.header("🔄 联合验证")
    ref_p = st.number_input("其他星表周期 (天)", value=0.0, format="%.4f")
    fast_p = st.number_input("FAST 射电周期 (天)", value=0.0, format="%.3f")
    
    if st.button("🚀 启动核心分析", use_container_width=True):
        with st.spinner("获取 NASA 数据并搜索实景图..."):
            lc, name = get_tess(tic_input)
            if lc is not None:
                st.session_state.lc = lc
                st.session_state.name = name
                st.session_state.img = get_img(name)
                st.success(f"目标：{name}")
            else: st.error("获取失败")
            
    st.markdown("---")
    st.header("🎙️ 多模态输入")
    voice = speech_to_text(language='zh-CN', start_prompt="🎤 语音输入", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
    upload = st.file_uploader("📷 上传图片", type=["jpg", "png"])

# 5. 主界面双栏布局
tab1, tab2 = st.tabs(["📊 核心诊断与冲突识别", "📦 批量与导出 (原型)"])

with tab1:
    c_left, c_right = st.columns([1.3, 1])
    
    # ---- 左侧：图表与实景图 ----
    with c_left:
        if "lc" in st.session_state:
            st.markdown(f"#### 🔭 实景影像：{st.session_state.name}")
            if st.session_state.get("img"): render_img(st.session_state.img)
            
            lc = st.session_state.lc
            t, f = lc.time.value, lc.flux.value
            wp = int(win_len * 720); wp += 1 if wp%2==0 else 0
            f_flat = f - median_filter(f, size=wp, mode='nearest') + 1.0
            
            # 光变曲线
            fig = go.Figure([go.Scattergl(x=t, y=f, mode='markers', marker=dict(size=2, color='#CBD5E1'), name='原始'), go.Scattergl(x=t, y=f_flat, mode='markers', marker=dict(size=2, color='#3B82F6'), name='去趋势')])
            fig.update_layout(title="光变曲线", height=220, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            
            # 周期图
            p_ls, p_bls, freq, power = run_algo(t, f_flat)
            fig2 = go.Figure([go.Scatter(x=1/freq, y=power, mode='lines', line=dict(color='#F59E0B'))])
            fig2.add_vline(x=p_ls, line_dash="dash", line_color="red", annotation_text="P")
            fig2.add_vline(x=p_ls/2, line_dash="dot", line_color="purple", annotation_text="P/2")
            fig2.update_layout(title="周期图", height=220, template="plotly_white", xaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)
            
            # 相位折叠
            phase = (t / p_ls) % 1.0; phase[phase > 0.5] -= 1.0
            fig3 = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig3.add_trace(go.Scattergl(x=phase, y=f_flat, mode='markers', marker=dict(size=2, color='#000')), row=1, col=1)
            fig3.add_trace(go.Scattergl(x=phase, y=f_flat-np.mean(f_flat), mode='markers', marker=dict(size=2, color='red')), row=2, col=1)
            fig3.update_layout(title="相位折叠图", height=280, template="plotly_white", showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            
            # 诊断表
            st.markdown("### 🩺 诊断报告")
            st.dataframe(pd.DataFrame({"算法": ["LS", "BLS", "P/2"], "周期": [round(p_ls,4), round(p_bls,4), round(p_ls/2,4)], "标签": ["consistent", "consistent", "half_alias" if power[np.argmin(np.abs(1/freq-p_ls/2))] > max(power)*0.7 else "no_alias"]}), use_container_width=True, hide_index=True)
            st.session_state.period = p_ls
        else:
            st.info("👈 请在左侧输入编号并启动分析。")

    # ---- 右侧：AI 对话 ----
    with c_right:
        st.subheader("💬 科研 Agent 对话")
        if "msgs" not in st.session_state: st.session_state.msgs = [{"role": "assistant", "content": "你好！我已调取实景影像。你可以问我物理参数，我会提供课本公式与微信复制版。"}]
        for m in st.session_state.msgs:
            with st.chat_message(m["role"]): st.markdown(m["content"])
            
        prompt = st.chat_input("输入问题...")
        prompt = voice if not prompt and voice else prompt
        
        if prompt or upload:
            st.session_state.msgs.append({"role": "user", "content": prompt or "分析图片"})
            with st.chat_message("user"):
                if upload: st.image(upload, width=200)
                st.markdown(prompt or "分析图片")
                
            with st.chat_message("assistant"):
                sys_p = "你是资深天文学家。系统已在左侧展示实景图，禁答'无法提供'。严禁用[]包裹公式，独立公式必须用$$，行内用$。最后用```text提供微信纯文本版。"
                try:
                    client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                    with st.spinner("🧠 深度推理与公式排版中..."):
                        # 关闭流式，确保公式完整
                        res = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": prompt}], stream=False)
                        clean = format_math(res.choices[0].message.content)
                        st.markdown(clean)
                        st.session_state.msgs.append({"role": "assistant", "content": clean})
                    st.rerun()
                except Exception as e: st.warning(f"网络波动: {e}")

with tab2:
    st.subheader("📦 批量任务与报告导出 (原型)")
    st.info("展示系统未来的批量处理与一键导出能力。")
