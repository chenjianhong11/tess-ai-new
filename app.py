import streamlit as st
import numpy as np, pandas as pd, plotly.graph_objects as go, plotly.subplots as sp
import lightkurve as lk
from astropy.timeseries import LombScargle, BoxLeastSquares
from openai import OpenAI
import base64, re, requests
from scipy.ndimage import median_filter
from duckduckgo_search import DDGS
from streamlit_mic_recorder import speech_to_text
from PIL import Image
from io import BytesIO

# 1. 页面设置
st.set_page_config(page_title="星鉴 AstroFusion", layout="wide")
st.markdown("""<style>:root{color-scheme:light!important;}.stApp{background:#F4F7F9!important;}[data-testid="stSidebar"]{background:#FFF!important;border-right:1px solid #E0E6ED;}div[data-baseweb="input"],input,textarea{background:#FFF!important;border-radius:8px!important;}[data-testid="stChatMessage"]{background:#FFF!important;border-radius:10px;border:1px solid #E5E7EB;}.stButton>button{background:#3B82F6!important;color:#FFF!important;border-radius:8px;font-weight:bold;}</style>""", unsafe_allow_html=True)

# 2. 密码锁
if not st.session_state.get("pwd_ok", False):
    st.title("🔒 星鉴 AstroFusion 竞赛版")
    if st.button("进入平台"):
        if st.text_input("访问码 (默认 123456)", type="password", key="pwd") == "123456": 
            st.session_state.pwd_ok = True; st.rerun()
    st.stop()

# 3. 核心函数
ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7" # 🔑 填你的 Key
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

def clean_math(t): # 暴力修复公式
    if not t: return t
    t = t.replace("\\[", "$$").replace("\\]", "$$").replace("\\(", "$").replace("\\)", "$")
    return re.sub(r'```math\n(.*?)\n```', r'$$\1$$', t, flags=re.DOTALL)

def get_img(tic): # 获取实景图(带兜底)
    try:
        res = DDGS().images(f"TIC {tic} space NASA", max_results=1)
        if res: return Image.open(BytesIO(requests.get(res[0]['image'], headers={'User-Agent':'Mozilla/5.0'}, timeout=5).content))
    except: pass
    return Image.open(BytesIO(requests.get("https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&w=800&q=80", timeout=5).content))

@st.cache_data(show_spinner=False)
def get_tess(tic):
    try:
        res = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
        if not res: return None
        lc = res[0].download().remove_nans().normalize()
        return lc[(lc.flux.value > 0.8) & (lc.flux.value < 1.2)]
    except: return None

def calc_period(t, f):
    freq = np.linspace(1/5.0, 1/0.2, 2000)
    ls = LombScargle(t, f - np.mean(f)); p_ls = 1/freq[np.argmax(ls.power(freq))]
    try: p_bls = 1/freq[np.argmax(BoxLeastSquares(t, f).power(freq, np.linspace(0.05, 0.2, 5)).power)]
    except: p_bls = p_ls
    return p_ls, p_bls, freq, ls.power(freq)

# 4. 侧边栏 (完整保留)
with st.sidebar:
    st.header("⚙️ 数据控制台")
    tic_input = st.text_input("TESS TIC 编号", value="260128333")
    win_len = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
    st.markdown("---"); st.header("🔄 联合验证")
    ref_p = st.number_input("其他星表周期 (天)", value=0.0, format="%.4f")
    fast_p = st.number_input("FAST 射电周期 (天)", value=0.0, format="%.3f")
    
    if st.button("🚀 启动核心分析", use_container_width=True):
        with st.spinner("获取数据与配图..."):
            lc = get_tess(tic_input)
            if lc is not None:
                st.session_state.lc, st.session_state.tic = lc, tic_input
                st.session_state.img = get_img(tic_input)
                st.success("分析完成！")
            else: st.error("获取失败")
            
    st.markdown("---")
    voice = speech_to_text(language='zh-CN', start_prompt="🎤 语音输入", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
    upload = st.file_uploader("📷 上传图片", type=["jpg", "png"])

# 5. 主界面双栏布局
c1, c2 = st.columns([1.3, 1])

with c1: # --- 左侧图表与诊断 ---
    if "lc" not in st.session_state:
        st.info("👈 请在左侧输入编号并启动分析。")
    else:
        if st.session_state.get("img"): 
            st.markdown(f"#### 🔭 目标实景模拟：TIC {st.session_state.tic}")
            st.image(st.session_state.img, use_column_width=True)
        
        lc = st.session_state.lc; t, f = lc.time.value, lc.flux.value
        wp = int(win_len * 720); wp += 1 if wp%2==0 else 0
        f_flat = f - median_filter(f, size=wp, mode='nearest') + 1.0
        
        # 光变曲线
        fig1 = go.Figure([go.Scattergl(x=t, y=f, mode='markers', marker=dict(size=2, color='#CBD5E1'), name='原始'), go.Scattergl(x=t, y=f_flat, mode='markers', marker=dict(size=2, color='#3B82F6'), name='去趋势')])
        fig1.update_layout(title="光变曲线", height=220, template="plotly_white"); st.plotly_chart(fig1, use_container_width=True)
        
        # 周期图
        p_ls, p_bls, freq, power = calc_period(t, f_flat)
        fig2 = go.Figure([go.Scatter(x=1/freq, y=power, mode='lines', line=dict(color='#F59E0B'))])
        fig2.add_vline(x=p_ls, line_dash="dash", line_color="red", annotation_text="P")
        fig2.add_vline(x=p_ls/2, line_dash="dot", line_color="purple", annotation_text="P/2")
        fig2.update_layout(title="周期图", height=220, template="plotly_white", xaxis=dict(autorange="reversed")); st.plotly_chart(fig2, use_container_width=True)
        
        # 相位折叠
        phase = (t / p_ls) % 1.0; phase[phase > 0.5] -= 1.0
        fig3 = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig3.add_trace(go.Scattergl(x=phase, y=f_flat, mode='markers', marker=dict(size=2, color='#000')), row=1, col=1)
        fig3.add_trace(go.Scattergl(x=phase, y=f_flat-np.mean(f_flat), mode='markers', marker=dict(size=2, color='red')), row=2, col=1)
        fig3.update_layout(title="相位折叠图", height=260, template="plotly_white", showlegend=False); st.plotly_chart(fig3, use_container_width=True)
        
        # 诊断表
        st.markdown("### 🩺 诊断报告")
        is_half = power[np.argmin(np.abs(1/freq-p_ls/2))] > max(power)*0.7
        st.dataframe(pd.DataFrame({"算法": ["LS", "BLS", "P/2"], "周期": [round(p_ls,4), round(p_bls,4), round(p_ls/2,4)], "标签": ["consistent", "consistent", "half_alias" if is_half else "no_alias"]}), use_container_width=True, hide_index=True)
        
        if ref_p > 0: st.info(f"🔄 **多源比对**：与参考周期比值 **{p_ls/ref_p:.2f}**")
        if fast_p > 0: st.info(f"📡 **FAST 验证**：与射电周期比值 **{p_ls/fast_p:.2f}**")
        st.session_state.period = p_ls

with c2: # --- 右侧 AI 对话 ---
    st.subheader("💬 科研 Agent 对话")
    if "msgs" not in st.session_state: 
        st.session_state.msgs = [{"role": "assistant", "content": "你好！系统已配图。你可以问我物理参数，我会提供**课本公式**与**微信复制版**。"}]
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
            sys_p = "你是资深天文学家。禁答'无法提供图片'。严禁用[]包裹公式，独立公式用$$，行内用$。最后用```text提供微信纯文本版。"
            context = f"当前周期 {st.session_state.get('period', 0):.4f}天。"
            try:
                client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                with st.spinner("🧠 排版中..."):
                    res = client.chat.completions.create(model="glm-4-flash", messages=[{"role": "system", "content": sys_p + context}, {"role": "user", "content": prompt}], stream=False)
                    clean = clean_math(res.choices[0].message.content)
                    st.markdown(clean)
                    st.session_state.msgs.append({"role": "assistant", "content": clean})
                st.rerun()
            except Exception as e: st.warning(f"网络波动: {e}")
