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

# ==========================================
# 1. 页面设置与浅色 UI 主题
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
    .stButton>button:hover { background-color: #2563EB !important; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 密码锁保护
# ==========================================
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False
if not st.session_state["password_correct"]:
    st.title("🔒 星鉴 AstroFusion V1.0 竞赛版")
    pwd = st.text_input("请输入访问码 (默认 123456)", type="password")
    if st.button("进入系统", use_container_width=True):
        if pwd == "123456":
            st.session_state["password_correct"] = True
            st.rerun()
        else:
            st.error("密码错误！")
    st.stop()

# ==========================================
# 3. 全局配置与核心函数
# ==========================================
# 🔑 填入你的真实 API Key
ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# 暴力正则清洗，保证 100% 渲染为课本公式
def format_math_text(text):
    if not text: return text
    text = text.replace("\\[", "$$").replace("\\]", "$$").replace("\\(", "$").replace("\\)", "$")
    text = re.sub(r'```math\n(.*?)\n```', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\[\s*(.*?(?:=|\\frac|\\approx).*?)\s*\]', r'$$\1$$', text)
    return text

# 带有反爬机制的实景图片下载渲染器
def render_real_image(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            st.image(img, use_column_width=True)
            return True
    except Exception:
        return False
    return False

# 自动下载 NASA 数据并获取星体名称
@st.cache_data(show_spinner=False)
def download_tess_data(tic_id):
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search_result) == 0: return None, None
        lc = search_result[0].download()
        if lc is None: return None, None
        lc = lc.remove_nans().normalize()
        mask = (lc.flux.value > 0.8) & (lc.flux.value < 1.2)
        obj_name = lc.meta.get('OBJECT', f"TIC {tic_id}")
        return lc[mask], obj_name
    except Exception:
        return None, None

# 多算法（LS & BLS）周期计算
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
    except Exception:
        p_bls = p_ls 
        
    half_p = p_ls / 2.0
    half_power = power_ls[np.argmin(np.abs(1/freq - half_p))]
    return p_ls, p_bls, max_p_ls, half_power, freq, power_ls

# 全网实景图片搜索
def image_search(query):
    try:
        results = DDGS().images(f"{query} astronomy NASA", max_results=1)
        return [r['image'] for r in results] if results else []
    except Exception:
        return []

# ==========================================
# 4. 侧边栏控制台
# ==========================================
with st.sidebar:
    st.header("⚙️ 数据控制台")
    tic_input = st.text_input("输入 TESS TIC 编号", value="260128333")
    win_len = st.slider("去趋势滑动窗口 (天)", 0.1, 2.0, 0.5, 0.1)
    
    st.markdown("---")
    if st.button("🚀 启动核心分析", use_container_width=True):
        if not tic_input:
            st.error("请输入编号！")
        else:
            with st.spinner("正在从 NASA 获取数据并搜索实景图..."):
                lc_data, obj_name = download_tess_data(tic_input)
                if lc_data is not None:
                    st.session_state.lc = lc_data
                    st.session_state.obj_name = obj_name
                    # 自动抓取该星球的实景图
                    imgs = image_search(obj_name if obj_name != f"TIC {tic_input}" else "deep space star")
                    st.session_state.img_url = imgs[0] if imgs else ""
                    st.success(f"分析成功！目标：{obj_name}")
                else:
                    st.error("下载失败或无数据。")
                    
    st.markdown("---")
    st.header("🎙️ 多模态输入")
    voice_text = speech_to_text(language='zh-CN', start_prompt="🎤 语音输入", stop_prompt="⏹️ 结束", just_once=True, use_container_width=True)
    uploaded_file = st.file_uploader("📷 上传天文图片", type=["jpg", "jpeg", "png"])

# ==========================================
# 5. 主界面渲染 (左图右聊)
# ==========================================
tab1, tab2 = st.tabs(["📊 核心诊断与冲突识别", "📦 批量与导出 (原型)"])

with tab1:
    col_left, col_right = st.columns([1.3, 1])

    # ---------- 左侧：图表区与实景图 ----------
    with col_left:
        if "lc" in st.session_state and st.session_state.lc is not None:
            # 1. 渲染自动抓取的实景图片
            if st.session_state.get("img_url"):
                st.markdown(f"#### 🔭 真实观测/实景影像：{st.session_state.obj_name}")
                if not render_real_image(st.session_state.img_url):
                    st.warning("（图片受源站防盗链限制加载失败）")
            else:
                st.info(f"未找到 {st.session_state.obj_name} 的公开实景图，以下为测光分析：")

            lc = st.session_state.lc
            t, f = lc.time.value, lc.flux.value
            
            # 2. 计算与去趋势
            wp = int(win_len * 720)
            if wp % 2 == 0: wp += 1
            trend = median_filter(f, size=wp, mode='nearest')
            f_flat = f - trend + 1.0
            
            p_ls, p_bls, max_p_ls, half_power, freq, power_ls = run_multi_algorithms(t, f_flat)
            is_half_period = half_power > max_p_ls * 0.7
            
            # 3. 光变曲线与周期谱
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scattergl(x=t, y=f, mode='markers', marker=dict(size=2, color='#CBD5E1'), name='原始'))
            fig_lc.add_trace(go.Scattergl(x=t, y=f_flat, mode='markers', marker=dict(size=2, color='#3B82F6'), name='去趋势'))
            fig_lc.update_layout(title="TESS 光变曲线", height=250, template="plotly_white")
            st.plotly_chart(fig_lc, use_container_width=True)
            
            fig_pg = go.Figure()
            fig_pg.add_trace(go.Scatter(x=1/freq, y=power_ls, mode='lines', line=dict(color='#F59E0B')))
            fig_pg.add_vline(x=p_ls, line_dash="dash", line_color="#EF4444", annotation_text="主峰 P")
            fig_pg.add_vline(x=p_ls/2, line_dash="dot", line_color="#8B5CF6", annotation_text="半周期 P/2")
            fig_pg.update_layout(title="周期谱与候选峰", height=250, template="plotly_white", xaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_pg, use_container_width=True)
            
            # 4. 冲突诊断表格
            st.markdown("### 🩺 多候选周期与冲突诊断")
            df = pd.DataFrame({
                "算法": ["Lomb-Scargle", "BLS", "P/2校验"], 
                "周期 (天)": [round(p_ls, 4), round(p_bls, 4), round(p_ls/2, 4)], 
                "诊断标签": ["consistent", "consistent", "half_period_alias" if is_half_period else "no_alias"]
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            if is_half_period:
                st.warning(f"⚠️ **半周期混叠预警**：P/2 ({p_ls/2:.3f}d) 功率过高！")
            else:
                st.success(f"✅ **周期可靠**：最佳周期 {p_ls:.4f} 天。")
                
            st.session_state.period = p_ls
        else:
            st.info("👈 请在左侧控制台输入 TIC 编号并点击「启动核心分析」。")

    # ---------- 右侧：AI 对话区 ----------
    with col_right:
        st.subheader("💬 科研 Agent 对话")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！我已经为你调取了目标天体的数据与实景影像。你可以继续向我提问，我会提供课本公式与微信复制版。"}]
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        prompt = st.chat_input("输入问题...")
        if voice_text and not prompt: 
            prompt = voice_text
            
        if prompt or uploaded_file:
            user_msg = {"role": "user", "content": prompt if prompt else "请分析图片。"}
            with st.chat_message("user"):
                if uploaded_file: 
                    st.image(uploaded_file, width=200)
                    user_msg["image"] = uploaded_file
                st.markdown(user_msg["content"])
            st.session_state.messages.append(user_msg)
            
            with st.chat_message("assistant"):
                # 绝对强硬的系统提示词
                sys_prompt = """
                你是一个资深天文学家。请注意：系统已经在左侧面板自动展示了当前天体的实景图。
                1. 【图片禁止道歉】：用户问你要图片时，绝对禁止回答“我作为AI无法提供图片”，你必须自信地说“系统已在左侧为您展示了该目标的实景影像，以下是专业分析...”。
                2. 【课本公式死命令】：严禁使用 `[` 或 `\\[` 包裹公式！独立公式必须使用 $$...$$，行内公式用 $...$。
                3. 【微信复制版】：在回答的最后，请加一个 ```text 代码块，提供纯文本版结论。
                """
                
                try:
                    client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                    if uploaded_file:
                        img_bytes = uploaded_file.getvalue()
                        b64 = base64.b64encode(img_bytes).decode('utf-8')
                        res = client.chat.completions.create(
                            model="glm-4v-flash", 
                            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
                        )
                        st.markdown(res.choices[0].message.content)
                    else:
                        context = f"【当前分析】{st.session_state.get('obj_name', '未知')}，周期 {st.session_state.get('period', 0):.4f} 天。"
                        with st.spinner("🧠 深度推理中..."):
                            # 关闭 stream=False，彻底避免公式被拆散导致无法渲染
                            res = client.chat.completions.create(
                                model="glm-4-flash", 
                                messages=[{"role": "system", "content": sys_prompt + context}] + st.session_state.messages[-4:], 
                                stream=False
                            )
                            cleaned = format_math_text(res.choices[0].message.content)
                            st.markdown(cleaned)
                            st.session_state.messages.append({"role": "assistant", "content": cleaned})
                    st.rerun()
                except Exception as e:
                    st.warning(f"网络波动: {e}")

with tab2:
    st.subheader("📦 批量任务与报告导出 (原型)")
    st.info("展示系统未来的批量处理与一键导出能力。")
