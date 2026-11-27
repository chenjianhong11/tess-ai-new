import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import lightkurve as lk
from astropy.timeseries import LombScargle, BoxLeastSquares
from openai import OpenAI
import os
import datetime
import requests
from scipy.ndimage import median_filter
from fpdf import FPDF
from io import BytesIO
from PIL import Image

# ==========================================
# 0. 初始化与本地存储
# ==========================================
st.set_page_config(page_title="TESS 食双星科研工作台", layout="wide")

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
META_FILE = os.path.join(DATA_DIR, "analysis_metadata.csv")

# 🔑 🔑 🔑 在这里把 Key 直接写死 (替换成你自己的) 🔑 🔑 🔑
ZHIPU_API_KEY = "你的智谱Key粘贴在这里" 
BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

# ==========================================
# 1. 核心算法：PDM
# ==========================================
def calc_pdm(t, y, periods, nbins=10):
    scores = []
    for p in periods:
        phase = (t % p) / p
        bins = np.linspace(0, 1, nbins + 1)
        bin_idx = np.digitize(phase, bins) - 1
        variance, n = 0, 0
        for i in range(nbins):
            mask = (bin_idx == i)
            if np.sum(mask) > 1:
                variance += np.var(y[mask]) * np.sum(mask)
                n += np.sum(mask)
        scores.append(variance / n if n > 0 else np.inf)
    return 1.0 / (np.array(scores) + 1e-10)

# ==========================================
# 2. 数据获取引擎
# ==========================================
@st.cache_data(show_spinner=False)
def fetch_data(tic, force_offline=False):
    local_path = os.path.join(DATA_DIR, f"TIC_{tic}.csv")
    if force_offline or os.path.exists(local_path):
        try:
            df = pd.read_csv(local_path)
            return df['time'].values, df['flux'].values, "成功读取本地离线光变数据"
        except Exception as e:
            return None, None, f"本地文件读取失败: {e}"
            
    try:
        search = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
        if len(search) == 0: return None, None, "NASA 数据库未找到该星体"
        lc = search[0].download().remove_nans().normalize()
        t, f = lc.time.value, lc.flux.value
        pd.DataFrame({'time': t, 'flux': f}).to_csv(local_path, index=False)
        return t, f, "在线下载成功，已缓存至本地"
    except Exception as e:
        return None, None, f"网络获取失败: {e}"

# ==========================================
# 3. 左侧控制面板
# ==========================================
with st.sidebar:
    st.title("⚙️ 控制面板")
    tic_input = st.text_input("TESS TIC 编号", value="260128333")
    win_len = st.slider("去趋势滑动窗口 (天)", 0.1, 2.0, 0.5, 0.1)
    
    st.markdown("---")
    st.subheader("🔌 拓展验证开关")
    use_ext = st.toggle("启用外部星表/FAST 对比")
    if use_ext:
        ref_p = st.number_input("外部星表周期 (天)", value=0.0, format="%.4f")
        fast_p = st.number_input("FAST 射电周期 (天)", value=0.0, format="%.3f")
    else:
        ref_p, fast_p = 0.0, 0.0
        
    use_offline = st.toggle("强制离线模式 (断网演示)")
    st.markdown("---")
    
    if st.button("🚀 启动核心分析", use_container_width=True, type="primary"):
        with st.spinner("正在获取数据并计算..."):
            t_data, f_data, msg = fetch_data(tic_input, use_offline)
            if t_data is not None:
                st.session_state.update({
                    't': t_data, 'f': f_data, 'tic': tic_input,
                    'win': win_len, 'ref_p': ref_p, 'fast_p': fast_p,
                    'has_data': True, 'meta_saved': False
                })
                st.success(msg)
            else:
                st.error(msg)
                st.session_state.has_data = False

# ==========================================
# 4. 主体两栏布局 (中：展示区，右：Agent)
# ==========================================
col_mid, col_right = st.columns([2.2, 1])

# ---------- 中间：可视化展示区 ----------
with col_mid:
    if st.session_state.get('has_data', False):
        t = st.session_state.t
        f = st.session_state.f
        win = st.session_state.win
        
        # 1. 星云实景图
        try:
            img_url = "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?auto=format&fit=crop&w=1200&q=80"
            res = requests.get(img_url, timeout=5)
            st.image(Image.open(BytesIO(res.content)), use_column_width=True, caption="宇宙星云实景 (NASA 模拟)")
        except:
            st.info("✨ [星云实景图加载中...]")
        
        # 2. 核心计算
        wp = int(win * 720)
        if wp % 2 == 0: wp += 1
        trend = median_filter(f, size=wp, mode='nearest')
        f_flat = f - trend + 1.0
        
        freq = np.linspace(1/5.0, 1/0.2, 2000)
        periods = 1 / freq
        
        power_ls = LombScargle(t, f_flat - np.mean(f_flat)).power(freq)
        power_bls = BoxLeastSquares(t, f_flat).power(periods, np.linspace(0.05, 0.2, 5)).power
        power_pdm = calc_pdm(t, f_flat, periods)
        
        p_ls = periods[np.argmax(power_ls)]
        p_bls = periods[np.argmax(power_bls)]
        p_pdm = periods[np.argmax(power_pdm)]
        
        # 3. 冲突诊断
        is_half = power_ls[np.argmin(np.abs(periods - p_ls/2))] > max(power_ls)*0.7
        is_double = power_ls[np.argmin(np.abs(periods - p_ls*2))] > max(power_ls)*0.7
        is_insufficient = len(t) < 500 or np.std(f_flat) < 0.001
        
        tag_ls = "insufficient_evidence" if is_insufficient else ("half_period_alias" if is_half else ("double_period_alias" if is_double else "consistent"))
        st.session_state.tag_ls = tag_ls
        st.session_state.p_ls = p_ls
        
        # 绘图: 光变与周期谱
        st.subheader("📈 分析图表")
        tab1, tab2 = st.tabs(["光变与周期谱", "相位折叠"])
        with tab1:
            fig1 = go.Figure([go.Scattergl(x=t, y=f, mode='markers', marker=dict(size=2, color='#CBD5E1'), name='原始'), go.Scattergl(x=t, y=f_flat, mode='markers', marker=dict(size=2, color='#3B82F6'), name='去趋势')])
            fig1.update_layout(title="光变曲线", height=250, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = go.Figure([go.Scatter(x=periods, y=power_ls, mode='lines', line=dict(color='#F59E0B'), name='LS')])
            fig2.add_vline(x=p_ls, line_dash="dash", line_color="red", annotation_text="P")
            fig2.add_vline(x=p_ls/2, line_dash="dot", line_color="purple", annotation_text="P/2")
            fig2.add_vline(x=p_ls*2, line_dash="dot", line_color="green", annotation_text="2P")
            fig2.update_layout(title="LS 周期图", height=250, template="plotly_white", xaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig2, use_container_width=True)
            
        with tab2:
            phase = (t / p_ls) % 1.0
            phase[phase > 0.5] -= 1.0
            fig3 = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            fig3.add_trace(go.Scattergl(x=phase, y=f_flat, mode='markers', marker=dict(size=2, color='#000')), row=1, col=1)
            fig3.add_trace(go.Scattergl(x=phase, y=f_flat-np.mean(f_flat), mode='markers', marker=dict(size=2, color='red')), row=2, col=1)
            fig3.update_layout(title="相位折叠与残差", height=350, template="plotly_white", showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig3, use_container_width=True)

        # 4. 候选周期表
        st.subheader("📑 候选周期诊断表")
        df_res = pd.DataFrame({
            "算法": ["Lomb-Scargle", "BLS", "PDM"],
            "周期 (天)": [round(p_ls, 4), round(p_bls, 4), round(p_pdm, 4)],
            "得分/功率": [round(max(power_ls), 4), round(max(power_bls), 4), round(max(power_pdm), 4)],
            "冲突标签": [tag_ls, "consistent", "consistent"]
        })
        st.dataframe(df_res, use_container_width=True, hide_index=True)
        
        if tag_ls == "insufficient_evidence":
            st.error("🚫 **拒判提示**：数据点过少或信噪比过低。建议人工复核。")
        elif tag_ls == "half_period_alias":
            st.warning("⚠️ **半周期混叠预警**：检测到极强 P/2 峰值，可能发生半周期误判。")
            
        if use_ext and ref_p > 0:
            st.info(f"🔄 **外部比对**：与参考周期比值为 **{p_ls/ref_p:.3f}**")
        if use_ext and fast_p > 0:
            st.info(f"📡 **FAST 比对**：与射电周期比值为 **{p_ls/fast_p:.3f}**")

        # 5. 导出与元数据
        st.subheader("📤 导出与元数据")
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            csv = df_res.to_csv(index=False).encode('utf-8')
            st.download_button("📄 导出候选周期 CSV", csv, f"TIC{tic_input}_Periods.csv", "text/csv", use_container_width=True)
            
        with c_exp2:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt=f"AstroFusion TESS Report - TIC {tic_input}", ln=1, align='C')
            pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=1)
            pdf.cell(200, 10, txt=f"Best Period (LS): {p_ls:.4f} days", ln=1)
            pdf.cell(200, 10, txt=f"Diagnosis: {tag_ls}", ln=1)
            pdf_out = pdf.output(dest='S').encode('latin-1')
            st.download_button("📑 导出诊断 PDF 报告", pdf_out, f"TIC{tic_input}_Report.pdf", "application/pdf", use_container_width=True)

        if not st.session_state.get('meta_saved', False):
            meta_df = pd.DataFrame([{"Timestamp": datetime.datetime.now(), "TIC": tic_input, "Window": win, "Best_Period": p_ls, "Tag": tag_ls, "Has_FAST": fast_p > 0}])
            meta_df.to_csv(META_FILE, mode='a', header=not os.path.exists(META_FILE), index=False)
            st.session_state.meta_saved = True

    else:
        st.info("👈 请在左侧面板输入参数并点击【启动核心分析】")

# ---------- 右侧：科研 Agent 对话 ----------
with col_right:
    st.subheader("💬 科研 Agent")
    
    # 因为没有让用户输入 Key 的框了，只要代码里填了就能直接用
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("针对当前天体提问..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 注入上下文
        context = "当前处于未分析状态。"
        if st.session_state.get('has_data', False):
            tic_val = st.session_state.tic
            p_val = st.session_state.get('p_ls', 0)
            tag_val = st.session_state.get('tag_ls', 'unknown')
            context = f"当前正在分析 TESS 目标 TIC {tic_val}。LS算法测得主周期为 {p_val:.4f} 天，诊断标签为 {tag_val}。"
            if use_ext and ref_p > 0: context += f" 外部星表周期为 {ref_p} 天。"
            if use_ext and fast_p > 0: context += f" FAST 射电周期为 {fast_p} 天。"
        
        sys_prompt = f"你是资深天文学家，正在辅助科研。{context} 请基于上述数据回答，严禁胡编乱造。如果用户问无关问题，请拉回当前数据分析。"
        
        with st.chat_message("assistant"):
            try:
                client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                res = client.chat.completions.create(
                    model="glm-4-flash",
                    messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages[-4:],
                    stream=True
                )
                
                def generate_chunks():
                    for chunk in res:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                            
                reply = st.write_stream(generate_chunks())
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"API 报错: {e}")
