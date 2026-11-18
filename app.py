import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import lightkurve as lk
from astropy.timeseries import LombScargle
from openai import OpenAI
import base64
from scipy.ndimage import median_filter

# ==========================================
# 1. 页面设置与 UI
# ==========================================
st.set_page_config(page_title="TESS 双星智能诊断平台", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# ==========================================
# 2. 密码锁
# ==========================================
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.title("🔒 TESS 科研智能诊断平台")
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
# 3. NASA 数据下载
# ==========================================
@st.cache_data(show_spinner=False)
def download_tess_data(tic_id):
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search_result) == 0:
            return None
        lc = search_result[0].download()
        if lc is None:
            return None
        lc = lc.remove_nans().normalize()
        return lc
    except Exception:
        return None

# ==========================================
# 4. 物理参数反演引擎（核心科研计算）
# ==========================================
def estimate_physical_params(period_days):
    try:
        # 1. 估算轨道半长轴 a (单位: 太阳半径 R_sun)
        # 假设双星系统总质量约为 1.5 太阳质量
        M_total_assumed = 1.5 
        a_au = (M_total_assumed * (period_days / 365.25)**2)**(1/3)
        a_rsun = a_au * 215.0 
        
        # 2. 估算单颗恒星半径 (假设为接触/半接触双星，半径占轨道半长轴约 20%)
        radius_est = a_rsun * 0.2 
        
        # 3. 估算有效温度 (基于短周期双星经验关系)
        temp_est = 5500 + (1.0 - period_days) * 1000
        if temp_est < 4000: temp_est = 4000
        if temp_est > 8000: temp_est = 8000
        
        # 4. 估算恒星质量 (基于质量-半径经验关系 M ∝ R^1.2)
        mass_est = (radius_est / 1.0)**1.2
        if mass_est < 0.1: mass_est = 0.1
        if mass_est > 5.0: mass_est = 5.0
        
        # 5. 计算恒星表面积 (相对太阳表面积的倍数)
        # 因为半径已经是 R_sun，所以表面积 = R^2
        surface_area_est = radius_est**2
        
        return {
            "a_rsun": round(a_rsun, 2),
            "radius": round(radius_est, 2),
            "temp": round(temp_est, 0),
            "mass": round(mass_est, 2),
            "area": round(surface_area_est, 2)
        }
    except Exception:
        return None

# ==========================================
# 5. 主程序
# ==========================================
if check_password():
    # 🔑 填入你的真实 API Key
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    with st.sidebar:
        st.header("🔭 数据与参数控制台")
        tic_input = st.text_input("输入 TESS TIC 编号", value="260128333")
        window_length = st.slider("去趋势滑动窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        if st.button("🚀 获取 NASA 数据并分析", use_container_width=True):
            with st.spinner("正在从 NASA 获取真实光变数据..."):
                lc_data = download_tess_data(tic_input)
                if lc_data is None:
                    st.error("下载失败或未找到数据，请检查 TIC 编号。")
                    st.session_state.lc = None
                else:
                    st.session_state.lc = lc_data
                    st.success("数据下载成功！")

        st.markdown("---")
        if st.button("🗑️ 清空 AI 对话", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "你好！请先在左侧下载真实数据。你可以问我关于周期、去趋势或物理参数的问题。"}]
            st.rerun()

    col_plot, col_chat = st.columns([1.3, 1])

    # ==========================================
    # 左侧：科研绘图与参数区
    # ==========================================
    with col_plot:
        st.subheader("📊 真实 TESS 观测与物理参数反演")
        
        if "lc" in st.session_state and st.session_state.lc is not None:
            lc = st.session_state.lc
            
            try:
                time_arr = lc.time.value
                flux_arr = lc.flux.value
                valid_mask = np.isfinite(time_arr) & np.isfinite(flux_arr)
                dropped_points = len(time_arr) - np.sum(valid_mask)
                time_arr = time_arr[valid_mask]
                flux_arr = flux_arr[valid_mask]

                window_points = int(window_length * 720)
                if window_points % 2 == 0: window_points += 1
                if window_points >= len(flux_arr): window_points = max(5, len(flux_arr) - 1)
                if window_points % 2 == 0: window_points += 1

                trend = median_filter(flux_arr, size=window_points, mode='nearest')
                flux_flat = flux_arr - trend + 1.0
                
                st.success(f"✅ 鲁棒性处理完成：已自动过滤 {dropped_points} 个异常数据点")
                
                # 1. 光变曲线图
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_arr, mode='markers', marker=dict(size=2.5, color='rgba(160, 160, 160, 0.4)'), name='原始观测 (Raw)'))
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_flat, mode='markers', marker=dict(size=2.5, color='rgba(31, 119, 180, 0.7)'), name='去趋势后 (Flattened)'))
                fig_lc.update_layout(
                    title=f"TIC {tic_input} 光变曲线", 
                    xaxis_title="时间 (BTJD)", yaxis_title="归一化流量", 
                    height=330, margin=dict(l=40, r=20, t=50, b=40),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_lc, use_container_width=True)
                
                # 2. 周期图 + 物理参数
                try:
                    flux_centered = flux_flat - np.mean(flux_flat)
                    if np.all(flux_centered == 0):
                        st.warning("数据起伏过小，无法计算周期图。")
                    else:
                        frequency, power = LombScargle(time_arr, flux_centered).autopower(minimum_frequency=0.1, maximum_frequency=10)
                        period = 1 / frequency
                        
                        max_power_idx = np.argmax(power)
                        best_period = period[max_power_idx]
                        max_power = power[max_power_idx]
                        
                        fig_pg = go.Figure()
                        fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', line=dict(color='#FFA15A', width=2), name='Lomb-Scargle Power'))
                        fig_pg.add_trace(go.Scatter(x=[best_period], y=[max_power], mode='markers', marker=dict(color='#EF553B', size=12, symbol='star'), name='最高峰'))
                        
                        fig_pg.update_layout(
                            title="Lomb-Scargle 周期图 (寻找最高峰)", 
                            xaxis_title="周期 (天)", yaxis_title="功率",
                            height=330, margin=dict(l=40, r=20, t=50, b=40),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=False),
                            yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.15)', zeroline=False),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        fig_pg.add_vline(x=best_period, line_dash="dash", line_color="#EF553B", opacity=0.5)
                        
                        st.plotly_chart(fig_pg, use_container_width=True)
                        
                        # --- 物理参数卡片展示 ---
                        st.markdown("### 🌟 智能反演物理参数")
                        st.caption("注：基于测光经验关系估算，精确参数需结合光谱视向速度解轨。")
                        
                        params = estimate_physical_params(best_period)
                        if params:
                            # 第一排：质量、温度、半径
                            p_col1, p_col2, p_col3 = st.columns(3)
                            p_col1.metric("总质量 (M☉)", f"{params['mass']}")
                            p_col2.metric("有效温度 (K)", f"{params['temp']}")
                            p_col3.metric("恒星半径 (R☉)", f"{params['radius']}")
                            
                            # 第二排：表面积、轨道半长轴
                            p_col4, p_col5 = st.columns(2)
                            p_col4.metric("恒星表面积 (相对太阳)", f"{params['area']}")
                            p_col5.metric("轨道半长轴 (R☉)", f"{params['a_rsun']}")
                            
                            # 保存给 AI 用
                            st.session_state.est_mass = params['mass']
                            st.session_state.est_temp = params['temp']
                            st.session_state.est_radius = params['radius']
                            st.session_state.est_area = params['area']
                        
                        st.session_state.current_period = best_period
                        st.session_state.tic_id = tic_input
                except Exception:
                    st.warning("周期图或参数估算遇到异常。")
                    
            except Exception:
                st.error("处理天文数据时遇到未知格式，请尝试更换 TIC 编号。")
        else:
            st.info("👈 请在左侧输入 TIC 编号，并点击「获取 NASA 数据并分析」开始。")

    # ==========================================
    # 右侧：极速 AI 区
    # ==========================================
    with col_chat:
        st.subheader("💬 天文专家 AI (极速版)")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！系统已就绪，可以反演质量、半径、温度、表面积等参数。请先在左侧下载数据！"}]
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if "image" in msg:
                    st.image(msg["image"], width=200)
                st.markdown(msg["content"])
                
        uploaded_file = st.file_uploader("📷 上传天文照片", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        prompt = st.chat_input("输入科研问题...")
        
        if prompt or uploaded_file:
            user_msg = {"role": "user", "content": prompt if prompt else "请分析这张图片。"}
            with st.chat_message("user"):
                if uploaded_file: 
                    st.image(uploaded_file, width=200)
                    user_msg["image"] = uploaded_file
                st.markdown(user_msg["content"])
            st.session_state.messages.append(user_msg)
            
            with st.chat_message("assistant"):
                try:
                    client = OpenAI(api_key=ZHIPU_API_KEY, base_url=BASE_URL)
                    
                    recent_messages = st.session_state.messages[-4:]
                    context = ""
                    if "current_period" in st.session_state:
                        context = f"当前 TIC {st.session_state.tic_id}，轨道周期 {st.session_state.current_period:.4f} 天，估算总质量 {st.session_state.est_mass} 太阳质量，表面温度 {st.session_state.est_temp}K，恒星半径 {st.session_state.est_radius} 太阳半径，表面积是太阳的 {st.session_state.est_area} 倍。"
                    
                    sys_prompt = f"你是资深天文学家。{context} 请简短、专业地回答（不超过 150 字）。公式用 $$ 包裹。"
                    
                    if uploaded_file:
                        with st.spinner("AI 快速识别中..."):
                            image_bytes = uploaded_file.getvalue()
                            base64_image = base64.b64encode(image_bytes).decode('utf-8')
                            messages_for_ai = [
                                {"role": "system", "content": "你是资深天文学家，请简短识别天文图片。"},
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt if prompt else "识别图片。"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                                ]}
                            ]
                            res = client.chat.completions.create(model="glm-4v-flash", messages=messages_for_ai)
                            reply = res.choices[0].message.content
                            st.markdown(reply)
                    else:
                        def generate_chunks():
                            res = client.chat.completions.create(
                                model="glm-4-flash",
                                messages=[{"role": "system", "content": sys_prompt}] + recent_messages,
                                stream=True
                            )
                            for chunk in res:
                                if chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content
                        reply = st.write_stream(generate_chunks())
                        
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception:
                    st.warning("⚠️ 网络波动，请重试。")
