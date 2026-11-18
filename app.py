import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import lightkurve as lk
from astropy.timeseries import LombScargle
from openai import OpenAI
import base64
from scipy.ndimage import median_filter

# 1. 页面基础设置
st.set_page_config(page_title="TESS 双星智能诊断平台", layout="wide")

# 2. 密码锁
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

# 3. NASA 数据下载
@st.cache_data(show_spinner=False)
def download_tess_data(tic_id):
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search_result) == 0:
            return None
        lc = search_result[0].download()
        if lc is None:
            return None
        return lc.remove_nans().normalize()
    except Exception:
        return None

# 4. 物理参数反演引擎
def estimate_physical_params(period_days):
    try:
        # 假设总质量 1.5 太阳质量，计算半长轴 (R_sun)
        a_au = (1.5 * (period_days / 365.25)**2)**(1/3)
        a_rsun = a_au * 215.0 
        # 估算半径 (假设占半长轴 20%)
        radius = a_rsun * 0.2 
        # 估算温度 (经验映射)
        temp = 5500 + (1.0 - period_days) * 1000
        temp = max(4000, min(temp, 8000))
        # 估算质量 (经验关系 M ∝ R^1.2)
        mass = max(0.1, min((radius / 1.0)**1.2, 5.0))
        # 表面积 (相对太阳)
        area = radius**2
        
        return {
            "a_rsun": round(a_rsun, 2),
            "radius": round(radius, 2),
            "temp": round(temp, 0),
            "mass": round(mass, 2),
            "area": round(area, 2)
        }
    except Exception:
        return None

# 5. 主程序
if check_password():
    # 🔑 填入你的真实 API Key
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    with st.sidebar:
        st.header("🔭 NASA 数据控制台")
        tic_input = st.text_input("输入 TESS TIC 编号", value="260128333")
        window_length = st.slider("去趋势滑动窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        if st.button("🚀 获取 NASA 数据并分析", use_container_width=True):
            with st.spinner("正在从 NASA 下载真实光变数据..."):
                lc_data = download_tess_data(tic_input)
                if lc_data is None:
                    st.error("下载失败或未找到数据，请检查 TIC 编号。")
                    st.session_state.lc = None
                else:
                    st.session_state.lc = lc_data
                    st.success("数据下载成功！")

        st.markdown("---")
        if st.button("🗑️ 清空 AI 对话", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "你好！请先在左侧下载数据，然后问我物理参数或公式推导。"}]
            st.rerun()

    col_plot, col_chat = st.columns([1.3, 1])

    # ==========================================
    # 左侧：科研绘图与全参数展示
    # ==========================================
    with col_plot:
        st.subheader("📊 真实 TESS 观测与物理参数反演")
        
        if "lc" in st.session_state and st.session_state.lc is not None:
            lc = st.session_state.lc
            try:
                time_arr = lc.time.value
                flux_arr = lc.flux.value
                valid = np.isfinite(time_arr) & np.isfinite(flux_arr)
                time_arr, flux_arr = time_arr[valid], flux_arr[valid]

                # 防崩溃的窗口计算
                window_points = int(window_length * 720)
                if window_points % 2 == 0: window_points += 1
                if window_points >= len(flux_arr): window_points = max(5, len(flux_arr) - 1)
                if window_points % 2 == 0: window_points += 1

                # 中值滤波去趋势
                trend = median_filter(flux_arr, size=window_points, mode='nearest')
                flux_flat = flux_arr - trend + 1.0
                
                st.success("✅ 数据清洗与去趋势完成！")
                
                # 1. 画光变曲线
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_arr, mode='markers', marker=dict(size=2, color='gray'), name='原始观测'))
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_flat, mode='markers', marker=dict(size=2, color='#1f77b4'), name='去趋势后'))
                fig_lc.update_layout(title=f"TIC {tic_input} 光变曲线", xaxis_title="时间 (BTJD)", yaxis_title="归一化流量", height=300)
                st.plotly_chart(fig_lc, use_container_width=True)
                
                # 2. 算周期图
                flux_centered = flux_flat - np.mean(flux_flat)
                if np.all(flux_centered == 0):
                    st.warning("数据起伏过小，无法计算周期。")
                else:
                    frequency, power = LombScargle(time_arr, flux_centered).autopower(minimum_frequency=0.1, maximum_frequency=10)
                    period = 1 / frequency
                    best_period = period[np.argmax(power)]
                    
                    fig_pg = go.Figure()
                    fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', line=dict(color='#ff7f0e')))
                    fig_pg.add_vline(x=best_period, line_dash="dash", line_color="red", annotation_text=f"最高峰: {best_period:.4f}天")
                    fig_pg.update_layout(title="Lomb-Scargle 周期图", xaxis_title="周期 (天)", yaxis_title="功率", height=300)
                    st.plotly_chart(fig_pg, use_container_width=True)
                    
                    # 3. 展示全部物理参数
                    st.markdown("### 🌟 智能反演物理参数")
                    st.caption("注：基于测光经验关系估算，精确参数需结合光谱数据。")
                    params = estimate_physical_params(best_period)
                    if params:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("总质量 (M☉)", f"{params['mass']}")
                        c2.metric("温度 (K)", f"{params['temp']}")
                        c3.metric("半径 (R☉)", f"{params['radius']}")
                        
                        c4, c5 = st.columns(2)
                        c4.metric("表面积 (相对太阳)", f"{params['area']}")
                        c5.metric("轨道半长轴 (R☉)", f"{params['a_rsun']}")
                        
                        # 保存给 AI 用
                        st.session_state.mass = params['mass']
                        st.session_state.temp = params['temp']
                        st.session_state.radius = params['radius']
                        st.session_state.area = params['area']
                        st.session_state.period = best_period
                        st.session_state.tic = tic_input
            except Exception as e:
                st.error("数据处理出现异常，请尝试更换 TIC 编号。")
        else:
            st.info("👈 请在左侧输入 TIC 编号，并点击「获取 NASA 数据并分析」开始。")

    # ==========================================
    # 右侧：极速 AI 与公式渲染
    # ==========================================
    with col_chat:
        st.subheader("💬 天文专家 AI")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！请先下载数据，然后问我关于物理参数或公式的问题。"}]
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if "image" in msg:
                    st.image(msg["image"], width=200)
                st.markdown(msg["content"])
                
        uploaded_file = st.file_uploader("📷 传图", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
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
                    
                    context = ""
                    if "period" in st.session_state:
                        context = f"当前 TIC {st.session_state.tic}，周期 {st.session_state.period:.4f} 天，质量 {st.session_state.mass} 太阳质量，温度 {st.session_state.temp}K，半径 {st.session_state.radius} 太阳半径，表面积是太阳的 {st.session_state.area} 倍。"
                    
                    sys_prompt = f"""
                    你是一个资深天文学家。{context} 
                    请专业、清晰地回答用户的问题，字数控制在 150 字以内。
                    
                    【极其重要 - 公式排版要求】：
                    你输出的所有数学公式，必须严格使用 Markdown LaTeX 语法。
                    1. 独立成行的公式，必须前后使用两个美元符号包裹，例如：$$ f = \\frac{{1}}{{P}} $$
                    2. 写在文字中间的公式，前后使用一个美元符号包裹，例如：周期 $P$ 和质量 $M$。
                    3. 绝对禁止使用 \\[ \\] 或 \\( \\) 这种旧格式！
                    """
                    
                    if uploaded_file:
                        with st.spinner("AI 识别中..."):
                            img_bytes = uploaded_file.getvalue()
                            b64_img = base64.b64encode(img_bytes).decode('utf-8')
                            messages = [
                                {"role": "system", "content": "你是资深天文学家，请简短识别天文图片。"},
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt if prompt else "识别图片。"},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                                ]}
                            ]
                            res = client.chat.completions.create(model="glm-4v-flash", messages=messages)
                            st.markdown(res.choices[0].message.content)
                    else:
                        def generate_chunks():
                            res = client.chat.completions.create(
                                model="glm-4-flash",
                                messages=[{"role": "system", "content": sys_prompt}] + st.session_state.messages[-4:],
                                stream=True
                            )
                            for chunk in res:
                                if chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content
                        reply = st.write_stream(generate_chunks())
                        
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception:
                    st.warning("⚠️ 网络波动，请重新发送。")
