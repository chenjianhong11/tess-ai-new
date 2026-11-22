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
st.set_page_config(page_title="星鉴 AstroFusion 科研平台", layout="wide")

# 2. 密码锁
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

# 3. 数据下载
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

# 4. 物理参数反演
def estimate_physical_params(period_days):
    try:
        a_au = (1.5 * (period_days / 365.25)**2)**(1/3)
        a_rsun = a_au * 215.0 
        radius = a_rsun * 0.2 
        temp = 5500 + (1.0 - period_days) * 1000
        temp = max(4000, min(temp, 8000))
        mass = max(0.1, min((radius / 1.0)**1.2, 5.0))
        area = radius**2
        return {"a": round(a_rsun, 2), "r": round(radius, 2), "t": round(temp, 0), "m": round(mass, 2), "area": round(area, 2)}
    except:
        return None

# 5. 主程序
if check_password():
    # 🔑 🔑 🔑 填入你的真实 API Key 🔑 🔑 🔑
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    with st.sidebar:
        st.header("🔭 TESS 数据控制台")
        tic_input = st.text_input("输入 TESS TIC 编号", value="260128333")
        window_length = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        st.markdown("---")
        st.header("🔄 多源周期比对")
        st.caption("输入其他光学星表（如 Gaia/ZTF）给出的参考周期。")
        ref_period = st.number_input("其他星表周期 (天)", value=0.0, step=0.001, format="%.4f")
        
        st.markdown("---")
        st.header("📡 天眼 (FAST) 联合验证")
        st.caption("输入 FAST 论文中的射电轨道周期（射电数据为独立证据）。")
        fast_period = st.number_input("FAST 射电轨道周期 (天)", value=4.200, step=0.001, format="%.3f")
        
        if st.button("🚀 获取 NASA 数据并分析", use_container_width=True):
            with st.spinner("正在从 NASA 下载真实数据..."):
                lc_data = download_tess_data(tic_input)
                if lc_data is None:
                    st.error("下载失败或未找到数据。")
                    st.session_state.lc = None
                else:
                    st.session_state.lc = lc_data
                    st.success("数据下载成功！")

    col_plot, col_chat = st.columns([1.3, 1])

    # ==== 左侧：科研绘图与诊断区 ====
    with col_plot:
        st.subheader("📊 真实观测与周期诊断")
        
        if "lc" in st.session_state and st.session_state.lc is not None:
            lc = st.session_state.lc
            try:
                time_arr = lc.time.value
                flux_arr = lc.flux.value
                valid = np.isfinite(time_arr) & np.isfinite(flux_arr)
                time_arr, flux_arr = time_arr[valid], flux_arr[valid]

                window_points = int(window_length * 720)
                if window_points % 2 == 0: window_points += 1
                if window_points >= len(flux_arr): window_points = max(5, len(flux_arr) - 1)
                if window_points % 2 == 0: window_points += 1

                trend = median_filter(flux_arr, size=window_points, mode='nearest')
                flux_flat = flux_arr - trend + 1.0
                
                # 1. 画光变曲线
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_arr, mode='markers', marker=dict(size=2, color='gray'), name='原始观测'))
                fig_lc.add_trace(go.Scattergl(x=time_arr, y=flux_flat, mode='markers', marker=dict(size=2, color='#1f77b4'), name='去趋势后'))
                fig_lc.update_layout(title=f"TIC {tic_input} 光变曲线", xaxis_title="时间 (BTJD)", yaxis_title="流量", height=280)
                st.plotly_chart(fig_lc, use_container_width=True)
                
                # 2. 周期图与谐波诊断
                flux_centered = flux_flat - np.mean(flux_flat)
                if np.all(flux_centered == 0):
                    st.warning("数据起伏过小，无法计算周期。")
                else:
                    frequency, power = LombScargle(time_arr, flux_centered).autopower(minimum_frequency=0.1, maximum_frequency=10)
                    period = 1 / frequency
                    
                    max_idx = np.argmax(power)
                    best_period = period[max_idx]
                    max_power = power[max_idx]
                    
                    half_period = best_period / 2.0
                    half_idx = np.argmin(np.abs(period - half_period))
                    half_power = power[half_idx]
                    
                    fig_pg = go.Figure()
                    fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', line=dict(color='#ff7f0e'), name='功率谱'))
                    fig_pg.add_vline(x=best_period, line_dash="dash", line_color="red", annotation_text=f"主峰: {best_period:.4f}天")
                    fig_pg.add_vline(x=half_period, line_dash="dot", line_color="purple", annotation_text=f"半周期: {half_period:.4f}天")
                    fig_pg.update_layout(title="Lomb-Scargle 周期图", xaxis_title="周期 (天)", yaxis_title="功率", height=280)
                    st.plotly_chart(fig_pg, use_container_width=True)
                    
                    # 3. 智能诊断报告
                    st.markdown("### 🩺 智能诊断报告")
                    if half_power > max_power * 0.7:
                        st.error(f"⚠️ **谐波误判预警**：检测到半周期（{half_period:.4f}天）处的功率极高。真实轨道周期可能是 {best_period:.4f} 天，也可能是 {half_period:.4f} 天。需结合其他证据复核！")
                    else:
                        st.success(f"✅ **周期判定可靠**：半周期处功率较低，无明显谐波混叠风险。最高可信周期为 {best_period:.4f} 天。")
                    
                    # 4. 多源比对诊断
                    if ref_period > 0:
                        st.markdown("### 🔄 多源周期比对")
                        ratio = best_period / ref_period
                        if 0.95 <= ratio <= 1.05:
                            st.success(f"🟢 **跨源一致**：TESS 周期 ({best_period:.4f} 天) 与参考周期 ({ref_period:.4f} 天) 高度吻合。")
                        elif 0.45 <= ratio <= 0.55:
                            st.warning(f"🟡 **疑似半周期偏差**：TESS 周期大约是参考周期的一半。请检查是否发生了 P/2 误判。")
                        elif 1.9 <= ratio <= 2.1:
                            st.warning(f"🟡 **疑似双倍周期偏差**：TESS 周期大约是参考周期的两倍。")
                        else:
                            st.error(f"🔴 **严重冲突**：两者周期差距巨大 (比值 {ratio:.2f})，可能存在采样混叠或观测到了不同的物理现象。")
                    
                    # 5. 天眼 (FAST) 联合验证诊断
                    st.markdown("### 📡 光学-射电轨道一致性诊断 (FAST)")
                    if fast_period > 0:
                        fast_ratio = best_period / fast_period
                        if 0.98 <= fast_ratio <= 1.02:
                            st.success(f"🌟 **光射一致**：TESS 光学周期 ({best_period:.4f} 天) 与 FAST 射电轨道周期 ({fast_period:.3f} 天) 高度吻合！")
                        elif 0.48 <= fast_ratio <= 0.52:
                            st.error(f"🚨 **光射冲突 (半周期)**：光学周期是 FAST 周期的一半！这极大概率是 P/2 误判。应以 FAST 射电轨道为准！")
                        elif 1.95 <= fast_ratio <= 2.05:
                            st.error(f"🚨 **光射冲突 (双倍周期)**：光学周期是 FAST 周期的两倍！可能存在采样混叠。")
                        else:
                            st.warning(f"⚠️ **光射偏差**：光学周期与 FAST 周期存在显著差异 (比值 {fast_ratio:.2f})，该目标可能需要进行更多的射电计时观测。")
                    
                    # 6. 显示参数
                    st.markdown("### 🌟 智能反演物理参数")
                    params = estimate_physical_params(best_period)
                    if params:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("总质量 (M☉)", f"{params['m']}")
                        c2.metric("温度 (K)", f"{params['t']}")
                        c3.metric("半径 (R☉)", f"{params['r']}")
                        st.session_state.params = params
                    
                    st.session_state.period = best_period
                    st.session_state.half_power_ratio = half_power / max_power
                    st.session_state.tic = tic_input
                    st.session_state.ref_period = ref_period
                    st.session_state.fast_period = fast_period
                    
            except Exception as e:
                st.error(f"数据处理异常，请尝试更换 TIC 编号。")
        else:
            st.info("👈 请在左侧输入 TIC 编号并点击下载。")

    # ==== 右侧：AI 对话区 ====
    with col_chat:
        st.subheader("💬 天文专家 AI")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！系统已就绪。下载数据后，你可以问我关于谐波诊断、多源比对、天眼验证或物理参数的问题。"}]
            
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
                        context = f"当前分析 TIC {st.session_state.tic}。系统计算主峰周期 {st.session_state.period:.4f} 天，半周期功率占比 {st.session_state.half_power_ratio*100:.1f}%。"
                        if st.session_state.ref_period > 0:
                            context += f" 其他星表参考周期 {st.session_state.ref_period:.4f} 天。"
                        if st.session_state.fast_period > 0:
                            context += f" FAST 射电轨道周期 {st.session_state.fast_period:.3f} 天。"
                    
                    sys_prompt = f"""
                    你是一个资深天文学家。{context} 
                    请专业、清晰地回答用户问题（150字以内）。
                    如果发现光射冲突（如相差两倍/一半），请重点解释 FAST 射电数据的独立证据价值，并说明 P/2 误判的物理原因。公式必须使用 $$ 包裹。
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
