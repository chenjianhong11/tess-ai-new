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
st.set_page_config(page_title="TESS 科研智能诊断平台", layout="wide")

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

# 3. 数据下载与清洗（最稳的防暴死逻辑）
@st.cache_data(show_spinner=False)
def get_clean_data(tic_id):
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search_result) == 0:
            return None, "未找到该星体的数据"
        
        lc = search_result[0].download()
        if lc is None:
            return None, "下载失败"
            
        lc = lc.remove_nans().normalize()
        
        # 极致的清洗：去掉所有无穷大和异常极值
        t = lc.time.value
        f = lc.flux.value
        mask = np.isfinite(t) & np.isfinite(f)
        t, f = t[mask], f[mask]
        
        if len(t) < 100:
            return None, "有效数据点过少"
            
        return (t, f), "成功"
    except Exception as e:
        return None, "网络或服务器错误"

# 4. 主程序
if check_password():
    # 🔑 填入你的真实 API Key！
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    with st.sidebar:
        st.header("🔭 NASA 数据控制台")
        tic_input = st.text_input("输入 TESS TIC 编号", value="260128333")
        window_length = st.slider("去趋势窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        if st.button("🚀 获取真实数据并分析", use_container_width=True):
            with st.spinner("正在从 NASA 获取并清洗数据..."):
                data, msg = get_clean_data(tic_input)
                if data is None:
                    st.error(msg)
                    st.session_state.data = None
                else:
                    st.session_state.data = data
                    st.success("数据清洗完成，绝对安全！")

        st.markdown("---")
        if st.button("🗑️ 清空 AI 对话", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "你好！请先下载数据，然后问我关于这颗星的任何问题。"}]
            st.rerun()

    col_plot, col_chat = st.columns([1.3, 1])

    with col_plot:
        st.subheader("📊 真实观测与周期图")
        
        if "data" in st.session_state and st.session_state.data is not None:
            t, f = st.session_state.data
            
            # 超级稳妥的滤波
            window_points = int(window_length * 720)
            if window_points % 2 == 0: window_points += 1
            if window_points >= len(f): window_points = max(5, len(f) - 1)
            if window_points % 2 == 0: window_points += 1

            try:
                trend = median_filter(f, size=window_points, mode='nearest')
                f_flat = f - trend + 1.0
            except:
                f_flat = f # 极端情况下兜底，直接不滤波
                st.warning("去趋势计算跳过，使用原始数据。")
            
            # 画图
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scattergl(x=t, y=f, mode='markers', marker=dict(size=2, color='gray'), name='原始数据'))
            fig_lc.add_trace(go.Scattergl(x=t, y=f_flat, mode='markers', marker=dict(size=2, color='#1f77b4'), name='去趋势后'))
            fig_lc.update_layout(title=f"TIC {tic_input} 光变曲线", height=300)
            st.plotly_chart(fig_lc, use_container_width=True)
            
            # 算周期图
            try:
                f_centered = f_flat - np.mean(f_flat)
                freq, power = LombScargle(t, f_centered).autopower(minimum_frequency=0.1, maximum_frequency=10)
                period = 1 / freq
                best_period = period[np.argmax(power)]
                
                fig_pg = go.Figure()
                fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', line=dict(color='#ff7f0e')))
                fig_pg.add_vline(x=best_period, line_dash="dash", line_color="red", annotation_text=f"周期: {best_period:.4f}天")
                fig_pg.update_layout(title="Lomb-Scargle 周期图", height=300)
                st.plotly_chart(fig_pg, use_container_width=True)
                
                # 保存结果给 AI
                st.session_state.period = best_period
                st.session_state.tic = tic_input
                
                st.success(f"✅ 分析完成！最高峰周期为 {best_period:.4f} 天。去右侧问 AI 吧！")
            except:
                st.warning("周期图无法计算（数据起伏过小）。")
        else:
            st.info("👈 请在左侧输入 TIC 编号并点击下载。")

    with col_chat:
        st.subheader("💬 天文专家 AI")
        
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！请先在左侧下载数据。"}]
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if "image" in msg:
                    st.image(msg["image"], width=200)
                st.markdown(msg["content"])
                
        uploaded_file = st.file_uploader("📷 传图", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        prompt = st.chat_input("输入问题...")
        
        if prompt or uploaded_file:
            user_msg = {"role": "user", "content": prompt if prompt else "分析图片。"}
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
                        context = f"当前 TIC {st.session_state.tic}，计算出的最高峰周期为 {st.session_state.period:.4f} 天。"
                    
                    sys_prompt = f"你是资深天文学家。{context} 请简短、专业回答（150字内）。公式用 $$ 包裹。"
                    
                    if uploaded_file:
                        with st.spinner("识别中..."):
                            img_bytes = uploaded_file.getvalue()
                            b64_img = base64.b64encode(img_bytes).decode('utf-8')
                            messages = [
                                {"role": "system", "content": "你是天文学家，请简短识别天文图片。"},
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
