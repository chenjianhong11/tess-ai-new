import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import lightkurve as lk
from astropy.timeseries import LombScargle
from openai import OpenAI
import base64

st.set_page_config(page_title="TESS 科研级 AI 平台", layout="wide", initial_sidebar_state="expanded")

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False
    if not st.session_state["password_correct"]:
        st.title("🔒 欢迎访问 TESS AI 科研平台")
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

@st.cache_data(show_spinner=False)
def download_tess_data(tic_id):
    try:
        search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(search_result) == 0:
            return None
        lc = search_result[0].download()
        lc = lc.remove_nans().normalize()
        return lc
    except Exception as e:
        return str(e)

if check_password():
    # 🔑 填入你的真实 API Key
    ZHIPU_API_KEY = "ebe2cfe96ecd4ed08adc2d9054349c7f.yZBRP0B46y0g3Nb7"  
    BASE_URL = "https://open.bigmodel.cn/api/paas/v4"

    with st.sidebar:
        st.header("🔭 NASA 数据控制台")
        tic_input = st.text_input("输入 TESS TIC 编号", value="260128333")
        window_length = st.slider("去趋势滑动窗口 (天)", 0.1, 2.0, 0.5, 0.1)
        
        if st.button("🚀 下载真实数据并分析", use_container_width=True):
            with st.spinner("正在从 NASA 下载真实光变数据..."):
                lc_data = download_tess_data(tic_input)
                if isinstance(lc_data, str):
                    st.error(f"下载失败: {lc_data}")
                elif lc_data is None:
                    st.error("未找到该星体的 TESS 数据！")
                else:
                    st.session_state.lc = lc_data
                    st.success("下载成功！")

        st.markdown("---")
        if st.button("🗑️ 清空 AI 对话", use_container_width=True):
            st.session_state.messages = [{"role": "assistant", "content": "你好！请先下载真实数据，然后问我任何天文问题，或上传图片！"}]
            st.rerun()

    col_plot, col_chat = st.columns([1.3, 1])

    with col_plot:
        st.subheader("📊 真实 TESS 观测数据与周期图")
        if "lc" in st.session_state:
            lc = st.session_state.lc
            
            # 动态计算安全的窗口长度，防止报错
            n_points = len(lc)
            calc_len = int(window_length * 720) 
            if calc_len % 2 == 0: calc_len += 1 
            safe_window = min(calc_len, max(5, (n_points // 5) | 1))
            
            try:
                lc_flat = lc.flatten(window_length=safe_window)
            except Exception:
                lc_flat = lc.flatten(window_length=min(101, (n_points // 2) | 1))
            
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=lc.time.value, y=lc.flux.value, mode='markers', marker=dict(size=2, color='gray'), name='原始观测'))
            fig_lc.add_trace(go.Scatter(x=lc_flat.time.value, y=lc_flat.flux.value, mode='markers', marker=dict(size=2, color='#1f77b4'), name='去趋势后'))
            fig_lc.update_layout(title=f"TIC {tic_input} 光变曲线", xaxis_title="时间 (BTJD)", yaxis_title="归一化流量", height=350)
            st.plotly_chart(fig_lc, use_container_width=True)
            
            time_arr = lc_flat.time.value
            flux_arr = lc_flat.flux.value - np.mean(lc_flat.flux.value)
            frequency, power = LombScargle(time_arr, flux_arr).autopower(minimum_frequency=0.1, maximum_frequency=10)
            period = 1 / frequency
            
            fig_pg = go.Figure()
            fig_pg.add_trace(go.Scatter(x=period, y=power, mode='lines', name='Lomb-Scargle Power', line=dict(color='#ff7f0e')))
            fig_pg.update_layout(title="Lomb-Scargle 周期图", xaxis_title="周期 (天)", yaxis_title="功率", height=350)
            
            max_power_idx = np.argmax(power)
            best_period = period[max_power_idx]
            fig_pg.add_vline(x=best_period, line_dash="dash", line_color="red", annotation_text=f"最高峰: {best_period:.3f}天")
            st.plotly_chart(fig_pg, use_container_width=True)
            
            st.session_state.current_period = best_period
            st.session_state.window_length = window_length
            st.session_state.tic_id = tic_input
        else:
            st.info("👈 请在左侧输入 TIC 编号，并点击「下载真实数据并分析」开始。")

    with col_chat:
        st.subheader("💬 天文专家 AI")
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "你好！请先在左侧下载 NASA 真实数据，然后问我关于周期、去趋势的问题，或者上传图片让我识别！"}]
            
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
                    if "current_period" in st.session_state:
                        context = f"【实时科研背景】当前分析 TIC {st.session_state.tic_id}。去趋势窗口：{st.session_state.window_length} 天。最高峰周期：{st.session_state.current_period:.4f} 天。"
                    sys_prompt = f"你是一个资深天文学家。{context} 请专业地回答用户。"
                    
                    if uploaded_file:
                        with st.spinner("AI 正在识别天文图像..."):
                            image_bytes = uploaded_file.getvalue()
                            base64_image = base64.b64encode(image_bytes).decode('utf-8')
                            messages_for_ai = [
                                {"role": "system", "content": "你是一个资深天文学家。请识别天文图片。"},
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt if prompt else "分析图片"},
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
                                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
                                stream=True
                            )
                            for chunk in res:
                                if chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content
                        reply = st.write_stream(generate_chunks())
                        
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.rerun()
                except Exception as e:
                    st.error(f"API 报错: {e}")
