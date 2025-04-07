# frontend.py
import streamlit as st
import requests
import sys
import time
# === Device Info (symbol only, backend handles real inference) ===
import torch
device_icon = "⚡" if torch.cuda.is_available() else "💻"

# Backend URL
# BASE_URL = "http://localhost:8000"  # Update with your local IP if accessed from other devices
BASE_URL = "http://172.16.150.95:8000"  # Replace with your actual backend IP

# Mac event loop fix (safe cross-platform)
import asyncio
if sys.platform == "darwin":
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        print(f"Event loop error: {e}")

# === Sidebar Styling ===
sidebar_style = """
    <style>
        [data-testid=stSidebar] {
            background: linear-gradient(135deg, #1f4068, #2a5298);
            color: white;
            padding: 2rem 1rem;
        }
        [data-testid=stSidebar] h1, 
        [data-testid=stSidebar] h2, 
        [data-testid=stSidebar] h3, 
        [data-testid=stSidebar] h4, 
        [data-testid=stSidebar] h5, 
        [data-testid=stSidebar] h6, 
        [data-testid=stSidebar] p, 
        [data-testid=stSidebar] label {
            color: white !important;
        }
        [data-testid=stSidebar] .stButton>button {
            color: #2a5298;
            background-color: white;
            border-radius: 0.5rem;
            transition: 0.3s ease;
        }
        [data-testid=stSidebar] .stButton>button:hover {
            background-color: #1f4068;
            color: white;
            border: 1px solid white;
        }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# === Home Page ===
def home_page():
    st.title("👁️ Welcome to RIC AI - AI Health Assistant")
    st.markdown("### 🤖 Revolutionizing Healthcare with AI")

    st.markdown("""
    RIC AI is an advanced medical assistant powered by artificial intelligence that offers:
    - Instant **health consultations** via an intelligent chatbot  
    - Accurate **eye disease detection** through deep learning models  
    - A smart, simple **dashboard** to track health evaluations  
    """)

    st.divider()

    st.subheader("📌 Project Overview")
    st.info("""
    Millions suffer from undiagnosed eye diseases and lack timely medical advice. RIC AI bridges this gap by providing:
    - ✅ Instant AI-driven consultations  
    - ✅ Real-time image-based eye screening  
    - ✅ A secure and private user experience  
    """)

    st.subheader("🎯 Objectives")
    st.markdown("""
    - 📌 Deliver real-time medical insights via an AI chatbot  
    - 📌 Detect diabetic retinopathy using fundus images  
    - 📌 Improve health awareness through accessible AI  
    - 📌 Ensure an intuitive and engaging interface  
    """)

    st.subheader("🛠️ How It Works")
    st.markdown("""
    1. Visit the app and select **Try This AI**  
    2. Ask health questions via the **Health Assistant**  
    3. Upload eye images to detect conditions like diabetic retinopathy  
    4. View predictions and results instantly  
    """)

    st.subheader("⚙️ Key Features")
    cols = st.columns(2)
    with cols[0]:
        st.success("✅ AI Health Chatbot")
        st.success("✅ Fundus Image Classifier")
    with cols[1]:
        st.success("✅ Smart Result Dashboard")
        st.success("✅ Privacy-Focused Design")

    st.subheader("🚀 Future Scope")
    st.markdown("""
    - 🔹 Improve model accuracy for broader diagnosis  
    - 🔹 Add voice-based health assistance  
    - 🔹 Enable multi-disease AI analysis  
    - 🔹 Integrate with real doctors and clinics  
    """)

    st.divider()
    st.markdown("### 🌎 Ready to Experience AI Healthcare?")
    if st.button("🚀 Try This AI"):
        st.session_state.page = "🧠 AI Health Toolkit"

# === Toolkit Page ===
def toolkit_page():
    st.title("🧠 AI Health Toolkit")
    st.markdown("Explore both the **Retinopathy Classifier** and the **Health Assistant LLM** from a single interface.")

    tab1, tab2 = st.tabs(["🩺 Retinopathy Classifier", "💬 Health Assistant (LLM)"])

    # === Retinopathy Classifier Tab ===
    with tab1:
        st.subheader("📷 Upload an Eye Fundus Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            with st.spinner("Predicting..."):
                files = {"file": uploaded_file.getvalue()}
                try:
                    response = requests.post(f"{BASE_URL}/predict_retinopathy/", files=files)
                    if response.ok:
                        result = response.json()
                        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
                        st.success(f"🩺 Prediction: {result['prediction']} (Confidence: {result['confidence']:.2%})")
                    else:
                        st.error("Prediction failed.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # === LLM Assistant Tab ===
    # === LLM Assistant Tab ===
        with tab2:
            st.subheader("💬 Ask a Health-Related Question")
            user_input = st.text_input("Enter your question")

            if user_input:
                with st.spinner("Generating response..."):
                    try:
                        with requests.post(f"{BASE_URL}/generate_stream/", json={"question": user_input}, stream=True) as r:
                            if r.status_code == 200:
                                full_response = ""
                                response_placeholder = st.empty()
                                for chunk in r.iter_content(chunk_size=None):
                                    if chunk:
                                        decoded = chunk.decode("utf-8")
                                        full_response += decoded
                                        response_placeholder.markdown(f"🧠 {full_response}")
                            else:
                                st.error("Failed to get a streaming response.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")


# === Navigation ===
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

st.sidebar.markdown(f"### {device_icon} <span style='color:white;'>Using Device:</span> `{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}`", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.title("🔧 Navigation")
st.session_state.page = st.sidebar.radio("📂 Go to", ["🏠 Home", "🧠 AI Health Toolkit"], index=["🏠 Home", "🧠 AI Health Toolkit"].index(st.session_state.page))

# === Render Page ===
if st.session_state.page == "🏠 Home":
    home_page()
elif st.session_state.page == "🧠 AI Health Toolkit":
    toolkit_page()
