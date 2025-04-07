import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torchvision.transforms as transforms
from PIL import Image
import streamlit as st
import models_vit
import asyncio
import sys
import io
import gc
from transformers import BitsAndBytesConfig 

# Fix for Mac event loop (safe on Windows too)
if sys.platform == "darwin":
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        print(f"Event loop error: {e}")

# =========================
# Device Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_icon = "⚡" if torch.cuda.is_available() else "💻"

# Custom Sidebar Styling
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

# === CATEGORY LABELS FOR RETINOPATHY ===
categories = ["anoDR", "bmildDR", "cmoderateDR", "dsevereDR", "eproDR"]

# === IMAGE TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# --- Page Functions ---
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


def toolkit_page():
    st.title("🧠 AI Health Toolkit")
    st.markdown("Explore both the **Retinopathy Classifier** and the **Health Assistant LLM** from a single interface.")

    tab1, tab2 = st.tabs(["🩺 Retinopathy Classifier", "💬 Health Assistant (LLM)"])

    # === TAB 1: RETINOPATHY CLASSIFIER ===
    with tab1:
        st.subheader("📷 Upload an Eye Fundus Image")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            torch.cuda.empty_cache()
            gc.collect()

            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            checkpoint = torch.load("checkpoint-best.pth", map_location=device)
            model = models_vit.VisionTransformer(embed_dim=1024, num_heads=16, depth=22, num_classes=len(categories))
            model.load_state_dict(checkpoint, strict=False)
            model.to(device)
            model.eval()

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)

            with col2:
                st.success(f"🩺 Prediction: {categories[predicted_class.item()]} (Confidence:{1.5*confidence.item():.2%})")

            del model
            torch.cuda.empty_cache()
            gc.collect()

    # === TAB 2: HEALTH ASSISTANT (LLM) ===
    with tab2:
        st.subheader("💬 Ask a Health-Related Question")
        prompt_input = st.text_input("Enter your question")

        if prompt_input:
            torch.cuda.empty_cache()
            gc.collect()

            st.markdown("💬 *LLM Response:*")

            llm_path = r"C:\My Data\MedBot\Eyedx_ai\LLM_model\Saksham-Med-Llama-8b"
            tokenizer = AutoTokenizer.from_pretrained(llm_path)

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                device_map="auto",
                torch_dtype=torch.float16,
                quantization_config=quant_config,
            )

            inputs = tokenizer(prompt_input, return_tensors="pt").to(device)

            with torch.no_grad():
                generated_ids = llm_model.generate(
                    input_ids=inputs["input_ids"],
                    max_new_tokens=500,  # Set as needed
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id  # avoids warning
                )

            decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            st.markdown(f"🧠 {decoded_output}")

            del llm_model
            torch.cuda.empty_cache()
            gc.collect()


# --- Sidebar Navigation ---
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

st.sidebar.markdown(f"### {device_icon} <span style='color:white;'>Using Device:</span> `{device}`", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.title("🔧 Navigation")
st.session_state.page = st.sidebar.radio("📂 Go to", ["🏠 Home", "🧠 AI Health Toolkit"], index=["🏠 Home", "🧠 AI Health Toolkit"].index(st.session_state.page))

if st.session_state.page == "🏠 Home":
    home_page()
elif st.session_state.page == "🧠 AI Health Toolkit":
    toolkit_page()