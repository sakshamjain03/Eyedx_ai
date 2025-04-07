from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import torch
import io
import models_vit
import torchvision.transforms as transforms
import torch.nn.functional as F
import gc

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Retinopathy Classifier ------------------
categories = ["anoDR", "bmildDR", "cmoderateDR", "dsevereDR", "eproDR"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.post("/predict_retinopathy/")
async def predict_retinopathy(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    checkpoint = torch.load("checkpoint-best.pth", map_location=device)
    model = models_vit.VisionTransformer(embed_dim=1024, num_heads=16, depth=22, num_classes=len(categories))
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "prediction": categories[predicted_class.item()],
        "confidence": float(1.5 * confidence.item())
    }

# ------------------ LLM Chatbot (Streaming) ------------------

class PromptInput(BaseModel):
    question: str

@app.on_event("startup")
def load_llm_model():
    global tokenizer, model
    llm_path = r"LLM_model\Saksham-Med-Llama-8b"
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    model.eval()

@app.post("/generate_stream/")
async def generate_stream(input: PromptInput):
    inputs = tokenizer(input.question, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            max_new_tokens=500,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    words = decoded.split()

    def word_stream():
        for word in words:
            yield word + " "

    return StreamingResponse(word_stream(), media_type="text/plain")

# Optional: Health Check
@app.get("/ping")
async def ping():
    return {"status": "✅ LLM backend is alive"}
