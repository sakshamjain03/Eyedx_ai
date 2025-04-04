from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import models_vit
import io
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

app = FastAPI()

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["http://localhost:5173"] for Vue/React dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Image Classification Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "checkpoint-best.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

image_model = models_vit.VisionTransformer(embed_dim=1024, num_heads=16, depth=22, num_classes=5)
image_model.load_state_dict(checkpoint, strict=False)
image_model.to(device)
image_model.eval()

categories = ["anoDR", "bmildDR", "cmoderateDR", "dsevereDR", "eproDR"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = image_model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)

        return JSONResponse(content={
            "category": categories[predicted_class.item()],
            "confidence": f"{confidence.item() * 100:.2f}%"
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

model_path = r"C:\My Data\New folder (2)\Eyedx_ai\LLM_model\Saksham-Med-Llama-8b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with 8-bit quantization for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use FP16 for RTX 4060
    device_map="auto"  # Automatically assigns model to GPU if available
)


@app.post("/generate/")
async def generate(request: dict):
    try:
        user_input = request.get("instruction", "")
        if not user_input:
            return JSONResponse(content={"error": "Instruction is required"}, status_code=400)

        # Generate response using the model
        def generate_response(prompt, max_length=200):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)  # device can be "cuda" or "cpu"
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=max_length)
            return tokenizer.decode(outputs[0], skip_special_tokens=True)

        response_text = generate_response(user_input)

        return JSONResponse(content={"generated_text": response_text})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        workers=1,
    )
