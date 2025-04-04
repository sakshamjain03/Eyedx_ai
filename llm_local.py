import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your fine-tuned model
model_path = r"C:\My Data\New folder (2)\Eyedx_ai\LLM_model\Saksham-Med-Llama-8b"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with 8-bit quantization for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use FP16 for RTX 4060
    device_map="auto"  # Automatically assigns model to GPU if available
)

def generate_response(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move input to GPU
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "What are the symptoms of diabetic retinopathy?"
response = generate_response(prompt)
print(response)
