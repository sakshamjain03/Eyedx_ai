import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your fine-tuned model
model_path = r"D:\Models\DeepSeek-R1-Distill-Qwen-1.5B"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with 8-bit quantization for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # Use FP16 for RTX 4060
    device_map="auto"  # Automatically assigns model to GPU if available
)

def generate_response(prompt, max_length=1000):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Move input to GPU
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "what is glaucoma ? what are its sympotoms"
response = generate_response(prompt)
print(response)