from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows frontend to communicate with backend
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Requests

MODEL_NAME = "Saksham03/MCBC"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32
    )
    model.to("cpu")  # Move the model to CPU
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

def generate_response(instruction):
    try:
        inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_length=100)

        response_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response_text
    except Exception as e:
        return str(e)

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        instruction = data['instruction']
        response_text = generate_response(instruction)
        return jsonify({'generated_text': response_text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
