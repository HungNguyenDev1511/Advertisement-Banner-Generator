from flask import Flask, request, jsonify, send_file
import torch
from torchvision import transforms
from PIL import Image
import io
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import load_file

app = Flask(__name__)

# ==== 1. LOAD MODELS ====
# Đường dẫn tới model pretrained
TEXT_ENCODER_PATH = "text_encoder"
VQVAE_MODEL_PATH = "vqvae/diffusion_pytorch_model.fp16.safetensors"

# Load tokenizer và text encoder (ví dụ dùng CLIP)
tokenizer = CLIPTokenizer.from_pretrained(TEXT_ENCODER_PATH)
text_encoder = CLIPTextModel.from_pretrained(TEXT_ENCODER_PATH)
text_encoder.eval()

# Giả lập mô hình VQVAE
class VQVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        return self.decoder(z)

vqvae = VQVAE()
vqvae.load_state_dict(load_file(VQVAE_MODEL_PATH))
vqvae.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
text_encoder = text_encoder.to(device)
vqvae = vqvae.to(device)

# ==== 2. PREPROCESS FUNCTION ====
def generate_image_from_prompt(prompt):
    # Encode prompt thành text embedding
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        text_embedding = text_encoder(**inputs).last_hidden_state  # Lấy text embedding

        # Giả lập latent vector từ text embedding
        # Trong thực tế, latent sẽ được sinh từ diffusion hoặc một model phức tạp hơn
        latent_vector = torch.randn(1, 256, 16, 16).to(device)  # Random latent vector

        # Decode latent vector thành hình ảnh
        output_image = vqvae(latent_vector)
        output_image = output_image.squeeze(0).clamp(0, 1)  # Loại bỏ batch dimension

        # Convert tensor sang PIL Image
        output_pil = transforms.ToPILImage()(output_image.cpu())
        return output_pil

# ==== 3. FLASK ROUTES ====
@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Nhận prompt từ request
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Generate image từ prompt
        image = generate_image_from_prompt(prompt)

        # Lưu hình ảnh vào buffer
        img_io = io.BytesIO()
        image.save(img_io, "JPEG")
        img_io.seek(0)

        return send_file(img_io, mimetype="image/jpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Text-to-Image API is running!"})

# ==== 4. RUN APP ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
