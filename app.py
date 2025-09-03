from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# 저장 경로
SAVE_DIR = "/home/aikusrv04/Doodle/FINAL/inputs"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        base64_image = data["image"]
        user_prompt = data["prompt"]
        theme = data["theme"]

        # Base64 → PIL.Image 변환
        image_data = base64.b64decode(base64_image.split(",")[1])
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # 파일 이름 자동 생성
        existing_files = [f for f in os.listdir(SAVE_DIR) if f.startswith("input_") and f.endswith(".png")]
        next_index = len(existing_files) + 1
        filename = f"input_{next_index}.png"
        filepath = os.path.join(SAVE_DIR, filename)

        # 저장
        image.save(filepath)
        print(f"✅ Image saved to: {filepath}")

        return jsonify({
            "status": "success",
            "saved_as": filename,
            "theme": theme,
            "prompt": user_prompt
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
