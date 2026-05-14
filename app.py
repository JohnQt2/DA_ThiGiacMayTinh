import os
import io
import sys
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template

# Fix encoding tren Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from PIL import Image
import tensorflow as tf

# ================= CẤU HÌNH =================
MODEL_PATH = "./efficientnetb3_model.keras"
CLASS_NAMES_PATH = "./class_names.txt"
IMG_SIZE = (224, 224)
# ============================================

app = Flask(__name__)

# Load model & class names một lần khi khởi động
print("[INFO] Loading EfficientNetB3 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Model loaded successfully!")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

CLASS_ICONS = {
    "bus": "🚌",
    "car": "🚗",
    "motorbike": "🏍️",
    "truck": "🚛",
}

CLASS_VI = {
    "bus": "Xe Buýt",
    "car": "Ô Tô",
    "motorbike": "Xe Máy",
    "truck": "Xe Tải",
}

def preprocess_image(image_bytes):
    """Tiền xử lý ảnh cho EfficientNetB3 (không cần rescale, model tự xử lý)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)  # giữ [0, 255]
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

@app.route("/")
def index():
    return render_template("index.html", class_names=class_names)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Không có file được gửi lên"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Tên file trống"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png"]:
        return jsonify({"error": "Chỉ chấp nhận file JPG hoặc PNG"}), 400

    try:
        image_bytes = file.read()

        # Tạo base64 để gửi lại frontend hiển thị preview
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        mime = "image/png" if ext == ".png" else "image/jpeg"

        # Tiền xử lý & dự đoán
        img_array = preprocess_image(image_bytes)
        predictions = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)

        # Lấy top-k kết quả
        top_indices = np.argsort(predictions)[::-1]
        results = []
        for idx in top_indices:
            name = class_names[idx]
            results.append({
                "class": name,
                "class_vi": CLASS_VI.get(name, name),
                "icon": CLASS_ICONS.get(name, "🚗"),
                "confidence": float(predictions[idx]) * 100,
            })

        return jsonify({
            "success": True,
            "image": f"data:{mime};base64,{img_b64}",
            "filename": file.filename,
            "prediction": results[0],
            "all_predictions": results,
        })

    except Exception as e:
        return jsonify({"error": f"Lỗi khi xử lý ảnh: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
