import os
import io
import sys
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template

# Fix encoding tren Windows
if sys.stdout.encoding != 'utf-8':
    try:
        if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and sys.stderr.buffer is not None:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        pass
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from ultralytics import YOLO

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

print("[INFO] Loading YOLOv8 model...")
yolo_model = YOLO("yolov8n.pt")
print("[INFO] YOLOv8 model loaded successfully!")

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

YOLO_CLASSES = {
    1: {"name": "bicycle", "name_vi": "Xe Đạp", "icon": "🚲", "color": (255, 255, 0)},       # Yellow
    2: {"name": "car", "name_vi": "Ô Tô", "icon": "🚗", "color": (0, 255, 0)},              # Green
    3: {"name": "motorcycle", "name_vi": "Xe Máy", "icon": "🏍️", "color": (255, 0, 0)},       # Red
    5: {"name": "bus", "name_vi": "Xe Buýt", "icon": "🚌", "color": (0, 255, 0)},           # Green
    7: {"name": "truck", "name_vi": "Xe Tải", "icon": "🚛", "color": (0, 255, 0)},          # Green
}

def annotate_image(image_bytes, detections):
    """
    Vẽ bounding box và nhãn lên ảnh sử dụng PIL.
    Detections: list of dicts {'box': [x1, y1, x2, y2], 'class_vi': ..., 'confidence': ..., 'color': (R, G, B)}
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    try:
        font_size = max(14, int(min(img.size) * 0.035))
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()
        
    for det in detections:
        x1, y1, x2, y2 = det['box']
        color = det['color']
        label = f"{det['class_vi']} {det['confidence']:.0f}%"
        
        # Độ dày của viền hộp
        thickness = max(2, int(min(img.size) * 0.006))
        for i in range(thickness):
            draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=color)
            
        # Tính kích thước text box vẽ nền chữ
        try:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)
            
        # Vẽ nền nhãn
        draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=color)
        
        # Chữ đen trên nền vàng, chữ trắng trên nền đỏ/xanh
        luminance = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        text_color = (0, 0, 0) if luminance > 186 else (255, 255, 255)
        
        draw.text((x1 + 3, y1 - text_h - 4), label, fill=text_color, font=font)
        
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=90)
    return output.getvalue()

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

    mode = request.form.get("mode", "single")

    try:
        image_bytes = file.read()

        if mode == "multiple":
            # Chuyển đổi thành PIL Image để YOLO chạy
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Chạy mô hình YOLOv8, chỉ phát hiện các lớp: bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
            results = yolo_model(pil_img, classes=[1, 2, 3, 5, 7], conf=0.25, verbose=False)
            
            detections = []
            counts = {}
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    meta = YOLO_CLASSES.get(cls_id)
                    if meta:
                        name = meta["name"]
                        name_vi = meta["name_vi"]
                        icon = meta["icon"]
                        color = meta["color"]
                        
                        detections.append({
                            "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            "class": name,
                            "class_vi": name_vi,
                            "icon": icon,
                            "confidence": conf * 100,
                            "color": color
                        })
                        
                        counts[name_vi] = counts.get(name_vi, 0) + 1
            
            # Vẽ bounding boxes lên ảnh nếu có phát hiện
            if len(detections) > 0:
                annotated_bytes = annotate_image(image_bytes, detections)
                img_b64 = base64.b64encode(annotated_bytes).decode("utf-8")
            else:
                img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Tạo danh sách đếm số lượng phương tiện
            summary = []
            for name_vi, count in counts.items():
                # Tìm icon và class tiếng Anh tương ứng
                icon = "🚗"
                name_en = "vehicle"
                for meta in YOLO_CLASSES.values():
                    if meta["name_vi"] == name_vi:
                        icon = meta["icon"]
                        name_en = meta["name"]
                        break
                summary.append({
                    "class_vi": name_vi,
                    "class": name_en,
                    "icon": icon,
                    "count": count
                })
            
            return jsonify({
                "success": True,
                "mode": "multiple",
                "image": f"data:image/jpeg;base64,{img_b64}",
                "filename": file.filename,
                "detections": detections,
                "summary": summary,
                "total_count": len(detections)
            })
            
        else:
            # Chế độ nhận diện đơn (EfficientNetB3)
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            mime = "image/png" if ext == ".png" else "image/jpeg"

            img_array = preprocess_image(image_bytes)
            predictions = model.predict(img_array, verbose=0)[0]  # shape: (num_classes,)

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
                "mode": "single",
                "image": f"data:{mime};base64,{img_b64}",
                "filename": file.filename,
                "prediction": results[0],
                "all_predictions": results,
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Lỗi khi xử lý ảnh: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
