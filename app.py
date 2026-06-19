import os
import io
import sys
import base64
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO

# Fix encoding trên Windows
if sys.stdout.encoding != 'utf-8':
    try:
        if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and sys.stderr.buffer is not None:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        pass

MODEL_PATH = "models/efficientnetb3_model.keras"
CLASS_NAMES_PATH = "./class_names.txt"
IMG_SIZE = (224, 224)

app = Flask(__name__)

# Load models
print("[INFO] Đang nạp mô hình...")
model = tf.keras.models.load_model(MODEL_PATH)
yolo_model = YOLO("yolov8n.pt")
print("[INFO] Đã nạp thành công các mô hình!")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

CLASS_ICONS = {"bus": "🚌", "car": "🚗", "motorbike": "🏍️", "truck": "🚛"}
CLASS_VI = {"bus": "Xe Buýt", "car": "Ô Tô", "motorbike": "Xe Máy", "truck": "Xe Tải"}
YOLO_CLASSES = {
    1: {"name": "bicycle", "name_vi": "Xe Đạp", "icon": "🚲", "color": (255, 255, 0)},
    2: {"name": "car", "name_vi": "Ô Tô", "icon": "🚗", "color": (0, 255, 0)},
    3: {"name": "motorcycle", "name_vi": "Xe Máy", "icon": "🏍️", "color": (255, 0, 0)},
    5: {"name": "bus", "name_vi": "Xe Buýt", "icon": "🚌", "color": (0, 255, 0)},
    7: {"name": "truck", "name_vi": "Xe Tải", "icon": "🚛", "color": (0, 122, 255)},
}

def annotate_image(image_bytes, detections):
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
        
        # Vẽ bounding box
        thickness = max(2, int(min(img.size) * 0.006))
        for i in range(thickness):
            draw.rectangle([x1 + i, y1 + i, x2 - i, y2 - i], outline=color)
            
        # Vẽ nền nhãn và nhãn chữ
        try:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
        except AttributeError:
            text_w, text_h = draw.textsize(label, font=font)
            
        draw.rectangle([x1, y1 - text_h - 6, x1 + text_w + 6, y1], fill=color)
        lum = color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114
        text_color = (0, 0, 0) if lum > 186 else (255, 255, 255)
        draw.text((x1 + 3, y1 - text_h - 4), label, fill=text_color, font=font)
        
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()

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
        
    mode = request.form.get("mode", "single")
    try:
        image_bytes = file.read()
        if mode == "multiple":
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            results = yolo_model(pil_img, classes=[1, 2, 3, 5, 7], conf=0.25, verbose=False)
            
            detections, counts = [], {}
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    
                    meta = YOLO_CLASSES.get(cls_id)
                    if meta:
                        detections.append({
                            "box": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                            "class": meta["name"],
                            "class_vi": meta["name_vi"],
                            "icon": meta["icon"],
                            "confidence": conf * 100,
                            "color": meta["color"]
                        })
                        counts[meta["name_vi"]] = counts.get(meta["name_vi"], 0) + 1
            
            img_b64 = base64.b64encode(annotate_image(image_bytes, detections) if detections else image_bytes).decode("utf-8")
            summary = []
            for name_vi, count in counts.items():
                icon, name_en = next(((m["icon"], m["name"]) for m in YOLO_CLASSES.values() if m["name_vi"] == name_vi), ("🚗", "vehicle"))
                summary.append({"class_vi": name_vi, "class": name_en, "icon": icon, "count": count})
                
            return jsonify({
                "success": True, "mode": "multiple", "image": f"data:image/jpeg;base64,{img_b64}",
                "filename": file.filename, "detections": detections, "summary": summary, "total_count": len(detections)
            })
        else:
            # Single mode: EfficientNetB3
            img_b64 = base64.b64encode(image_bytes).decode("utf-8")
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
            img_array = np.expand_dims(np.array(pil_img, dtype=np.float32), axis=0)
            
            predictions = model.predict(img_array, verbose=0)[0]
            top_indices = np.argsort(predictions)[::-1]
            results = [{
                "class": class_names[idx],
                "class_vi": CLASS_VI.get(class_names[idx], class_names[idx]),
                "icon": CLASS_ICONS.get(class_names[idx], "🚗"),
                "confidence": float(predictions[idx]) * 100
            } for idx in top_indices]
            
            return jsonify({
                "success": True, "mode": "single", "image": f"data:image/jpeg;base64,{img_b64}",
                "filename": file.filename, "prediction": results[0], "all_predictions": results
            })
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý ảnh: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)

