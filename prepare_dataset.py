import os
import sys
import io
import cv2
import argparse
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pathlib import Path
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

INPUT_DIR = "./test"
OUTPUT_DIR = "./image"
YOLO_MODEL_PATH = "./yolov8n.pt"

YOLO_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
CLASS_MAPPING = {
    "car": "car", "oto": "car", "o_to": "car",
    "bus": "bus", "xe_buyt": "bus",
    "truck": "truck", "xe_tai": "truck",
    "motorbike": "motorbike", "motorcycle": "motorbike", "xe_may": "motorbike"
}
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

def create_pascal_voc_xml(image_path, img_w, img_h, detections):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image_path.parent.name
    ET.SubElement(annotation, "filename").text = image_path.name
    ET.SubElement(annotation, "path").text = str(image_path.resolve())
    
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text = "3"
    
    ET.SubElement(annotation, "segmented").text = "0"
    
    for det in detections:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = det["name"]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(det["bbox"][0])
        ET.SubElement(bndbox, "ymin").text = str(det["bbox"][1])
        ET.SubElement(bndbox, "xmax").text = str(det["bbox"][2])
        ET.SubElement(bndbox, "ymax").text = str(det["bbox"][3])
        
    return annotation

def auto_annotate():
    print("\n--- BẮT ĐẦU TỰ ĐỘNG GÁN NHÃN VỚI YOLOv8 ---")
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"[LỖI] Thư mục đầu vào không tồn tại: {INPUT_DIR}")
        return

    print(f"Đang tải mô hình YOLO từ: {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    
    image_files = sorted(list({f for ext in IMAGE_EXTS for f in input_path.glob(f"*{ext}")}))
    if not image_files:
        print(f"[CẢNH BÁO] Không tìm thấy ảnh nào trong thư mục: {INPUT_DIR}")
        return
        
    print(f"Tìm thấy {len(image_files)} ảnh cần gán nhãn.")
    xml_count = 0
    
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        results = model(img, classes=list(YOLO_CLASSES.keys()), conf=0.25, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                xyxy = box.xyxy[0].tolist()
                class_name = YOLO_CLASSES.get(cls_id)
                if class_name:
                    detections.append({
                        "name": class_name,
                        "bbox": [max(0, int(xyxy[0])), max(0, int(xyxy[1])), min(w, int(xyxy[2])), min(h, int(xyxy[3]))]
                    })
        
        if not detections:
            continue
            
        xml_data = create_pascal_voc_xml(img_file, w, h, detections)
        xml_path = img_file.with_suffix(".xml")
        xml_str = ET.tostring(xml_data, encoding="utf-8")
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="    ", encoding="utf-8")
        
        with open(xml_path, "wb") as f:
            f.write(pretty_xml)
        xml_count += 1
        
    print(f"Hoàn tất! Đã tạo {xml_count} file nhãn XML Pascal VOC trong '{INPUT_DIR}'")

def find_image_file(xml_path, xml_filename_tag):
    parent = xml_path.parent
    if xml_filename_tag:
        img_path = parent / Path(xml_filename_tag).name
        if img_path.exists():
            return img_path
    for ext in IMAGE_EXTS:
        img_path = parent / f"{xml_path.stem}{ext}"
        if img_path.exists():
            return img_path
    return None

def crop_vehicles():
    print("\n--- BẮT ĐẦU CẮT ẢNH PHƯƠNG TIỆN TỪ NHÃN XML ---")
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    xml_files = sorted(list(input_path.glob("*.xml")))
    if not xml_files:
        print("[LỖI] Không tìm thấy file nhãn XML nào. Hãy chạy annotate trước.")
        return
        
    print(f"Tìm thấy {len(xml_files)} file nhãn XML.")
    output_path.mkdir(parents=True, exist_ok=True)
    
    total_cropped = 0
    stats = {}
    
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            filename_tag = root.find("filename")
            xml_filename = filename_tag.text if filename_tag is not None else None
            
            image_file = find_image_file(xml_file, xml_filename)
            if image_file is None:
                continue
                
            img = cv2.imread(str(image_file))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            for idx, obj in enumerate(root.findall("object")):
                label_raw = obj.find("name").text.strip().lower()
                class_name = CLASS_MAPPING.get(label_raw, label_raw)
                
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                    
                xmin = max(0, int(float(bndbox.find("xmin").text)))
                ymin = max(0, int(float(bndbox.find("ymin").text)))
                xmax = min(w, int(float(bndbox.find("xmax").text)))
                ymax = min(h, int(float(bndbox.find("ymax").text)))
                
                if xmax <= xmin or ymax <= ymin:
                    continue
                    
                crop_img = img[ymin:ymax, xmin:xmax]
                class_dir = output_path / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                
                out_name = f"{class_name}.{image_file.stem}_crop_{idx}.png"
                cv2.imwrite(str(class_dir / out_name), crop_img)
                
                stats[class_name] = stats.get(class_name, 0) + 1
                total_cropped += 1
        except Exception as e:
            print(f"Lỗi khi xử lý {xml_file.name}: {e}")
            
    print(f"Hoàn tất! Đã cắt tổng cộng {total_cropped} ảnh phương tiện lưu vào '{OUTPUT_DIR}'")
    for cls, count in stats.items():
        print(f"  + Lớp '{cls}': {count} ảnh")

def main():
    parser = argparse.ArgumentParser(description="Chuẩn bị dataset: Tự động gán nhãn YOLOv8 & Cắt ảnh phương tiện.")
    parser.add_argument("--annotate", action="store_true", help="Chạy YOLOv8 tự động gán nhãn XML")
    parser.add_argument("--crop", action="store_true", help="Cắt ảnh phương tiện từ các nhãn XML")
    parser.add_argument("--all", action="store_true", help="Chạy cả gán nhãn và cắt ảnh")
    
    args = parser.parse_args()
    
    # Nếu không truyền tham số nào, mặc định là --all
    if not (args.annotate or args.crop or args.all):
        args.all = True
        
    if args.all or args.annotate:
        auto_annotate()
    if args.all or args.crop:
        crop_vehicles()

if __name__ == "__main__":
    main()
