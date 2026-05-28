import os
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
from ultralytics import YOLO

# ================= CẤU HÌNH =================
# Thư mục chứa ảnh cần gán nhãn
INPUT_DIR = "./test"

# Đường dẫn đến mô hình YOLOv8 pre-trained
YOLO_MODEL_PATH = "./yolov8n.pt"

# Độ tin cậy tối thiểu để nhận diện phương tiện
CONF_THRESHOLD = 0.25

# Ánh xạ ID lớp của YOLOv8 sang tên nhãn chuẩn
# 2: car (ô tô), 3: motorcycle (xe máy), 5: bus (xe buýt), 7: truck (xe tải)
YOLO_CLASSES = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck"
}

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
# ============================================

def create_pascal_voc_xml(image_path, img_w, img_h, detections):
    """
    Tạo cấu trúc XML theo chuẩn Pascal VOC (LabelImg) cho một ảnh.
    """
    annotation = ET.Element("annotation")
    
    folder = ET.SubElement(annotation, "folder")
    folder.text = image_path.parent.name
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_path.name
    
    path = ET.SubElement(annotation, "path")
    path.text = str(image_path.resolve())
    
    source = ET.SubElement(annotation, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_w)
    height = ET.SubElement(size, "height")
    height.text = str(img_h)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    
    for det in detections:
        obj = ET.SubElement(annotation, "object")
        
        name = ET.SubElement(obj, "name")
        name.text = det["name"]
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(det["bbox"][0])
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(det["bbox"][1])
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(det["bbox"][2])
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(det["bbox"][3])
        
    return annotation

def auto_annotate():
    print("=== BẮT ĐẦU TỰ ĐỘNG GÁN NHÃN VỚI YOLOv8 ===")
    
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"[ERROR] Thư mục đầu vào không tồn tại: {INPUT_DIR}")
        return
        
    # Load model YOLOv8
    print(f"[INFO] Đang tải mô hình YOLO từ: {YOLO_MODEL_PATH}...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print("[INFO] Tải mô hình thành công!")
    except Exception as e:
        print(f"[ERROR] Không thể tải mô hình YOLO: {e}")
        return
        
    # Tìm các file ảnh trong thư mục
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(list(input_path.glob(f"*{ext}")))
    
    # Loại bỏ trùng lặp và sắp xếp
    image_files = sorted(list(set(image_files)))
    
    if not image_files:
        print(f"[WARNING] Không tìm thấy ảnh nào trong thư mục: {INPUT_DIR}")
        return
        
    print(f"[INFO] Tìm thấy {len(image_files)} ảnh cần gán nhãn.")
    
    xml_created_count = 0
    total_objects_detected = 0
    
    for img_file in image_files:
        try:
            # Đọc ảnh để lấy kích thước
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"  [WARNING] Không đọc được ảnh: {img_file.name}, bỏ qua.")
                continue
                
            h, w = img.shape[:2]
            
            # Chạy YOLOv8 detection
            results = model(img, classes=list(YOLO_CLASSES.keys()), conf=CONF_THRESHOLD, verbose=False)
            
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                    
                    class_name = YOLO_CLASSES.get(cls_id)
                    if class_name:
                        detections.append({
                            "name": class_name,
                            "bbox": [
                                max(0, int(xyxy[0])),
                                max(0, int(xyxy[1])),
                                min(w, int(xyxy[2])),
                                min(h, int(xyxy[3]))
                            ]
                        })
            
            if not detections:
                print(f"  [INFO] {img_file.name}: Không phát hiện phương tiện nào.")
                continue
                
            # Tạo XML Pascal VOC
            xml_data = create_pascal_voc_xml(img_file, w, h, detections)
            
            # Đường dẫn file XML tương ứng
            xml_path = img_file.with_suffix(".xml")
            
            # Ghi XML ra file thụt lề đẹp
            import xml.dom.minidom as minidom
            xml_str = ET.tostring(xml_data, encoding="utf-8")
            parsed_xml = minidom.parseString(xml_str)
            pretty_xml_str = parsed_xml.toprettyxml(indent="    ", encoding="utf-8")
            
            with open(xml_path, "wb") as f:
                # Loại bỏ dòng khai báo xml thừa nếu có để đồng nhất với LabelImg
                # (hoặc ghi thẳng bytes)
                f.write(pretty_xml_str)
                
            print(f"  [SUCCESS] Đã tạo nhãn XML cho {img_file.name} ({len(detections)} đối tượng)")
            xml_created_count += 1
            total_objects_detected += len(detections)
            
        except Exception as e:
            print(f"  [ERROR] Lỗi khi xử lý {img_file.name}: {e}")
            
    print()
    print("=" * 55)
    print(" HOÀN TẤT TỰ ĐỘNG GÁN NHÃN ")
    print("=" * 55)
    print(f"  Số ảnh đã gán nhãn XML: {xml_created_count}/{len(image_files)}")
    print(f"  Tổng số phương tiện phát hiện: {total_objects_detected}")
    print(f"  Thư mục lưu nhãn      : {input_path.resolve()}")
    print("=" * 55)

if __name__ == "__main__":
    auto_annotate()
