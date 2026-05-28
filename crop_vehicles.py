import os
import sys
import io
import cv2
import xml.etree.ElementTree as ET
from pathlib import Path

# Fix encoding trên Windows
if sys.stdout.encoding != 'utf-8':
    try:
        if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and sys.stderr.buffer is not None:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        pass

# ================= CẤU HÌNH =================
# Thư mục chứa ảnh gốc + file XML nhãn từ LabelImg (cùng thư mục)
INPUT_DIR = "./test"

# Thư mục đầu ra để lưu ảnh phương tiện đã cắt (phân theo lớp)
OUTPUT_DIR = "./image1"

# Ánh xạ nhãn từ LabelImg sang tên thư mục phân lớp đích
# (Nếu bạn đặt nhãn trong LabelImg đúng tên lớp thì không cần chỉnh)
CLASS_MAPPING = {
    "car": "car",
    "oto": "car",
    "o_to": "car",
    "bus": "bus",
    "xe_buyt": "bus",
    "truck": "truck",
    "xe_tai": "truck",
    "motorbike": "motorbike",
    "motorcycle": "motorbike",
    "xe_may": "motorbike",
}

# Các định dạng ảnh được hỗ trợ
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
# ============================================


def find_image_file(xml_path, xml_filename_tag):
    """Tìm file ảnh tương ứng với file XML (cùng thư mục)."""
    parent_dir = xml_path.parent
    xml_stem = xml_path.stem

    # Cách 1: Theo thẻ <filename> trong XML
    if xml_filename_tag:
        img_name = Path(xml_filename_tag).name
        img_path = parent_dir / img_name
        if img_path.exists():
            return img_path

    # Cách 2: Tìm ảnh cùng tên với file XML nhưng khác đuôi
    for ext in IMAGE_EXTENSIONS:
        img_path = parent_dir / f"{xml_stem}{ext}"
        if img_path.exists():
            return img_path

    return None


def crop_from_labelimg_xml(input_dir, output_dir, class_mapping):
    """
    Đọc file XML nhãn từ LabelImg (Pascal VOC), cắt ảnh phương tiện
    theo bounding box và lưu vào thư mục phân lớp tương ứng.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"[ERROR] Thư mục đầu vào không tồn tại: {input_dir}")
        return

    # Tìm file XML
    xml_files = sorted(input_path.glob("*.xml"))
    if not xml_files:
        print(f"[ERROR] Không tìm thấy file nhãn XML nào trong: {input_dir}")
        print()
        print("=" * 58)
        print(" HƯỚNG DẪN: TẠO NHÃN BẰNG LABELIMG TRƯỚC KHI CHẠY SCRIPT")
        print("=" * 58)
        print()
        print("  Bước 1: Mở LabelImg")
        print(f"          > labelimg {input_path.resolve()}")
        print()
        print("  Bước 2: Trong LabelImg:")
        print("          - Chọn 'PascalVOC' ở góc trái (định dạng lưu)")
        print("          - Nhấn 'W' để vẽ bounding box quanh phương tiện")
        print("          - Nhập nhãn: car, bus, motorbike, truck")
        print("          - Nhấn Ctrl+S để lưu file XML")
        print("          - Nhấn 'D' để chuyển sang ảnh tiếp theo")
        print()
        print("  Bước 3: Sau khi đánh nhãn xong, chạy lại script này:")
        print("          > python crop_vehicles.py")
        print("=" * 58)
        return

    print(f"[INFO] Tìm thấy {len(xml_files)} file nhãn XML.")
    print(f"[INFO] Bắt đầu cắt ảnh phương tiện...\n")

    output_path.mkdir(parents=True, exist_ok=True)

    stats = {}
    total_cropped = 0
    processed = 0

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Lấy tên file ảnh từ thẻ <filename>
            filename_tag = root.find("filename")
            xml_filename = filename_tag.text if filename_tag is not None else None

            # Tìm file ảnh thực tế
            image_file = find_image_file(xml_file, xml_filename)
            if image_file is None:
                print(f"  [WARNING] Không tìm thấy ảnh cho: {xml_file.name}, bỏ qua.")
                continue

            # Đọc ảnh gốc
            img = cv2.imread(str(image_file))
            if img is None:
                print(f"  [WARNING] Không đọc được ảnh: {image_file.name}, bỏ qua.")
                continue

            h, w = img.shape[:2]
            cropped_count = 0

            # Duyệt các object đã đánh nhãn trong XML
            for idx, obj in enumerate(root.findall("object")):
                label_raw = obj.find("name").text.strip().lower()

                # Ánh xạ nhãn sang tên lớp chuẩn
                class_name = class_mapping.get(label_raw, label_raw)

                # Đọc tọa độ bounding box
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue

                xmin = max(0, int(float(bndbox.find("xmin").text)))
                ymin = max(0, int(float(bndbox.find("ymin").text)))
                xmax = min(w, int(float(bndbox.find("xmax").text)))
                ymax = min(h, int(float(bndbox.find("ymax").text)))

                if xmax <= xmin or ymax <= ymin:
                    print(f"  [WARNING] Box không hợp lệ trong {xml_file.name}: "
                          f"[{xmin},{ymin},{xmax},{ymax}], bỏ qua.")
                    continue

                # Cắt ảnh phương tiện
                crop_img = img[ymin:ymax, xmin:xmax]

                # Tạo thư mục lớp đích
                class_dir = output_path / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                # Lưu ảnh đã cắt
                out_name = f"{class_name}.{image_file.stem}_crop_{idx}.png"
                cv2.imwrite(str(class_dir / out_name), crop_img)

                stats[class_name] = stats.get(class_name, 0) + 1
                cropped_count += 1
                total_cropped += 1

            processed += 1
            if cropped_count > 0:
                print(f"  {image_file.name}: cắt được {cropped_count} phương tiện")

        except Exception as e:
            print(f"  [ERROR] Lỗi xử lý {xml_file.name}: {e}")

    # Báo cáo kết quả
    print()
    print("=" * 55)
    print(" HOÀN TẤT CẮT ẢNH PHƯƠNG TIỆN TỪ NHÃN LABELIMG")
    print("=" * 55)
    print(f"  File XML đã xử lý : {processed}/{len(xml_files)}")
    print(f"  Tổng ảnh đã cắt   : {total_cropped}")
    print(f"  Thư mục đầu ra    : {output_path.resolve()}")
    print()
    if stats:
        print("  Thống kê theo lớp:")
        for cls, count in sorted(stats.items()):
            print(f"    + {cls:12s}: {count} ảnh")
    print("=" * 55)


if __name__ == "__main__":
    crop_from_labelimg_xml(INPUT_DIR, OUTPUT_DIR, CLASS_MAPPING)
