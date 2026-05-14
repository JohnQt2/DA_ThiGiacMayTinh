import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path

class ImagePreprocessor:
    def __init__(self, input_dir, output_dir, target_size=(224, 224)):
        """
        Khởi tạo bộ tiền xử lý ảnh.
        :param input_dir: Thư mục chứa dữ liệu gốc (chia theo thư mục con là các class).
        :param output_dir: Thư mục lưu dữ liệu sau khi xử lý.
        :param target_size: Kích thước ảnh đầu ra mong muốn.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        
        # Tự động lấy danh sách các class dựa trên tên thư mục con
        if self.input_dir.exists():
            self.classes = [d.name for d in self.input_dir.iterdir() if d.is_dir()]
        else:
            self.classes = []
            print(f"Cảnh báo: Thư mục đầu vào {input_dir} không tồn tại!")
        
        # Tạo cấu trúc thư mục đầu ra
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
        for cls in self.classes:
            (self.output_dir / cls).mkdir(exist_ok=True)

    def is_valid_image(self, file_path):
        """1. LÀM SẠCH: Kiểm tra xem file có phải là ảnh hợp lệ và đọc được không"""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return False
            return True
        except Exception:
            return False

    def clean_and_resize(self):
        """1. LÀM SẠCH & CHUẨN HÓA: Lọc ảnh lỗi, định dạng sai, và resize về kích thước chuẩn"""
        print("Đang làm sạch và resize dữ liệu...")
        for cls in self.classes:
            class_dir = self.input_dir / cls
            out_class_dir = self.output_dir / cls
            
            for img_path in class_dir.glob("*.*"):
                # Chỉ xử lý các định dạng ảnh phổ biến
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                    
                if self.is_valid_image(img_path):
                    # Đọc và resize ảnh
                    img = cv2.imread(str(img_path))
                    img_resized = cv2.resize(img, self.target_size)
                    # Lưu vào thư mục đích
                    cv2.imwrite(str(out_class_dir / img_path.name), img_resized)
                else:
                    print(f"  -> Ảnh lỗi hoặc bị hỏng, đã bỏ qua: {img_path}")

    def augment_image(self, image):
        """3. AUGMENTATION: Thực hiện Data Augmentation ngẫu nhiên trên một ảnh"""
        # Xác suất 50% lật ngang (Horizontal Flip)
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            
        # Xác suất 50% xoay ngẫu nhiên (Random Rotation)
        if random.random() > 0.5:
            angle = random.randint(-20, 20)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            # Dùng borderMode=cv2.BORDER_REPLICATE để tránh viền đen khi xoay
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
        # Xác suất 50% thay đổi độ sáng (Brightness Adjustment)
        if random.random() > 0.5:
            value = random.randint(-30, 30)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            if value >= 0:
                lim = 255 - value
                v[v > lim] = 255
                v[v <= lim] += value
            else:
                lim = -value
                v[v < lim] = 0
                v[v >= lim] -= -value
                
            final_hsv = cv2.merge((h, s, v))
            image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
            
        return image

    def balance_data(self):
        """2. CÂN BẰNG: Tự động cân bằng dữ liệu giữa các class bằng Oversampling kết hợp Augmentation"""
        print("\nĐang cân bằng dữ liệu...")
        class_counts = {}
        # Đếm số lượng ảnh hiện tại trong thư mục đầu ra
        for cls in self.classes:
            class_dir = self.output_dir / cls
            class_counts[cls] = len(list(class_dir.glob("*.*")))
            
        if not class_counts or sum(class_counts.values()) == 0:
            print("Không có dữ liệu trong thư mục đầu ra để cân bằng. Vui lòng chạy clean_and_resize() trước.")
            return

        # Tìm class có số lượng mẫu nhiều nhất
        max_count = max(class_counts.values())
        print(f"Số lượng ảnh mục tiêu (class lớn nhất): {max_count}")
        
        for cls, count in class_counts.items():
            if count == 0:
                continue
            if count < max_count:
                needed = max_count - count
                print(f"  -> Class '{cls}' đang có {count} ảnh. Cần sinh thêm {needed} ảnh.")
                
                class_dir = self.output_dir / cls
                existing_images = list(class_dir.glob("*.*"))
                
                for i in range(needed):
                    # Chọn ngẫu nhiên một ảnh từ các ảnh hiện có để augment
                    src_img_path = random.choice(existing_images)
                    img = cv2.imread(str(src_img_path))
                    
                    # Tạo ảnh mới qua augmentation
                    aug_img = self.augment_image(img)
                    
                    # Lưu ảnh sinh thêm
                    new_img_name = f"aug_{i}_{src_img_path.name}"
                    cv2.imwrite(str(class_dir / new_img_name), aug_img)

    def visualize_samples(self, num_samples=5):
        """4. TRỰC QUAN HÓA: Hiển thị một số mẫu dữ liệu từ các class sau khi xử lý"""
        print("\nĐang trực quan hóa mẫu dữ liệu...")
        num_classes = len(self.classes)
        if num_classes == 0:
            return

        fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples * 3, num_classes * 3))
        
        # Xử lý trường hợp chỉ có 1 class để tránh lỗi index mảng
        if num_classes == 1:
            axes = [axes]
            
        for i, cls in enumerate(self.classes):
            class_dir = self.output_dir / cls
            images = list(class_dir.glob("*.*"))
            
            # Chọn ngẫu nhiên ảnh để hiển thị
            selected_images = random.sample(images, min(num_samples, len(images)))
            
            for j in range(num_samples):
                ax = axes[i][j] if num_classes > 1 else axes[j]
                if j < len(selected_images):
                    img_path = selected_images[j]
                    img = cv2.imread(str(img_path))
                    # Chuyển hệ màu từ BGR (OpenCV) sang RGB (Matplotlib) để hiển thị đúng màu
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                    ax.imshow(img)
                    ax.set_title(f"Class: {cls} \n{img_path.name[:15]}...", fontsize=9)
                ax.axis('off')
                
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    INPUT_DIRECTORY = "./image" 
    OUTPUT_DIRECTORY = "./image_output"
    
    print("=== BẮT ĐẦU QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU ẢNH ===")
    preprocessor = ImagePreprocessor(input_dir=INPUT_DIRECTORY, output_dir=OUTPUT_DIRECTORY)
    
    # Bước 1: Làm sạch (loại bỏ ảnh hỏng) và resize về 224x224
    preprocessor.clean_and_resize()
    
    # Bước 2 & 3: Cân bằng dữ liệu sử dụng Data Augmentation cho các class thiểu số
    preprocessor.balance_data()
    
    # Bước 4: Trực quan hóa dữ liệu sau khi xử lý (Lấy random 5 ảnh mỗi class)
    preprocessor.visualize_samples(num_samples=5)
    
    print("\nQuy trình hoàn tất.")
