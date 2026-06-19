import os
import sys
import io
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
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

class ImagePreprocessor:
    def __init__(self, input_dir="./image", output_dir="./image_output", target_size=(224, 224)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.classes = [d.name for d in self.input_dir.iterdir() if d.is_dir()] if self.input_dir.exists() else []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for cls in self.classes:
            (self.output_dir / cls).mkdir(exist_ok=True)

    def is_valid_image(self, file_path):
        try:
            img = cv2.imread(str(file_path))
            return img is not None
        except:
            return False

    def clean_and_resize(self):
        print("Đang làm sạch và resize dữ liệu về 224x224...")
        for cls in self.classes:
            out_class_dir = self.output_dir / cls
            for img_path in (self.input_dir / cls).glob("*.*"):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                if self.is_valid_image(img_path):
                    img = cv2.imread(str(img_path))
                    img_resized = cv2.resize(img, self.target_size)
                    cv2.imwrite(str(out_class_dir / img_path.name), img_resized)

    def augment_image(self, image):
        # Lật ngang ngẫu nhiên
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        # Xoay ngẫu nhiên (-20 đến 20 độ)
        if random.random() > 0.5:
            angle = random.randint(-20, 20)
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        # Thay đổi độ sáng ngẫu nhiên
        if random.random() > 0.5:
            value = random.randint(-30, 30)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            if value >= 0:
                v[v > (255 - value)] = 255
                v[v <= (255 - value)] += value
            else:
                v[v < -value] = 0
                v[v >= -value] -= -value
            image = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        return image

    def balance_data(self):
        print("Đang cân bằng dữ liệu giữa các class (Oversampling + Augmentation)...")
        class_counts = {cls: len(list((self.output_dir / cls).glob("*.*"))) for cls in self.classes}
        if not class_counts or sum(class_counts.values()) == 0:
            return
            
        max_count = max(class_counts.values())
        print(f"Số lượng ảnh mục tiêu: {max_count}")
        
        for cls, count in class_counts.items():
            if count == 0:
                continue
            if count < max_count:
                needed = max_count - count
                existing = list((self.output_dir / cls).glob("*.*"))
                print(f"  -> Sinh thêm {needed} ảnh cho class '{cls}'...")
                for i in range(needed):
                    src_path = random.choice(existing)
                    img = cv2.imread(str(src_path))
                    aug_img = self.augment_image(img)
                    cv2.imwrite(str(self.output_dir / cls / f"aug_{i}_{src_path.name}"), aug_img)

    def visualize_samples(self, num_samples=5):
        print("Đang vẽ biểu đồ trực quan hóa dữ liệu...")
        num_classes = len(self.classes)
        if num_classes == 0:
            return
            
        fig, axes = plt.subplots(num_classes, num_samples, figsize=(num_samples * 2.5, num_classes * 2.5))
        if num_classes == 1:
            axes = [axes]
            
        for i, cls in enumerate(self.classes):
            images = list((self.output_dir / cls).glob("*.*"))
            selected = random.sample(images, min(num_samples, len(images)))
            for j in range(num_samples):
                ax = axes[i][j] if num_classes > 1 else axes[j]
                if j < len(selected):
                    img = cv2.imread(str(selected[j]))
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    ax.set_title(f"{cls} ({j+1})", fontsize=9)
                ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("=== BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU ===")
    preprocessor = ImagePreprocessor()
    preprocessor.clean_and_resize()
    preprocessor.balance_data()
    preprocessor.visualize_samples()
    print("=== TIỀN XỬ LÝ HOÀN TẤT ===")
