# Hệ thống Nhận diện và Phân loại Phương tiện Giao thông thông minh

## Giới thiệu
Đồ án tập trung xây dựng hệ thống tự động phát hiện, phân loại và đếm số lượng phương tiện giao thông trên ảnh, video bằng cách kết hợp mô hình phát hiện đối tượng **YOLOv8** và mô hình phân loại chi tiết **EfficientNetB3** + **MobileNetV2**.

## Cài đặt môi trường
1. Tạo môi trường ảo:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Cài đặt thư viện:
   ```bash
   pip install -r requirements.txt
   ```

## Sử dụng
1. Huấn luyện mô hình (nếu cần):
   ```bash
   python train_efficientnetb3.py
   python train_mobilenetv2.py
   ```

2. Chạy ứng dụng:
   ```bash
   python app.py
   ```
   Truy cập: http://[IP_ADDRESS]

## Công nghệ sử dụng
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Detection**: YOLOv8 (PyTorch)
- **Classification**: EfficientNetB3, MobileNetV2 (TensorFlow/Keras)
- **Data Augmentation**: Albumentations
