# Hệ thống Nhận diện và Phân loại Phương tiện Giao thông thông minh

## Giới thiệu
Đồ án tập trung xây dựng hệ thống tự động phát hiện, phân loại và đếm số lượng phương tiện giao thông trên ảnh, video bằng cách kết hợp mô hình phát hiện đối tượng **YOLOv8** và mô hình phân loại chi tiết **EfficientNetB3** + **MobileNetV2**.

## Cấu trúc thư mục dự án
```
detection-app/
├── models/
│   ├── yolov8n.pt
│   ├── efficientnetb3_model.keras
│   └── mobilenetv2_model.keras
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── image_output/
├── test_images/
├── utils/
│   ├── annotation_utils.py
│   ├── data_loader.py
│   ├── preprocess_utils.py
│   └── train_utils.py
├── app.py
├── requirements.txt
├── README.md
└── generate_annotations.py
```

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

## Kết quả mô hình

### YOLOv8
- Phát hiện đối tượng: 5 class ([bicycle, car, motorcycle, bus, truck])
- Độ chính xác: Tùy thuộc vào model đã export

### EfficientNetB3
- Phân loại chi tiết: 4 class ([bus, car, motorbike, truck])
- Accuracy: ~96-98%

### MobileNetV2
- Phân loại chi tiết: 4 class ([bus, car, motorbike, truck])
- Accuracy: ~94-96%

## Công nghệ sử dụng
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript
- **Detection**: YOLOv8 (PyTorch)
- **Classification**: EfficientNetB3, MobileNetV2 (TensorFlow/Keras)
- **Data Augmentation**: Albumentations
