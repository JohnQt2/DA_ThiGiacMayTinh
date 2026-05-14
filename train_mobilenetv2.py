import os
import time
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================= CẤU HÌNH =================
DATA_DIR = "./image_output"
IMG_SIZE = (224, 224) # Kích thước chuẩn của MobileNetV2
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = "mobilenetv2_model.keras"
# ============================================

def create_datasets():
    print(f"Đang load dữ liệu từ thư mục {DATA_DIR}...")
    
    # Khởi tạo dataset cho training (80%)
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Khởi tạo dataset cho validation (20%)
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    class_names = train_dataset.class_names
    print(f"Các class được tìm thấy ({len(class_names)} class): {class_names}")
    
    # Tối ưu hóa hiệu năng đọc dữ liệu
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, validation_dataset, class_names

def build_model(num_classes):
    print("\nĐang khởi tạo cấu trúc mô hình MobileNetV2 (Transfer Learning)...")
    
    # Input layer
    inputs = Input(shape=IMG_SIZE + (3,))
    
    # Hàm tiền xử lý đặc biệt của MobileNetV2 (đưa giá trị pixel từ [0,255] về [-1, 1])
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Khởi tạo base model MobileNetV2 pre-trained trên tập dữ liệu ImageNet
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,), 
        include_top=False,  # Bỏ phần phân loại cũ của ImageNet
        weights='imagenet'
    )
    
    # Đóng băng (Freeze) các lớp của base_model để giữ lại các đặc trưng đã học
    base_model.trainable = False
    
    # Đưa ảnh qua base model
    x = base_model(x, training=False)
    
    # Thêm các lớp Custom Head cho bài toán phân loại của bạn
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x) # Giảm Overfitting
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    # Tạo mô hình hoàn chỉnh
    model = Model(inputs, outputs)
    
    # Biên dịch mô hình
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy', # Dùng 'sparse' vì label đang là số nguyên (0, 1, 2...)
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Độ chính xác (Accuracy)')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Sai số (Loss)')
    
    plt.tight_layout()
    plt.show()

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Lỗi: Không tìm thấy thư mục dữ liệu {DATA_DIR}!")
        print("Vui lòng chạy file preprocess_dataset.py trước để tạo ra thư mục này.")
        return

    # 1. Load data
    train_dataset, validation_dataset, class_names = create_datasets()
    num_classes = len(class_names)
    
    # 2. Xây dựng model
    model = build_model(num_classes)
    
    # 3. Cấu hình Callbacks
    callbacks = [
        # Tự động lưu mô hình mỗi khi validation accuracy tăng lên
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        # Dừng huấn luyện sớm nếu val_loss không cải thiện sau 5 epoch để tránh Overfitting
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]
    
    # 4. Bắt đầu huấn luyện
    print(f"\nBắt đầu huấn luyện mô hình với tối đa {EPOCHS} epochs...")
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    print(f"\nHuấn luyện hoàn tất trong {training_time:.2f} giây! Model tốt nhất đã được lưu tại: {MODEL_SAVE_PATH}")
    
    # Lưu kết quả history và thời gian train ra file JSON để so sánh sau này
    results = {
        "history": history.history,
        "training_time": training_time
    }
    with open("mobilenetv2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
        
    # 5. Lưu danh sách các class vào file text để sử dụng cho file test sau này
    with open("class_names.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    print("Đã lưu danh sách các class vào file 'class_names.txt'")
    
    # 6. Vẽ biểu đồ đánh giá
    print("\nĐang mở biểu đồ đánh giá kết quả huấn luyện...")
    plot_history(history)

if __name__ == "__main__":
    main()
