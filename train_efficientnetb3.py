import os
import time
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ================= CẤU HÌNH =================
DATA_DIR = "./image_output"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_SAVE_PATH = "efficientnetb3_model.keras"
# ============================================

def create_datasets():
    print(f"Đang load dữ liệu từ thư mục {DATA_DIR}...")
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
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
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    
    return train_dataset, validation_dataset, class_names

def build_model(num_classes):
    print("\nĐang khởi tạo cấu trúc mô hình EfficientNetB3 (Transfer Learning)...")
    
    inputs = Input(shape=IMG_SIZE + (3,))
    
    # Keras EfficientNet đã tích hợp sẵn lớp chuẩn hóa (Normalization) bên trong model
    # nên KHÔNG CẦN gọi hàm preprocess_input như MobileNetV2, truyền thẳng inputs vào.
    
    base_model = EfficientNetB3(
        input_shape=IMG_SIZE + (3,), 
        include_top=False, 
        weights='imagenet'
    )
    
    base_model.trainable = False
    
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x) # EfficientNet có xu hướng overfit hơn nên dùng dropout cao hơn một chút
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
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
        return

    train_dataset, validation_dataset, class_names = create_datasets()
    num_classes = len(class_names)
    
    model = build_model(num_classes)
    
    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    ]
    
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
    
    results = {
        "history": history.history,
        "training_time": training_time
    }
    with open("efficientnetb3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    
    print("\nĐang mở biểu đồ đánh giá kết quả huấn luyện...")
    plot_history(history)

if __name__ == "__main__":
    main()
