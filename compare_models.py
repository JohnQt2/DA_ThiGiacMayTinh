import os
import json
import matplotlib.pyplot as plt

def load_results(filepath):
    if not os.path.exists(filepath):
        print(f"Không tìm thấy file: {filepath}")
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    print("Đang đọc dữ liệu kết quả huấn luyện...")
    mb_results = load_results('mobilenetv2_results.json')
    ef_results = load_results('efficientnetb3_results.json')
    
    if mb_results is None or ef_results is None:
        print("\n[LỖI] Vui lòng chạy cả hai file train_mobilenetv2.py và train_efficientnetb3.py trước khi thực hiện so sánh.")
        return

    # Lấy dữ liệu
    mb_history = mb_results['history']
    ef_history = ef_results['history']
    
    mb_time = mb_results['training_time']
    ef_time = ef_results['training_time']

    # ================= IN BẢNG SO SÁNH TRÊN TERMINAL =================
    print("\n" + "="*50)
    print(" BẢNG SO SÁNH KẾT QUẢ HUẤN LUYỆN ".center(50))
    print("="*50)
    print(f"{'Tiêu chí (Metric)':<20} | {'MobileNetV2':<12} | {'EfficientNetB3':<12}")
    print("-"*50)
    print(f"{'Thời gian train':<20} | {mb_time:10.2f}s | {ef_time:10.2f}s")
    
    # Lấy Epoch cuối cùng
    mb_final_val_acc = mb_history['val_accuracy'][-1] * 100
    ef_final_val_acc = ef_history['val_accuracy'][-1] * 100
    print(f"{'Val Accuracy (cuối)':<20} | {mb_final_val_acc:9.2f}% | {ef_final_val_acc:9.2f}%")
    
    # Lấy giá trị cao nhất
    mb_best_val_acc = max(mb_history['val_accuracy']) * 100
    ef_best_val_acc = max(ef_history['val_accuracy']) * 100
    print(f"{'Val Accuracy (Best)':<20} | {mb_best_val_acc:9.2f}% | {ef_best_val_acc:9.2f}%")
    print("="*50)


    # ================= VẼ BIỂU ĐỒ SO SÁNH TRỰC QUAN =================
    plt.figure(figsize=(15, 5))
    
    # 1. Biểu đồ Tốc độ (Thời gian)
    plt.subplot(1, 3, 1)
    models = ['MobileNetV2', 'EfficientNetB3']
    times = [mb_time, ef_time]
    colors = ['#1f77b4', '#2ca02c'] # Xanh dương, xanh lá
    plt.bar(models, times, color=colors)
    plt.ylabel('Thời gian (giây)')
    plt.title('Tốc độ huấn luyện (Train Time)')
    
    # Ghi số giây lên đầu các cột bar
    for i, v in enumerate(times):
        plt.text(i, v + (max(times)*0.01), f"{v:.1f}s", ha='center', fontweight='bold')

    # Lấy danh sách số epoch thực tế (có thể khác nhau do EarlyStopping)
    mb_epochs = range(1, len(mb_history['accuracy']) + 1)
    ef_epochs = range(1, len(ef_history['accuracy']) + 1)

    # 2. Biểu đồ Độ chính xác trên tập Validation
    plt.subplot(1, 3, 2)
    plt.plot(mb_epochs, mb_history['val_accuracy'], marker='o', color='#1f77b4', label='MobileNetV2')
    plt.plot(ef_epochs, ef_history['val_accuracy'], marker='s', color='#2ca02c', label='EfficientNetB3')
    plt.title('So sánh Độ chính xác (Validation Accuracy)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. Biểu đồ Sai số trên tập Validation
    plt.subplot(1, 3, 3)
    plt.plot(mb_epochs, mb_history['val_loss'], marker='o', color='#1f77b4', label='MobileNetV2')
    plt.plot(ef_epochs, ef_history['val_loss'], marker='s', color='#2ca02c', label='EfficientNetB3')
    plt.title('So sánh Sai số (Validation Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
