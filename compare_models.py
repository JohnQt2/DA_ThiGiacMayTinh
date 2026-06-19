import os
import sys
import io
import json
import numpy as np
import matplotlib.pyplot as plt

# Fix encoding trên Windows
if sys.stdout.encoding != 'utf-8':
    try:
        if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer') and sys.stderr.buffer is not None:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except (AttributeError, io.UnsupportedOperation):
        pass

def load_results(path):
    if not os.path.exists(path):
        print(f"[ERROR] Không tìm thấy file: {path}")
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    mb_results = load_results('models/mobilenetv2_results.json')
    ef_results = load_results('models/efficientnetb3_results.json')
    
    if mb_results is None or ef_results is None:
        print("[LỖI] Vui lòng huấn luyện cả 2 mô hình trước khi so sánh.")
        return

    mb_history, ef_history = mb_results['history'], ef_results['history']
    mb_time, ef_time = mb_results['training_time'], ef_results['training_time']
    mb_metrics, ef_metrics = mb_results.get('metrics', {}), ef_results.get('metrics', {})

    # ================= IN BẢNG SO SÁNH =================
    print("\n" + "="*50)
    print(" BẢNG SO SÁNH KẾT QUẢ HUẤN LUYỆN ".center(50))
    print("="*50)
    print(f"{'Tiêu chí (Metric)':<20} | {'MobileNetV2':<12} | {'EfficientNetB3':<12}")
    print("-"*50)
    print(f"{'Thời gian train':<20} | {mb_time:10.2f}s | {ef_time:10.2f}s")
    
    if mb_metrics and ef_metrics:
        if 'loss' in mb_metrics and 'loss' in ef_metrics:
            print(f"{'Val Loss':<20} | {mb_metrics.get('loss', 0):10.4f}   | {ef_metrics.get('loss', 0):10.4f}")
        print(f"{'Accuracy':<20} | {mb_metrics.get('accuracy', 0)*100:9.2f}% | {ef_metrics.get('accuracy', 0)*100:9.2f}%")
        print(f"{'Precision (macro)':<20} | {mb_metrics.get('precision', 0)*100:9.2f}% | {ef_metrics.get('precision', 0)*100:9.2f}%")
        print(f"{'Recall (macro)':<20} | {mb_metrics.get('recall', 0)*100:9.2f}% | {ef_metrics.get('recall', 0)*100:9.2f}%")
        print(f"{'F1-score (macro)':<20} | {mb_metrics.get('f1_score', 0)*100:9.2f}% | {ef_metrics.get('f1_score', 0)*100:9.2f}%")
        if 'mAP' in mb_metrics and 'mAP' in ef_metrics:
            print(f"{'mAP (macro)':<20} | {mb_metrics.get('mAP', 0)*100:9.2f}% | {ef_metrics.get('mAP', 0)*100:9.2f}%")
    print("="*50)

    # ================= VẼ BIỂU ĐỒ SO SÁNH =================
    plt.figure(figsize=(15, 8))
    
    # 1. Thời gian train
    plt.subplot(2, 2, 1)
    plt.bar(['MobileNetV2', 'EfficientNetB3'], [mb_time, ef_time], color=['#1f77b4', '#2ca02c'])
    plt.ylabel('Thời gian (giây)')
    plt.title('Tốc độ huấn luyện (Train Time)')
    for i, v in enumerate([mb_time, ef_time]):
        plt.text(i, v + 10, f"{v:.1f}s", ha='center', fontweight='bold')

    mb_epochs = range(1, len(mb_history['accuracy']) + 1)
    ef_epochs = range(1, len(ef_history['accuracy']) + 1)

    # 2. Validation Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(mb_epochs, mb_history['val_accuracy'], marker='o', label='MobileNetV2')
    plt.plot(ef_epochs, ef_history['val_accuracy'], marker='s', label='EfficientNetB3')
    plt.title('Độ chính xác (Validation Accuracy)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3. Validation Loss
    plt.subplot(2, 2, 3)
    plt.plot(mb_epochs, mb_history['val_loss'], marker='o', label='MobileNetV2')
    plt.plot(ef_epochs, ef_history['val_loss'], marker='s', label='EfficientNetB3')
    plt.title('Sai số (Validation Loss)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 4. Các chỉ số đánh giá
    if mb_metrics and ef_metrics:
        plt.subplot(2, 2, 4)
        labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        mb_vals = [mb_metrics.get('accuracy', 0)*100, mb_metrics.get('precision', 0)*100, mb_metrics.get('recall', 0)*100, mb_metrics.get('f1_score', 0)*100]
        ef_vals = [ef_metrics.get('accuracy', 0)*100, ef_metrics.get('precision', 0)*100, ef_metrics.get('recall', 0)*100, ef_metrics.get('f1_score', 0)*100]
        
        if 'mAP' in mb_metrics and 'mAP' in ef_metrics:
            labels.append('mAP')
            mb_vals.append(mb_metrics.get('mAP', 0)*100)
            ef_vals.append(ef_metrics.get('mAP', 0)*100)
            
        x = np.arange(len(labels))
        width = 0.35
        plt.bar(x - width/2, mb_vals, width, label='MobileNetV2')
        plt.bar(x + width/2, ef_vals, width, label='EfficientNetB3')
        plt.ylabel('Phần trăm (%)')
        plt.title('So sánh các chỉ số đánh giá')
        plt.xticks(x, labels)
        plt.legend()
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        for i, v in enumerate(mb_vals):
            plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=8)
        for i, v in enumerate(ef_vals):
            plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center', fontsize=8)

    plt.tight_layout()
    os.makedirs('ketqua', exist_ok=True)
    plt.savefig('ketqua/sosanh.png', dpi=300)
    print("Đã lưu biểu đồ so sánh tại: ketqua/sosanh.png")
    plt.show()

if __name__ == "__main__":
    main()
