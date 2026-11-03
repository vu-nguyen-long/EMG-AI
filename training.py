import os
import numpy as np
import wfdb
from scipy.signal import butter, lfilter
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# PART 1: Preprocessing
# ==========================================
# Butterworth filter Function
def butter_bandpass(lowcut, highcut, fs, order=4): 
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq    
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Bandpass filter Function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#Hàm đọc cặp .hea, .bin file
def load_and_preprocess_record(record_path, window_size=256, overlap=128):
    record = wfdb.rdrecord(record_path)
    signals = record.p_signal 
    fs = record.fs #Lấy fs
    raw_signal = signals[:, 0]# Đọc tín hiệu

    filtered_signal = bandpass_filter(raw_signal, lowcut=20.0, highcut=4000.0, fs=fs, order=4)#Lọc bằng bandpass

    # Kiểm tra an toàn độ lệch chuẩn trước khi chia
    std_val = np.std(filtered_signal)
    if std_val == 0 or np.isnan(std_val):
        return np.array([]), fs

    normalized_signal = (filtered_signal - np.mean(filtered_signal)) / (np.std(filtered_signal) + 1e-8)#Dữ liệu đã chuẩn hóa

    # Kiểm tra và loại bỏ ngay các tín hiệu chứa NaN/Inf để tránh bùng nổ loss: nan
    # CÓ THỂ CẢI TIẾN: vá bằng gtri trb
    if np.isnan(normalized_signal).any() or np.isinf(normalized_signal).any():
        return np.array([]), fs

    windows = []
    step_size = window_size - overlap
    for i in range(0, len(normalized_signal) - window_size + 1, step_size):
        window = normalized_signal[i:i + window_size]
        windows.append(window)
        
    return np.array(windows), fs


# ==========================================
# PART 2: BUILD 1D-CNN MODEL
# ==========================================
def build_lightweight_1d_cnn(input_shape=(256, 1), num_classes=3):
    model = models.Sequential([
        #Input Layer
        layers.Input(shape=input_shape),

        #Layer 1: Conv1D + MaxPool
        layers.Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'),# => (256,32)
        layers.MaxPooling1D(pool_size=2),# => (128,32)
        
        #Layer 2: Conv1D + MaxPool
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),# => (128,64)
        layers.MaxPooling1D(pool_size=2),# => (64,64)

        #Layer 3: Conv1D + MaxPool
        layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'),# =>(64, 64)
        layers.MaxPooling1D(pool_size=2),# =>(32, 64)
        
        #Layer 4: Global Average Pooling 
        #Lay gtri trung binh của 32 gtri trong 1 kenh => có 64 gtri trung binh tuong ung vs 64 kenh
        #Neu dùng Flatten(), sẽ có 32 x 64 (gtri)=> Qua nhieu tham số để tính toán
        layers.GlobalAveragePooling1D(),# => (64,)

        #Output Layer
        layers.Dense(num_classes, activation='softmax')# =>(3,)
    ])
    return model

# ==========================================
# PART 4: EVALUATE
# ==========================================
def evaluate_model(model, X_test, y_test):
    print("\n Đang chạy đánh giá chi tiết trên tập kiểm thử...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    target_names = ['Healty_controls', 'Polymyositis', 'ALS']
    
    print("\n" + "="*50)
    print(" BÁO CÁO ĐÁNH GIÁ CHI TIẾT (CLASSIFICATION REPORT)")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    
    print("\n" + "="*50)
    print(" MA TRẬN NHẦM LẪN (CONFUSION MATRIX)")
    print("="*50)
    print(confusion_matrix(y_test, y_pred))

# ==========================================
# PART 3: LABEL AND TRAINING 
# ==========================================
def train_model(data_dir):
    # Gom dữ liệu theo từng lớp riêng biệt
    class_data = {0: [], 1: [], 2: []}
    
    file_list = os.listdir(data_dir)
    hea_files = [f for f in file_list if f.endswith('.hea')]
    
    print(f"Đang quét thư mục, tìm thấy {len(hea_files)} bản ghi...")
    
    for file_name in hea_files:
        record_name = os.path.splitext(file_name)[0]
        record_path = os.path.join(data_dir, record_name)
        
        try:
            # 1. Đọc file header để lấy danh sách comments
            header = wfdb.rdheader(record_path)
            
            diagnosis_str = ""
            for comment in header.comments:
                if comment.startswith("diagnosis:"):
                    diagnosis_str = comment.split(":")[1].strip().lower()
                    break
            
            # 2. Gán nhãn dựa trên giá trị diagnosis thực tế
            # healty controls -> 0
            if 'healty controls' in diagnosis_str:
                label = 0
            # Polymyositis -> 1
            elif 'polymyositis' in diagnosis_str:
                label = 1
            #  ALS -> 2
            elif 'als' in diagnosis_str:
                label = 2
            else:
                # Nếu gặp chuẩn đoán khác không nằm trong 3 nhóm trên thì bỏ qua
                continue
                
            # Nếu hợp lệ, tiến hành đọc và tiền xử lý tín hiệu
            windows, fs = load_and_preprocess_record(record_path)
            if len(windows) > 0:
                labels = np.full((windows.shape[0],), label)
                # Gom thẳng vào từ điển class_data theo từng nhãn tương ứng
                class_data[label].append((windows, labels))
                
        except Exception as e:
            print(f"Lỗi xử lý file {record_name}: {e}")
            
    # Kiểm tra xem có dữ liệu hợp lệ không
    total_valid = sum(len(class_data[l]) for l in [0, 1, 2])
    if total_valid == 0:
        print("Vẫn không có dữ liệu hợp lệ để huấn luyện. Hãy kiểm tra lại từ khóa diagnosis trong file .hea!")
        return None, None, None
    

    # Chia tỉ lệ 80/10/10 cho từng lớp xong gộp lại
    train_X_list, train_y_list = [], []
    val_X_list, val_y_list = [], []
    test_X_list, test_y_list = [], []
    
    for label in [0, 1, 2]:
        if len(class_data[label]) == 0:
            continue
            
        # Nối tất cả các mẫu của riêng lớp này lại
        X_cls = np.vstack([item[0] for item in class_data[label]])
        y_cls = np.concatenate([item[1] for item in class_data[label]])
        
        # 10% Test data
        X_temp, X_test_cls, y_temp, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.10, random_state=42)
        
        # 80% training,  10% validation <=> 11.11% của tập temp) 
        X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(X_temp, y_temp, test_size=0.1111, random_state=42)
        
        # Gom phần 80% vào kho Train chung
        train_X_list.append(X_train_cls)
        train_y_list.append(y_train_cls)
        
        # Gom phần 10% Validation vào kho Val chung
        val_X_list.append(X_val_cls)
        val_y_list.append(y_val_cls)
        
        # Gom phần 10% Test vào kho Test chung
        test_X_list.append(X_test_cls)
        test_y_list.append(y_test_cls)

    # Gộp toàn bộ dữ liệu của các lớp lại thành các tập lớn hoàn chỉnh
    X_train = np.vstack(train_X_list)
    y_train = np.concatenate(train_y_list)
    
    X_val = np.vstack(val_X_list)
    y_val = np.concatenate(val_y_list)
    
    X_test = np.vstack(test_X_list)
    y_test = np.concatenate(test_y_list)
    
    # Thêm chiều kênh cuối cùng cho phù hợp với Conv1D (N, 256, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"Kích thước tập Train: {X_train.shape[0]} mẫu")
    print(f"Kích thước tập Validation: {X_val.shape[0]} mẫu")
    print(f"Kích thước tập Test: {X_test.shape[0]} mẫu")

    # Do số sample của Healty nhiều nên cần tính toán trọng số lớp
    classes_unique = np.unique(y_train)
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=classes_unique,
        y=y_train
    )
    class_weights_dict = dict(zip(classes_unique, class_weights_array))
    print(f"Trọng số các lớp (Class Weights): {class_weights_dict}")
    
    # Khởi tạo mô hình
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)# Giảm learning rate để tránh loss = NaN
    model = build_lightweight_1d_cnn(input_shape=(256, 1), num_classes=3)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    # Huấn luyện mô hình
    print("\n Bắt đầu quá trình huấn luyện mô hình...")
    history = model.fit(
        X_train, y_train, #Training data
        validation_data=(X_val, y_val),#Validation data
        epochs=20,# Training 5 lần
        batch_size=64, # Xử lí 64 mẫu mỗi batch trước khi thực hiện 1 bước cập nhật weights
        class_weight=class_weights_dict # Truyền trọng số để phạt nặng nếu đoán sai lớp ít mẫu
    )
    
    # Lưu mô hình sau khi train xong để chuẩn bị đem lên Jetson Nano
    model.save("emg_lightweight_model.h5")
    print("\n Đã huấn luyện xong và lưu mô hình thành tệp 'emg_lightweight_model.h5'!")

    return model, X_test, y_test



# ==========================================
#               MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    data_directory = r'C:\Users\Admin\Documents\PlatformIO\Projects\EMG_AI\archive'
    
    # Bước 1: Gọi hàm huấn luyện và nhận lại kết quả
    trained_model, X_test_data, y_test_data = train_model(data_directory)
    
    # Bước 2: Gọi nối tiếp hàm đánh giá ngay sau khi train xong (nếu train thành công)
    if trained_model is not None:
        evaluate_model(trained_model, X_test_data, y_test_data)