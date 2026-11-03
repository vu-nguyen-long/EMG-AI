import numpy as np
import matplotlib.pyplot as plt

# 1. Đọc dữ liệu từ file .bin (chọn dtype phù hợp như int16, int32 hoặc float32 tùy định dạng gốc)
file_path = r'C:\Users\Admin\Downloads\archive\N2001M07BB02.bin'
data = np.fromfile(file_path, dtype=np.int16)

# 2. Áp dụng hệ số khuếch đại (gain) hoặc chuyển đổi đơn vị nếu cần thiết
gain = 500
signal_data = data / gain

# 3. Vẽ đồ thị tín hiệu theo thời gian hoặc số mẫu
plt.figure(figsize=(12, 4))
plt.plot(signal_data, color='b', linewidth=0.6)
plt.title('Đồ thị tín hiệu')
plt.xlabel('Số mẫu (Samples)')
plt.ylabel('Biên độ')
plt.grid(True)
plt.show()