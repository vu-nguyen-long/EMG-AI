import os
import wfdb

def inspect_header_files(data_dir, max_show=10):
    """
    Hàm đọc và in ra thông tin từ các file .hea trong thư mục
    """
    if not os.path.exists(data_dir):
        print(f"Thư mục '{data_dir}' không tồn tại. Vui lòng kiểm tra lại đường dẫn!")
        return

    file_list = os.listdir(data_dir)
    hea_files = [f for f in file_list if f.endswith('.hea')]
    
    print(f"Tổng số file .hea tìm thấy trong thư mục: {len(hea_files)}\n")
    print("=" * 50)

    #Khai báo các label bệnh
    als = 0
    mainly_bulbar_als = 0
    healty_controls = 0
    plexus_scapulohumoral_dystrophy = 0
    polymyositis = 0
    myopathy_unclassified = 0
    
    # Duyệt qua một số file đầu tiên để kiểm tra (mặc định tối đa 10 file)
    for i, file_name in enumerate(hea_files[:max_show]):
        record_name = os.path.splitext(file_name)[0]
        record_path = os.path.join(data_dir, record_name)
        
        try:
            # Sử dụng hàm rdheader của wfdb chỉ để đọc thông tin header
            header = wfdb.rdheader(record_path)
            
            print(f"[{i+1}] Tên bản ghi (Record Name): {record_name}")
            #print(f"    - Tần số lấy mẫu (fs): {header.fs} Hz")
            #print(f"    - Số kênh tín hiệu: {header.n_sig}")
            #print(f"    - Tên các kênh: {header.sig_name}")
            #print(f"    - Ghi chú (Comments): {header.comments}")
            #print("-" * 50)

            for comment in header.comments:
                if comment.startswith("diagnosis:"):
                    diagnosis_str = comment.split(":")[1].strip().lower()

                    """""
                    if diagnosis_str == r'als':
                        als+=1
                    elif diagnosis_str == r'mainly bulbar als':
                        mainly_bulbar_als+=1
                    elif diagnosis_str == r'healty controls':
                        healty_controls+=1
                    elif diagnosis_str == r'plexus scapulohumoral dystrophy':
                        plexus_scapulohumoral_dystrophy+=1
                    elif diagnosis_str == r'polymyositis':
                        polymyositis+=1
                    elif diagnosis_str == r'myopathy unclassified':
                        myopathy_unclassified+=1
                    
                    """
                    break
                    
        except Exception as e:
            print(f"Lỗi khi đọc file {record_name}: {e}")
    """
    print("=" * 50)
    print(f'als: {als}')
    print(f'mainly bulbar als: {mainly_bulbar_als}')
    print(f'healty controls: {healty_controls}')
    print(f'plexus scapulohumoral dystrophy: {plexus_scapulohumoral_dystrophy}')
    print(f'polymyositis: {polymyositis}')
    print(f'myopathy unclassified: {myopathy_unclassified}')
    print(f'Total: {als + mainly_bulbar_als + healty_controls + plexus_scapulohumoral_dystrophy + polymyositis + myopathy_unclassified}')
    print("=" * 50)
     """   

# --- CÁCH SỬ DỤNG ---
# Thay 'dataset_n2001' bằng thư mục chứa dữ liệu thực tế của bạn
folder_path = r'C:\Users\Admin\Documents\PlatformIO\Projects\EMG_AI\archive' 
inspect_header_files(folder_path, max_show=475)