import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import base64
from io import BytesIO

def analyze_correlation(file_path, x_col, y_col):
    """
    Phân tích tương quan chi tiết giữa hai biến từ file CSV
    
    Parameters:
    file_path (str): Đường dẫn đến file CSV
    x_col (str): Tên cột biến x
    y_col (str): Tên cột biến y
    """
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    x = df[x_col].values
    y = df[y_col].values
    n = len(x)
    
    # 1. Tính các giá trị trung bình
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    print("\n1. Giá trị trung bình:")
    print(f"Trung bình {x_col}: {mean_x:.4f}")
    print(f"Trung bình {y_col}: {mean_y:.4f}")
    
    # 2. Tính phương sai
    var_x = np.var(x, ddof=1)  # ddof=1 for sample variance
    var_y = np.var(y, ddof=1)
    print("\n2. Phương sai:")
    print(f"Phương sai {x_col}: {var_x:.4f}")
    print(f"Phương sai {y_col}: {var_y:.4f}")
    
    # 3. Tính độ lệch chuẩn
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    print("\n3. Độ lệch chuẩn:")
    print(f"Độ lệch chuẩn {x_col}: {std_x:.4f}")
    print(f"Độ lệch chuẩn {y_col}: {std_y:.4f}")
    
    # 4. Tính hiệp phương sai
    covariance = np.cov(x, y)[0][1]
    print("\n4. Hiệp phương sai:")
    print(f"Hiệp phương sai {x_col},{y_col}: {covariance:.4f}")
    
    # 5. Tính hệ số tương quan
    correlation, p_value = stats.pearsonr(x, y)
    print("\n5. Hệ số tương quan Pearson:")
    print(f"r = {correlation:.4f}")
    print(f"p-value = {p_value:.4f}")
    
    # 6. Tính hệ số hồi quy b1
    b1 = covariance / var_x
    print("\n6. Hệ số hồi quy:")
    print(f"b1 = {b1:.4f}")
    print(f"b1 * sigma_x / sigma_y = {b1 * std_x / std_y:.4f}")
    
    # 7. Thống kê mô tả
    print("\nTHỐNG KÊ MÔ TẢ:")
    stats_df = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
        x_col: [n, mean_x, std_x, min(x), np.percentile(x, 25), np.median(x), np.percentile(x, 75), max(x)],
        y_col: [n, mean_y, std_y, min(y), np.percentile(y, 25), np.median(y), np.percentile(y, 75), max(y)]
    })
    print(stats_df.round(4))
    
    # 8. Kiểm định ý nghĩa thống kê
    print("\nKIỂM ĐỊNH Ý NGHĨA THỐNG KÊ:")
    if p_value < 0.05:
        print(f"Hệ số tương quan có ý nghĩa thống kê (p = {p_value:.4f} < 0.05)")
    else:
        print(f"Hệ số tương quan không có ý nghĩa thống kê (p = {p_value:.4f} > 0.05)")
    
    # 9. Đánh giá mức độ tương quan
    print("\nĐÁNH GIÁ MỨC ĐỘ TƯƠNG QUAN:")
    r_abs = abs(correlation)
    if r_abs < 0.3:
        strength = "yếu"
    elif r_abs < 0.5:
        strength = "trung bình"
    elif r_abs < 0.7:
        strength = "khá"
    else:
        strength = "mạnh"
    
    print(f"Mức độ tương quan: {strength}")
    print(f"Hướng tương quan: {'dương' if correlation > 0 else 'âm'}")
    
    # 10. Vẽ biểu đồ phân tán
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'Biểu đồ phân tán giữa {x_col} và {y_col}')
    
    # Thêm đường hồi quy tuyến tính
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8)
    
    # Thêm công thức đường hồi quy và hệ số tương quan
    equation = f'y = {z[0]:.2f}x + {z[1]:.2f}'
    plt.text(0.05, 0.95, f'Phương trình: {equation}\nr = {correlation:.4f}', 
             transform=plt.gca().transAxes, 
             verticalalignment='top')
    
    plt.grid(True, alpha=0.3)
    
    # Thay plt.show() bằng việc lưu vào buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()  # Đóng figure để giải phóng bộ nhớ
    
    # Tạo dictionary chứa tất cả thông tin thống kê
    detailed_stats = {
        "mean": {
            "x": mean_x,
            "y": mean_y,
            "x_label": x_col,
            "y_label": y_col
        },
        "variance": {
            "x": var_x,
            "y": var_y
        },
        "std": {
            "x": std_x,
            "y": std_y
        },
        "covariance": covariance,
        "correlation": correlation,
        "p_value": p_value,
        "regression": {
            "b1": b1,
            "b1_normalized": b1 * std_x / std_y
        }
    }
    
    return plot_base64, detailed_stats, stats_df
