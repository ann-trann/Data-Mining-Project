import pandas as pd

def descriptive_statistics(data):
    # Tạo DataFrame từ dữ liệu
    df = pd.DataFrame(data)
    
    # Thống kê mô tả cho các cột số
    numerical_stats = df.describe()
    
    # Thống kê mô tả cho các cột phân loại
    categorical_stats = df.describe(include=['object'])
    
    # Kiểm tra giá trị thiếu
    missing_values = df.isnull().sum()
    
    # Chuẩn bị kết quả
    results = {
        "numerical_statistics": numerical_stats.to_dict(),
        "categorical_statistics": categorical_stats.to_dict(),
        "missing_values": missing_values.to_dict()
    }
    
    return results
