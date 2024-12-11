# Data-Mining-Project
 
## Giới thiệu 
Dự án này bao gồm các thuật toán khai thác dữ liệu như luật kết hợp, cây quyết định, phân cụm KMeans và Naive Bayes. Dữ liệu được sử dụng bao gồm các tập dữ liệu về khách hàng, mua sắm di động, giao dịch siêu thị và dữ liệu thời tiết. 

## Cấu trúc thư mục
```
Data-Mining-Project/
├── algorithms/
│   ├── association_rules.py
│   ├── decision_tree.py
│   ├── kmeans.py
│   └── naive_bayes.py
├── data/
│   ├── mall_customers.csv
│   ├── mobile_purchase.csv
│   ├── supermarket_transactions.csv
│   └── weather_data.csv
├── app.py
├── data_adder.py
├── data_loader.py
├── interface.py
├── README.md
└── requirements.txt
```

## Yêu cầu hệ thống
- Python 3.x
- Các thư viện được liệt kê trong file `requirements.txt`

## Cài đặt
1. **Clone repository**:
    ```bash
    git clone https://github.com/ann-trann/Data-Mining-Project
    cd Data-Mining-Project
    ```

2. **Cài đặt các thư viện cần thiết**:
    ```bash
    pip install -r requirements.txt
    ```

## Chạy dự án
1. **Chạy ứng dụng**:
    ```bash
    python app.py
    ```

2. **Truy cập giao diện người dùng**:
    Mở trình duyệt web và truy cập `http://127.0.0.1:7860/` để sử dụng ứng dụng.

## Mô tả các file chính
- `app.py`: File chính để chạy ứng dụng.
- `data_adder.py`: Thêm dữ liệu vào hệ thống.
- `data_loader.py`: Tải dữ liệu từ các file CSV.
- `interface.py`: Tạo giao diện người dùng.
- `algorithms/`: Chứa các thuật toán khai thác dữ liệu.
  - `association_rules.py`: Luật kết hợp.
  - `decision_tree.py`: Cây quyết định.
  - `kmeans.py`: KMeans.
  - `naive_bayes.py`: Naive Bayes.
- `data/`: Chứa các tập dữ liệu CSV.

