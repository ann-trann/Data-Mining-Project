import pandas as pd
from collections import defaultdict
from itertools import combinations

def read_csv(file_path):
    """Đọc dữ liệu từ tệp CSV."""
    return pd.read_csv(file_path)

def get_equivalence_classes(df, attributes):
    """Tìm các lớp tương đương dựa trên tập thuộc tính."""
    # Tạo một tuple các giá trị cho mỗi đối tượng
    groups = {}
    for idx, row in df.iterrows():
        # Chuyển tuple key thành string để có thể serialize JSON
        key = ';'.join(str(row[attr]) for attr in attributes)
        if key not in groups:
            groups[key] = []
        groups[key].append(row['ID'])
    return groups

def lower_approximation(equivalence_classes, X):
    """Tính xấp xỉ dưới của X qua tập thuộc tính B."""
    lower_approx = set()
    for key, indices in equivalence_classes.items():
        if set(indices).issubset(X):
            lower_approx.update(indices)
    return lower_approx

def upper_approximation(equivalence_classes, X):
    """Tính xấp xỉ trên của X qua tập thuộc tính B."""
    upper_approx = set()
    for key, indices in equivalence_classes.items():
        if set(indices) & X:
            upper_approx.update(indices)
    return upper_approx

def accuracy_of_rough_set(lower_approx, upper_approx):
    """Tính độ chính xác của tập thô."""
    if not upper_approx:
        return 0
    return len(lower_approx) / len(upper_approx)

def calculate_dependency(data, condition_attributes, decision_attribute):
    """Tính phụ thuộc thuộc tính."""
    equivalence_classes_condition = get_equivalence_classes(data, condition_attributes)
    equivalence_classes_decision = get_equivalence_classes(data, [decision_attribute])

    lower_approximation_size = 0
    for condition_class in equivalence_classes_condition.values():
        for decision_class in equivalence_classes_decision.values():
            if set(condition_class).issubset(decision_class):
                lower_approximation_size += len(condition_class)
                break

    dependency = lower_approximation_size / len(data)
    return dependency

def discernibility_matrix(data, decision_attribute):
    """Tạo ma trận phân biệt."""
    n = len(data)
    attributes = data.columns[1:-1]  # Bỏ qua cột ID và cột quyết định
    matrix = [[set() for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if data.iloc[i][decision_attribute] != data.iloc[j][decision_attribute]:
                for attr in attributes:
                    if data.iloc[i][attr] != data.iloc[j][attr]:
                        matrix[i][j].add(attr)
    return matrix

def discernibility_function(matrix):
    """Tạo hàm phân biệt từ ma trận phân biệt."""
    d_function = set()
    for row in matrix:
        for cell in row:
            if cell:
                d_function.add(frozenset(cell))
    return d_function

def find_reducts(d_function, attributes):
    """Tìm các rút gọn từ hàm phân biệt."""
    reducts = []
    for r in range(1, len(attributes) + 1):
        for comb in combinations(attributes, r):
            comb_set = frozenset(comb)
            if all(any(comb_set.issuperset(term) for term in d_function) for term in d_function):
                reducts.append(comb_set)
    # Loại bỏ các rút gọn không cần thiết
    minimal_reducts = []
    for reduct in reducts:
        if not any(reduct > other for other in reducts):
            minimal_reducts.append(reduct)
    return minimal_reducts

def format_equivalence_by_attribute(equivalence_classes, attribute):
    """Format kết quả lớp tương đương theo một thuộc tính."""
    result = f"Lớp tương đương theo thuộc tính '{attribute}':\n"
    for key, ids in equivalence_classes.items():
        # key đã là string
        result += f"- Nhóm {key}: {sorted(ids)}\n"
    return result

def format_all_equivalence_classes(equivalence_classes):
    """Format kết quả lớp tương đương theo tất cả thuộc tính."""
    result = "Lớp tương đương theo tất cả các thuộc tính:\n"
    for key, ids in equivalence_classes.items():
        # key đã là string
        result += f"- Nhóm {key}: {sorted(ids)}\n"
    return result

def format_approximations(lower_approx, upper_approx):
    """Format kết quả xấp xỉ trên và dưới."""
    result = "Xấp xỉ dưới của X qua tập thuộc tính B:\n"
    result += f"- {sorted(lower_approx)}\n\n"
    result += "Xấp xỉ trên của X qua tập thuộc tính B:\n"
    result += f"- {sorted(upper_approx)}"
    return result

def format_accuracy(lower_approx, upper_approx):
    """Format kết quả độ chính xác của tập thô."""
    accuracy = accuracy_of_rough_set(lower_approx, upper_approx)
    return f"Độ chính xác của tập thô: {accuracy:.2f}"

def format_dependency(dependency, condition_attrs, decision_attr):
    """Format kết quả phụ thuộc thu��c tính."""
    return f"Phụ thuộc thuộc tính của {decision_attr} vào {condition_attrs}: {dependency:.3f}"

def format_reducts(reducts):
    """Format kết quả các rút gọn."""
    result = "Các rút gọn (reducts) của hệ quyết định:\n"
    for i, reduct in enumerate(reducts, 1):
        result += f"{i}. {sorted(reduct)}\n"
    return result

def perform_rough_set_analysis(df):
    """Thực hiện phân tích tập thô."""
    # Lấy các thuộc tính (bỏ qua cột ID và cột quyết định)
    attributes = list(df.columns[1:-1])
    decision_attr = df.columns[-1]
    
    # 1. Lớp tương đương theo từng thuộc tính
    equivalence_by_attr = {}
    formatted_eq_by_attr = {}
    for attribute in attributes:
        eq_classes = get_equivalence_classes(df, [attribute])
        # Không cần chuyển đổi key vì đã là string
        equivalence_by_attr[attribute] = eq_classes
        formatted_eq_by_attr[attribute] = format_equivalence_by_attribute(eq_classes, attribute)
    
    # 2. Lớp tương đương theo tất cả các thuộc tính
    eq_classes_all = get_equivalence_classes(df, attributes)
    formatted_eq_all = format_all_equivalence_classes(eq_classes_all)
    
    # 3. Tập X (các đối tượng cần xấp xỉ)
    X = set(df["ID"])
    
    # 4. Xấp xỉ dưới và trên
    lower_approx = lower_approximation(eq_classes_all, X)
    upper_approx = upper_approximation(eq_classes_all, X)
    accuracy = accuracy_of_rough_set(lower_approx, upper_approx)
    formatted_approximations = format_approximations(lower_approx, upper_approx)
    formatted_accuracy = format_accuracy(lower_approx, upper_approx)
    
    # 5. Tính phụ thuộc thuộc tính
    dependency = calculate_dependency(df, attributes, decision_attr)
    formatted_dependency = format_dependency(dependency, attributes, decision_attr)
    
    # 6. Tìm các rút gọn
    matrix = discernibility_matrix(df, decision_attr)
    d_function = discernibility_function(matrix)
    reducts = find_reducts(d_function, attributes)
    formatted_reducts = format_reducts(reducts)
    
    # Đóng gói kết quả với cả dữ liệu gốc và định dạng
    return {
        "equivalence_by_attribute": {
            "raw": equivalence_by_attr,
            "formatted": formatted_eq_by_attr
        },
        "equivalence_all_attributes": {
            "raw": eq_classes_all,
            "formatted": formatted_eq_all
        },
        "approximations": {
            "raw": {
                "lower": list(lower_approx),
                "upper": list(upper_approx),
                "accuracy": float(accuracy)
            },
            "formatted": formatted_approximations,
            "formatted_accuracy": formatted_accuracy
        },
        "dependency": {
            "raw": float(dependency),  # Đảm bảo dependency là số float
            "formatted": formatted_dependency
        },
        "reducts": {
            "raw": [list(reduct) for reduct in reducts],  # Chuyển set thành list
            "formatted": formatted_reducts
        }
    }