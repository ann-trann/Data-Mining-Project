import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from itertools import combinations

def preprocess_transactions(df):
    """
    Chuyển đổi DataFrame thành từ điển giao dịch.
    """
    transactions = {}
    
    # Kiểm tra dữ liệu
    if df.empty:
        raise ValueError("Không tìm thấy giao dịch")
    
    
    # Chuyển đổi từng dòng thành tập các mặt hàng
    for index, row in df.iterrows():
        items = set(row.dropna().astype(str))
        transactions[index] = items
    
    return transactions


def calculate_support(itemset, transactions):
    """
    Tính support cho một tập mặt hàng.
    """
    count = sum(1 for items in transactions.values() if itemset.issubset(items))
    return count / len(transactions)

def apriori(transactions, min_support):
    """
    Thuật toán Apriori tìm tập phổ biến.
    """
    # Lấy tất cả các mặt hàng duy nhất
    items = set(item for sublist in transactions.values() for item in sublist)
    
    # Bắt đầu với các tập một mặt hàng
    current_itemsets = [{item} for item in items]
    frequent_itemsets = []

    while current_itemsets:
        candidate_frequent_itemsets = []
        
        # Lọc các tập đạt ngưỡng support
        for itemset in current_itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                candidate_frequent_itemsets.append(itemset)
                frequent_itemsets.append((frozenset(itemset), support))

        # Tạo các tập ứng viên mới
        current_itemsets = []
        for i in range(len(candidate_frequent_itemsets)):
            for j in range(i + 1, len(candidate_frequent_itemsets)):
                # Hợp nhất hai tập phổ biến
                union_set = candidate_frequent_itemsets[i].union(candidate_frequent_itemsets[j])
                if len(union_set) == len(candidate_frequent_itemsets[i]) + 1:
                    current_itemsets.append(union_set)

        # Loại bỏ các tập trùng lặp
        current_itemsets = list(set(map(frozenset, current_itemsets)))

    return sorted(frequent_itemsets, key=lambda x: len(x[0]))


def find_maximal_frequent_itemsets(frequent_itemsets):
    """
    Find maximal frequent itemsets.
    
    Args:
        frequent_itemsets (list): List of frequent itemsets with support
    
    Returns:
        list: Maximal frequent itemsets
    """
    maximal_itemsets = []
    for itemset1, _ in frequent_itemsets:
        is_maximal = True
        for itemset2, _ in frequent_itemsets:
            if itemset1 != itemset2 and itemset1.issubset(itemset2):
                is_maximal = False
                break
        if is_maximal:
            maximal_itemsets.append(itemset1)
    return maximal_itemsets


def calculate_confidence(A, B, transactions):
    """
    Calculate confidence for an association rule.
    
    Args:
        A (set): Antecedent set
        B (set): Consequent set
        transactions (dict): Transaction dictionary
    
    Returns:
        float: Confidence value
    """
    union_set = A.union(B)
    
    support_union = calculate_support(union_set, transactions)
    support_A = calculate_support(A, transactions)
    
    return support_union / support_A if support_A > 0 else 0


def generate_itemsets_optimized(transactions, min_support):
    """
    Tối ưu hóa thuật toán Apriori.
    """
    # Tìm support của các mặt hàng đơn
    single_item_supports = {}
    
    # Tất cả các mặt hàng
    all_items = set(item for sublist in transactions.values() for item in sublist)
    
    # Tính support cho từng mặt hàng
    for item in all_items:
        support = calculate_support({item}, transactions)
        if support >= min_support:
            single_item_supports[item] = support
    
    # Tìm tập hai mặt hàng
    two_item_itemsets = []
    for item_combo in combinations(single_item_supports.keys(), 2):
        itemset = set(item_combo)
        support = calculate_support(itemset, transactions)
        if support >= min_support:
            two_item_itemsets.append((itemset, support))
    
    # Kết hợp kết quả
    results = [
        (frozenset([item]), support) for item, support in single_item_supports.items()
    ] + two_item_itemsets
    
    return sorted(results, key=lambda x: (len(x[0]), x[1]), reverse=True)


def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    """
    Generate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets (list): List of frequent itemsets
        transactions (dict): Transaction dictionary
        min_confidence (float): Minimum confidence threshold
    
    Returns:
        list: Association rules with confidence
    """
    rules = []
    
    for itemset, _ in frequent_itemsets:
        itemset = set(itemset)
        
        # Generate all possible rule combinations
        for r in range(1, len(itemset)):
            for A in combinations(itemset, r):
                A = set(A)
                B = itemset - A
                
                if B:
                    confidence = calculate_confidence(A, B, transactions)
                    if confidence >= min_confidence:
                        rules.append((A, B, confidence))
    
    return rules


def plot_association_rules(frequent_itemsets):
    """
    Create a bar plot of itemset supports.
    
    Args:
        frequent_itemsets (list): List of frequent itemsets with support
    
    Returns:
        str: Base64 encoded plot image
    """
    plt.figure(figsize=(10, 6))
    itemsets = [', '.join(map(str, itemset)) for itemset, _ in frequent_itemsets]
    supports = [support for _, support in frequent_itemsets]
    
    plt.bar(itemsets, supports)
    plt.title('Frequent Itemsets Support')
    plt.xlabel('Itemsets')
    plt.ylabel('Support')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return plot_base64


def run_association_rules(df, min_support=0.4, min_confidence=0.4):
    # Tiền xử lý giao dịch
    transactions = preprocess_transactions(df)
    
    # Tìm tập phổ biến bằng thuật toán Apriori
    frequent_itemsets = apriori(transactions, min_support)
    
    # Các bước còn lại giữ nguyên như trước
    plot_base64 = plot_association_rules(frequent_itemsets)
    
    # Tìm tập phổ biến tối đại
    maximal_frequent_itemsets = find_maximal_frequent_itemsets(frequent_itemsets)
    
    # Sinh luật kết hợp
    association_rules = generate_association_rules(
        frequent_itemsets, transactions, min_confidence
    )
    
    # Định dạng kết quả
    def format_itemsets(itemsets):
        return '\n'.join([
            f"{set(itemset)}: {support:.2f}" 
            for itemset, support in itemsets
        ])
    
    def format_rules(rules):
        return '\n'.join([
            f"{set(A)} => {set(B)}, confidence: {confidence:.2f}"
            for A, B, confidence in rules
        ])
    
    frequent_itemsets_str = format_itemsets(frequent_itemsets)
    maximal_frequent_itemsets_str = '\n'.join([str(set(itemset)) for itemset in maximal_frequent_itemsets])
    association_rules_str = format_rules(association_rules)
    
    return (
        plot_base64, 
        frequent_itemsets_str, 
        association_rules_str, 
        maximal_frequent_itemsets_str
    )