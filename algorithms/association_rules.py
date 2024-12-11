import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# Clean and filter itemsets
def clean_and_filter_itemsets(itemsets):
    cleaned_itemsets = []
    for itemset in itemsets:
        # Loại bỏ các phần tử rỗng và khoảng trắng
        cleaned_itemset = [
            item.strip() for item in itemset 
            if item and item.strip()
        ]
        
        # Sắp xếp và chuyển thành tuple
        if cleaned_itemset:
            cleaned_itemsets.append(tuple(sorted(cleaned_itemset)))
    
    return cleaned_itemsets


# Format frequent itemsets as text
def format_frequent_itemsets(frequent_itemsets):
    seen_itemsets = set()
    
    # Sao chép và sắp xếp lại frequent_itemsets
    sorted_itemsets = frequent_itemsets.copy()
    sorted_itemsets['itemset_length'] = sorted_itemsets['itemsets'].apply(lambda x: len(x))
    sorted_itemsets = sorted_itemsets.sort_values(['itemset_length', 'support'])
    
    frequent_itemsets_str = []
    for _, row in sorted_itemsets.iterrows():
        # Làm sạch và lọc itemsets
        cleaned_itemsets = clean_and_filter_itemsets([row['itemsets']])[0]
        
        # Kiểm tra xem tập đã xuất hiện chưa
        if cleaned_itemsets and cleaned_itemsets not in seen_itemsets:
            seen_itemsets.add(cleaned_itemsets)
            
            itemset_str = '{' + ', '.join(cleaned_itemsets) + '}'
            frequent_itemsets_str.append(f"Tập phổ biến: {itemset_str}: {row['support']:.2f}")
    
    return '\n'.join(frequent_itemsets_str)


# Format association rules as text
def format_association_rules(df, min_support=0.2):
    # Prepare data for apriori algorithm
    te = TransactionEncoder()
    te_ary = te.fit(df.values).transform(df.values)
    te_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 0)]
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets))
    
    # Filter out rules with:
    # 1. Empty antecedents or consequents
    # 2. Rules with leading comma or empty strings
    rules = rules[
        (rules['antecedents'].apply(lambda x: len(x) > 0)) & 
        (rules['consequents'].apply(lambda x: len(x) > 0)) &
        (rules['antecedents'].apply(lambda x: all(str(item).strip() != '' for item in x))) &
        (rules['consequents'].apply(lambda x: all(str(item).strip() != '' for item in x)))
    ]

    # Format rules
    rules_str = []
    for _, rule in rules.iterrows():
        # Convert antecedents and consequents to sorted lists
        antecedents = sorted(list(rule['antecedents']))
        consequents = sorted(list(rule['consequents']))
        
        # Create formatted strings
        antecedent_str = '{' + ', '.join(antecedents) + '}'
        consequent_str = '{' + ', '.join(consequents) + '}'
        
        rules_str.append(
            f"{antecedent_str} => {consequent_str}, độ tin cậy: {rule['confidence']:.2f}"
        )
    
    return '\n'.join(rules_str) if rules_str else "No association rules found."


# Run the association rules algorithm
def run_association_rules(df, min_support=0.2):
    # Loại bỏ các hàng chứa phần tử rỗng
    df = df.apply(lambda row: [item.strip() for item in row if item and item.strip()], axis=1)
    
    # Prepare data for apriori algorithm
    te = TransactionEncoder()
    te_ary = te.fit(df.values).transform(df.values)
    te_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate frequent itemsets (filtering out empty and whitespace sets)
    frequent_itemsets = apriori(
        te_df, 
        min_support=min_support, 
        use_colnames=True
    )
    frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(
        lambda x: frozenset(item for item in x if item and str(item).strip())
    )
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) > 0]
    
    # Sắp xếp theo support giảm dần
    # frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    
    # Chuẩn bị nhãn cho biểu đồ
    itemset_labels = [
        '{' + ', '.join(sorted(str(item) for item in itemset if item)) + '}' 
        for itemset in frequent_itemsets['itemsets']
    ]
    
    # Tạo biểu đồ
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(frequent_itemsets)), frequent_itemsets['support'], 
            align='center', alpha=0.7, color='skyblue')
    plt.title('Frequent Itemsets Support Distribution')
    plt.xlabel('Itemsets')
    plt.ylabel('Support')
    plt.xticks(range(len(frequent_itemsets)), itemset_labels, rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    
    # Format kết quả
    frequent_itemsets_text = format_frequent_itemsets(frequent_itemsets)
    association_rules_text = format_association_rules(df, min_support)

    return plt.gcf(), frequent_itemsets_text, association_rules_text
