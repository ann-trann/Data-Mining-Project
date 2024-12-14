import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def plot_to_base64(fig=None):
    # Use current figure if no figure is provided
    if fig is None:
        fig = plt.gcf()
    
    # Create a BytesIO buffer
    buffer = io.BytesIO()
    
    # Save the figure to the buffer
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Encode to base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Close the figure to free memory
    plt.close(fig)
    
    return image_base64

def format_association_rules(df, min_support=0.2):
    # Convert DataFrame of lists to clean transactions
    def clean_transaction(transaction):
        # Remove None values and convert to strings
        return [
            str(item).strip() 
            for item in transaction 
            if item is not None and str(item).strip()
        ]
    
    # Process transactions, removing empty lists
    transactions = [
        clean_transaction(trans) 
        for trans in df.values 
        if any(item is not None and str(item).strip() for item in trans)
    ]
    
    # If no valid transactions, return no rules found
    if not transactions:
        return "No association rules found."
    
    # Prepare data for transaction encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    te_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate frequent itemsets
    frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 0)]
    
    # If no frequent itemsets, return no rules found
    if frequent_itemsets.empty:
        return "No association rules found."
    
    # Attempt to generate association rules with different method signatures
    try:
        # Try the newer method first
        rules = association_rules(
            transactions, 
            min_support=min_support, 
            min_confidence=0.5
        )
    except TypeError:
        try:
            # Try the method that requires num_itemsets
            rules = association_rules(
                frequent_itemsets, 
                metric="confidence", 
                min_threshold=0.5,
                num_itemsets=len(frequent_itemsets)
            )
        except Exception as e:
            # If all methods fail, return error message
            print(f"Error generating association rules: {e}")
            return "Unable to generate association rules."
    
    # Filter out rules with empty antecedents or consequents
    rules = rules[
        (rules['antecedents'].apply(lambda x: len(x) > 0)) & 
        (rules['consequents'].apply(lambda x: len(x) > 0)) &
        (rules['antecedents'].apply(lambda x: all(str(item).strip() != '' for item in x))) &
        (rules['consequents'].apply(lambda x: all(str(item).strip() != '' for item in x)))
    ]

    # If no rules after filtering, return no rules found
    if rules.empty:
        return "No association rules found."

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
    
    return '\n'.join(rules_str)


def run_association_rules(df, min_support=0.5):
    # Convert DataFrame to list of transactions, handling mixed data types
    def process_row(row):
        # Convert each item to string and filter out invalid items
        return [
            str(item).strip() 
            for item in row 
            if pd.notna(item) and str(item).strip()
        ]
    
    # Process the DataFrame into transactions
    transactions = df.apply(process_row, axis=1).tolist()
    
    # Filter out empty transactions
    transactions = [trans for trans in transactions if trans]
    
    # If no valid transactions, raise an error
    if not transactions:
        raise ValueError("No valid transactions found in the dataset")
    
    # Prepare data for apriori algorithm
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
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
    
    # If no frequent itemsets, raise an error
    if frequent_itemsets.empty:
        raise ValueError(f"No frequent itemsets found with min_support={min_support}")
    
    # Prepare labels for the plot
    itemset_labels = [
        '{' + ', '.join(sorted(str(item) for item in itemset if item)) + '}' 
        for itemset in frequent_itemsets['itemsets']
    ]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(frequent_itemsets)), frequent_itemsets['support'], 
           align='center', alpha=0.7, color='skyblue')
    ax.set_title('Frequent Itemsets Support Distribution')
    ax.set_xlabel('Itemsets')
    ax.set_ylabel('Support')
    ax.set_xticks(range(len(frequent_itemsets)))
    ax.set_xticklabels(itemset_labels, rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    
    # Convert plot to base64
    plot_base64 = plot_to_base64(fig)
    
    # Format results
    frequent_itemsets_text = format_frequent_itemsets(frequent_itemsets)
    association_rules_text = format_association_rules(pd.DataFrame(transactions), min_support)

    return plot_base64, frequent_itemsets_text, association_rules_text

# Thêm hàm format_frequent_itemsets để hoàn thiện
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

# Hàm clean_and_filter_itemsets
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



