import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def run_kmeans(df, n_clusters=3):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    results_df = df.copy()
    results_df['Cluster'] = clusters
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                marker='x', s=200, color='red', label='Centroids')
    plt.title('Customer Segments')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score')
    plt.colorbar(scatter)
    plt.legend()
    
    return plt.gcf(), results_df


def run_decision_tree(df):
    le = LabelEncoder()
    df_encoded = df.copy()
    for column in df_encoded.columns:
        df_encoded[column] = le.fit_transform(df_encoded[column])
    
    X = df_encoded.drop('Buy Mobile', axis=1)
    y = df_encoded['Buy Mobile']
    
    model = DecisionTreeClassifier(random_state=42, max_depth=3)  # Limit depth for better visualization
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    
    # Create a figure with two subplots
    plt.figure(figsize=(20, 10))
    
    # Subplot 1: Decision Tree Visualization
    plt.subplot(1, 2, 1)
    plot_tree(model, 
              feature_names=X.columns, 
              class_names=['No', 'Yes'], 
              filled=True, 
              rounded=True, 
              fontsize=10)
    plt.title('Decision Tree Visualization')
    
    # Subplot 2: Feature Importance
    plt.subplot(1, 2, 2)
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    feature_importance.sort_values().plot(kind='barh')
    plt.title('Feature Importance')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return plt.gcf()




def manual_naive_bayes_calculation(df, new_sample):
    """
    Perform manual Naive Bayes calculation with detailed steps
    """
    # Prepare calculation steps
    calculation_steps = []
    
    # Encode categorical features
    le_dict = {}
    df_encoded = df.copy()
    for column in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        le_dict[column] = le
    
    # Target column (Play ball)
    target_column = 'Play ball'
    
    # Separate features and target
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]
    
    # Encode new sample
    new_sample_encoded = {}
    for col, val in new_sample.items():
        new_sample_encoded[col] = le_dict[col].transform([val])[0]
    
    # Calculate class probabilities
    class_names = le_dict[target_column].classes_
    class_probs = {}
    class_counts = {}
    
    for class_name in class_names:
        class_index = le_dict[target_column].transform([class_name])[0]
        class_count = (y == class_index).sum()
        class_prob = class_count / len(y)
        class_probs[class_name] = class_prob
        class_counts[class_name] = class_count
        
        calculation_steps.append(f"P(C{class_name.lower()}) = {class_count}/{len(y)} = {class_prob:.3f}")
    
    # Calculate feature probabilities for each class
    feature_probs = {}
    for class_name in class_names:
        class_index = le_dict[target_column].transform([class_name])[0]
        class_data = df_encoded[y == class_index]
        
        class_feature_probs = {}
        for col in X.columns:
            feature_val = new_sample_encoded[col]
            feature_count = ((class_data[col] == feature_val).sum())
            feature_prob = feature_count / class_counts[class_name]
            
            class_feature_probs[col] = feature_prob
            calculation_steps.append(
                f"P({col} = {le_dict[col].inverse_transform([feature_val])[0]} | C{class_name.lower()}) = "
                f"{feature_count}/{class_counts[class_name]} = {feature_prob:.3f}"
            )
        
        feature_probs[class_name] = class_feature_probs
    
    # Calculate final probabilities
    final_probs = {}
    for class_name in class_names:
        # Multiply all feature probabilities
        feature_prob_prod = 1
        for col in X.columns:
            feature_prob_prod *= feature_probs[class_name][col]
        
        # Multiply by class probability
        final_prob = feature_prob_prod * class_probs[class_name]
        final_probs[class_name] = final_prob
        
        calculation_steps.append(
            f"P(X|C{class_name.lower()}) * P(C{class_name.lower()}) = "
            f"{feature_prob_prod:.3f} * {class_probs[class_name]:.3f} = {final_prob:.3f}"
        )
    
    # Determine prediction
    prediction = max(final_probs, key=final_probs.get)
    calculation_steps.append(f"\nPredicted Class: {prediction}")
    
    return calculation_steps, prediction


def run_naive_bayes(df, outlook=None, temperature=None, humidity=None, wind=None):
    # If a new sample is provided, process it
    if all([outlook, temperature, humidity, wind]):
        return process_new_sample(df, outlook, temperature, humidity, wind)
    
    # Otherwise, show default visualization
    plt.figure(figsize=(16, 6))
    
    # Subplot for calculation steps
    plt.subplot(1, 2, 1)
    plt.text(0.5, 0.5, "Add a new sample\nto see Naive Bayes\ncalculation steps", 
             horizontalalignment='center', 
             verticalalignment='center', 
             fontsize=10)
    plt.title('Calculation Steps')
    plt.axis('off')
    
    # Subplot for visualization
    plt.subplot(1, 2, 2)
    # Basic visualization of data distribution
    plt.title('Naive Bayes Visualization')
    
    plt.tight_layout()
    return plt.gcf()


# Modified to handle new sample input
def process_new_sample(df, outlook, temperature, humidity, wind):
    new_sample = {
        'Outlook': outlook,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind': wind
    }
    
    # Perform Naive Bayes calculation
    calculation_steps, prediction = manual_naive_bayes_calculation(df, new_sample)
    
    # Prepare calculation steps for display
    calculation_text = "\n".join(calculation_steps)
    
    # Visualization with calculation steps
    plt.figure(figsize=(16, 6))
    
    # Subplot for calculation steps
    plt.subplot(1, 2, 1)
    plt.text(0.2, 0.5, calculation_text, 
             horizontalalignment='left', 
             verticalalignment='center', 
             fontsize=12, 
             family='monospace')
    plt.title('Naive Bayes Calculation Steps')
    plt.axis('off')
    
    # Subplot for visualization
    plt.subplot(1, 2, 2)
    # Basic visualization of data distribution
    feature_counts = df.groupby('Play ball').size()
    plt.bar(feature_counts.index, feature_counts.values)
    plt.title('Class Distribution')
    plt.ylabel('Count')
    
    plt.tight_layout()
    return plt.gcf()






def format_frequent_itemsets(frequent_itemsets):
    # Tạo một set để theo dõi các tập đã xuất hiện
    seen_itemsets = set()
    
    # Sắp xếp lại frequent_itemsets để có thể lọc
    sorted_itemsets = frequent_itemsets.copy()
    
    # Thêm độ dài và sắp xếp
    sorted_itemsets['itemset_length'] = sorted_itemsets['itemsets'].apply(lambda x: len(x))
    sorted_itemsets = sorted_itemsets.sort_values(['itemset_length', 'support'])
    
    frequent_itemsets_str = []
    for _, row in sorted_itemsets.iterrows():
        # Chuyển frozenset sang tuple đã sắp xếp, loại bỏ các chuỗi rỗng
        itemset = tuple(sorted(list(filter(bool, row['itemsets']))))
        
        # Kiểm tra xem tập này đã xuất hiện chưa
        if itemset and itemset not in seen_itemsets:
            seen_itemsets.add(itemset)
            
            itemset_str = '{' + ', '.join(itemset) + '}'
            frequent_itemsets_str.append(f"Tập phổ biến: {itemset_str}: {row['support']:.2f}")
    
    return '\n'.join(frequent_itemsets_str)

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

def run_association_rules(df, min_support=0.2):
    # Prepare data for apriori algorithm
    te = TransactionEncoder()
    te_ary = te.fit(df.values).transform(df.values)
    te_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Generate frequent itemsets (filtering out empty set)
    frequent_itemsets = apriori(te_df, min_support=min_support, use_colnames=True)
    frequent_itemsets = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) > 0)]
    
    # Sort frequent itemsets by support in descending order
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
    
    # Prepare itemset labels for the x-axis
    itemset_labels = ['{' + ', '.join(sorted(list(itemset))) + '}' for itemset in frequent_itemsets['itemsets']]
    
    # Create a visualization of frequent itemsets
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(frequent_itemsets)), frequent_itemsets['support'], 
            align='center', alpha=0.7, color='skyblue')
    plt.title('Frequent Itemsets Support Distribution')
    plt.xlabel('Itemsets')
    plt.ylabel('Support')
    plt.xticks(range(len(frequent_itemsets)), itemset_labels, rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    
    # Format the frequent itemsets and rules as text
    frequent_itemsets_text = format_frequent_itemsets(frequent_itemsets)
    association_rules_text = format_association_rules(df, min_support)

    return plt.gcf(), frequent_itemsets_text, association_rules_text
