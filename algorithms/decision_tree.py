import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Decision Tree Algorithm
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
