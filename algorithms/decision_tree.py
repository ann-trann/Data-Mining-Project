import os
import json
import tempfile
import base64
import io
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def plot_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)  # Close the specific figure
    return plot_data

def run_decision_tree_analysis(data, features, target, criterion='entropy'):
    # Create DataFrame from data
    df = pd.DataFrame(data)
    
    # Prepare data for encoding
    X = df[features]
    y = df[target]
    
    # Encode categorical features and target
    label_encoders = {}
    X_encoded = X.copy()
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X_encoded[column] = le.fit_transform(X[column])
        label_encoders[column] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Handle small dataset scenarios
    if len(df) < 10:  # Điều chỉnh ngưỡng nhỏ hơn
        # Sử dụng train_test_split với test_size nhỏ hơn
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, 
            test_size=0.3,  # Giảm kích thước test set
            random_state=42
        )
    else:
        # Normal train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, 
            test_size=0.2, 
            random_state=42
        )
    
    # Train Decision Tree
    dt_classifier = DecisionTreeClassifier(random_state=42, criterion=criterion)
    dt_classifier.fit(X_train, y_train)
    
    # Tính accuracy trên toàn bộ dataset thay vì chỉ test set
    accuracy = dt_classifier.score(X_encoded, y_encoded)
    
    # Visualize Decision Tree
    fig, ax = plt.subplots(figsize=(20,10))
    plot_tree(dt_classifier, 
              feature_names=features, 
              class_names=target_encoder.classes_, 
              filled=True, 
              rounded=True, 
              fontsize=10,
              ax=ax)
    decision_tree_plot = plot_to_base64(fig)
    
    # Feature Importance
    fig, ax = plt.subplots(figsize=(10,6))
    feature_importance = dt_classifier.feature_importances_
    indices = np.argsort(feature_importance)[::-1]
    
    ax.set_title("Feature Importances")
    ax.bar(range(len(features)), feature_importance[indices])
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([features[i] for i in indices], rotation=45)
    plt.tight_layout()
    feature_importance_plot = plot_to_base64(fig)
    
    # Prepare results
    results = {
        "accuracy": float(accuracy),
        "feature_importance": {
            features[i]: float(feature_importance[i]) 
            for i in range(len(features))
        },
        "decision_tree_plot": decision_tree_plot,
        "feature_importance_plot": feature_importance_plot,
        "classes": list(target_encoder.classes_)
    }
    
    return results