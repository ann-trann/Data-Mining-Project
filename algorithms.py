import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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
    
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    
    results_df = df.copy()
    results_df['Predicted'] = le.inverse_transform(y_pred)
    
    plt.figure(figsize=(12, 5))
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    feature_importance.sort_values().plot(kind='barh')
    plt.title('Feature Importance')
    
    # return results_df, plt.gcf()
    return plt.gcf()

def run_naive_bayes(df):
    le = LabelEncoder()
    df_encoded = df.copy()
    for column in df_encoded.columns:
        df_encoded[column] = le.fit_transform(df_encoded[column])
    
    X = df_encoded.drop('Play ball', axis=1)
    y = df_encoded['Play ball']
    
    model = GaussianNB()
    model.fit(X, y)
    
    y_pred = model.predict(X)
    accuracy = (y_pred == y).mean()
    
    results_df = df.copy()
    results_df['Predicted'] = ['Yes' if pred == 1 else 'No' for pred in y_pred]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(['Accuracy'], [accuracy], color='skyblue')
    plt.title('Naive Bayes Accuracy')
    plt.ylim(0, 1)
    
    confusion = pd.crosstab(results_df['Play ball'], results_df['Predicted'])
    plt.subplot(1, 2, 2)
    plt.imshow(confusion, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # return results_df, plt.gcf()
    return plt.gcf()
