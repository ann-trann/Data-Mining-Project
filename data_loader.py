import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

def load_kmeans_data():
    data_path = os.path.join(current_dir, 'data', 'mall_customers.csv')
    data = pd.read_csv(data_path)
    return data

def load_decision_tree_data():
    data_path = os.path.join(current_dir, 'data', 'mobile_purchase.csv')
    data = pd.read_csv(data_path)
    return data

def load_naive_bayes_data():
    data_path = os.path.join(current_dir, 'data', 'weather_data.csv')
    data = pd.read_csv(data_path)
    return data