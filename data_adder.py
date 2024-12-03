import pandas as pd
import gradio as gr


def add_kmeans_data(df, gender, age, income, spending):
    new_row = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Annual Income (k$)': [income],
        'Spending Score (1-100)': [spending]
    })
    updated_df = pd.concat([df, new_row], ignore_index=True)
    return updated_df


def add_decision_tree_data(df, age, income, region, credit_rating, buy_mobile):
    new_row = pd.DataFrame({
        'age': [age],
        'income': [income],
        'Region': [region],
        'credit_rating': [credit_rating],
        'Buy Mobile': [buy_mobile]
    })
    updated_df = pd.concat([df, new_row], ignore_index=True)
    return updated_df


def add_naive_bayes_data(df, outlook, temperature, humidity, wind, play_ball):
    new_row = pd.DataFrame({
        'Outlook': [outlook],
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Wind': [wind],
        'Play ball': [play_ball]
    })
    return pd.concat([df, new_row], ignore_index=True)

