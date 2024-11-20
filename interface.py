import gradio as gr
from data_loader import load_kmeans_data, load_decision_tree_data, load_naive_bayes_data
from data_adder import add_kmeans_data, add_decision_tree_data, add_naive_bayes_data
from algorithms import run_kmeans, run_decision_tree, run_naive_bayes

def create_interface():
    # Load dữ liệu ban đầu
    kmeans_df = load_kmeans_data()
    dt_df = load_decision_tree_data()
    nb_df = load_naive_bayes_data()

    with gr.Blocks() as interface:
        gr.Markdown("# Data Mining Project")
        
        with gr.Tabs():
            # K-means tab
            with gr.Tab("K-means Clustering"):
                with gr.Row():
                    # Cột nhập dữ liệu
                    with gr.Column(scale=1):
                        gender = gr.Dropdown(choices=['Male', 'Female'], label="Gender")
                        age = gr.Number(label="Age")
                        income = gr.Number(label="Annual Income (k$)")
                        spending = gr.Slider(1, 100, label="Spending Score")
                        kmeans_add_btn = gr.Button("Add Data")
                    
                    # Cột hiển thị bảng dữ liệu
                    with gr.Column(scale=2):
                        kmeans_data = gr.DataFrame(value=kmeans_df)
                
                n_clusters = gr.Slider(2, 5, value=3, step=1, label="Number of Clusters")
                kmeans_run_btn = gr.Button("Run K-means")
                kmeans_plot = gr.Plot()
                kmeans_output = gr.DataFrame()

            # Decision Tree tab
            with gr.Tab("Decision Tree"):
                with gr.Row():
                    # Cột nhập dữ liệu
                    with gr.Column(scale=1):
                        age_group = gr.Dropdown(choices=['<20', '21...50', '>50'], label="Age")
                        income_level = gr.Dropdown(choices=['low', 'medium', 'high'], label="Income")
                        region = gr.Dropdown(choices=['USA', 'PK'], label="Region")
                        credit_rating = gr.Dropdown(choices=['Low', 'High'], label="Credit Rating")
                        buy_mobile = gr.Dropdown(choices=['yes', 'no'], label="Buy Mobile")
                        dt_add_btn = gr.Button("Add Data")
                    
                    # Cột hiển thị bảng dữ liệu
                    with gr.Column(scale=2):
                        dt_data = gr.DataFrame(value=dt_df)

                dt_run_btn = gr.Button("Run Decision Tree")
                # dt_output = gr.DataFrame()
                dt_plot = gr.Plot()

            # Naive Bayes tab
            with gr.Tab("Naive Bayes"):
                with gr.Row():
                    # Cột nhập dữ liệu
                    with gr.Column(scale=1):
                        outlook = gr.Dropdown(choices=['Sunny', 'Overcast', 'Rainy'], label="Outlook")
                        temperature = gr.Dropdown(choices=['Hot', 'Mild', 'Cool'], label="Temperature")
                        humidity = gr.Dropdown(choices=['High', 'Normal'], label="Humidity")
                        wind = gr.Dropdown(choices=['Weak', 'Strong'], label="Wind")
                        play_ball = gr.Dropdown(choices=['Yes', 'No'], label="Play ball")
                        nb_add_btn = gr.Button("Add Data")
                    
                    # Cột hiển thị bảng dữ liệu
                    with gr.Column(scale=2):
                        nb_data = gr.DataFrame(value=nb_df)

                nb_run_btn = gr.Button("Run Naive Bayes")
                # nb_output = gr.DataFrame()
                nb_plot = gr.Plot()
        
        # Event handlers
        kmeans_add_btn.click(
            add_kmeans_data,
            inputs=[kmeans_data, gender, age, income, spending],
            outputs=[kmeans_data]
        )
        
        dt_add_btn.click(
            add_decision_tree_data,
            inputs=[dt_data, age_group, income_level, region, credit_rating, buy_mobile],
            outputs=[dt_data]
        )
        
        nb_add_btn.click(
            add_naive_bayes_data,
            inputs=[nb_data, outlook, temperature, humidity, wind, play_ball],
            outputs=[nb_data]
        )
        
        kmeans_run_btn.click(
            run_kmeans,
            inputs=[kmeans_data, n_clusters],
            outputs=[kmeans_plot, kmeans_output]
        )

        dt_run_btn.click(
            run_decision_tree,
            inputs=[dt_data],
            outputs=[dt_plot]
        )

        nb_run_btn.click(
            run_naive_bayes,
            inputs=[nb_data],
            outputs=[nb_plot]
        )

    return interface
