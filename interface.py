import gradio as gr
from data_loader import load_kmeans_data, load_decision_tree_data, load_naive_bayes_data, load_transaction_data
from data_adder import add_kmeans_data, add_decision_tree_data
from algorithms.kmeans import run_kmeans_interactive_3d
from algorithms.decision_tree import run_decision_tree
from algorithms.naive_bayes import run_naive_bayes
from algorithms.association_rules import run_association_rules


def create_interface():
    # Load dữ liệu ban đầu
    kmeans_df = load_kmeans_data()
    dt_df = load_decision_tree_data()
    nb_df = load_naive_bayes_data()
    transaction_df = load_transaction_data()

    with gr.Blocks() as interface:
        gr.Markdown("# Data Mining Project")
        
        with gr.Tabs():
            
            #==================== Association Rules tab ====================#
            with gr.Tab("Frequent Itemsets & Association Rules"):
                with gr.Row():
                    # Data display column
                    with gr.Column(scale=2):
                        transaction_data = gr.DataFrame(value=transaction_df)
                    
                    # Input column
                    with gr.Column(scale=1):
                        min_support = gr.Slider(0.01, 1.0, value=0.5, step=0.01, label="Minimum Support")
                
                # Run button
                ar_run_btn = gr.Button("Find Frequent Itemsets")
                
                # Outputs
                ar_plot = gr.Plot()
                fi_output = gr.Textbox(label="Frequent Itemsets")
                ar_output = gr.Textbox(label="Association Rules")


            #==================== Naive Bayes tab ====================#
            with gr.Tab("Naive Bayes"):
                with gr.Row():
                    # Input column
                    with gr.Column(scale=1):
                        outlook = gr.Dropdown(choices=['Sunny', 'Overcast', 'Rainy'], label="Outlook")
                        temperature = gr.Dropdown(choices=['Hot', 'Mild', 'Cool'], label="Temperature")
                        humidity = gr.Dropdown(choices=['High', 'Normal'], label="Humidity")
                        wind = gr.Dropdown(choices=['Weak', 'Strong'], label="Wind")
                        use_laplace = gr.Checkbox(label="Use Laplace Smoothing")
                        nb_new_sample_btn = gr.Button("New Sample")
                        nb_new_sample = gr.Textbox(label="New Sample")
                    
                    # Data display column
                    with gr.Column(scale=2):
                        nb_data = gr.DataFrame(value=nb_df)

                nb_run_btn = gr.Button("Classification using Naive Bayes")
                nb_plot = gr.Plot()

            # Hàm cập nhật nội dung Textbox
            def create_new_sample_text(outlook, temperature, humidity, wind):
                return f"Outlook: {outlook}, Temperature: {temperature}, Humidity: {humidity}, Wind: {wind}"

            # Gắn sự kiện `change` để cập nhật Textbox khi giá trị thay đổi
            outlook.change(
                create_new_sample_text,
                inputs=[outlook, temperature, humidity, wind],
                outputs=[nb_new_sample]
            )
            temperature.change(
                create_new_sample_text,
                inputs=[outlook, temperature, humidity, wind],
                outputs=[nb_new_sample]
            )
            humidity.change(
                create_new_sample_text,
                inputs=[outlook, temperature, humidity, wind],
                outputs=[nb_new_sample]
            )
            wind.change(
                create_new_sample_text,
                inputs=[outlook, temperature, humidity, wind],
                outputs=[nb_new_sample]
            )

            
            #==================== Decision Tree tab ====================#
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
                        success_markdown_dt = gr.Markdown(visible=False)
                    
                    # Cột hiển thị bảng dữ liệu
                    with gr.Column(scale=2):
                        dt_data = gr.DataFrame(value=dt_df)

                dt_run_btn = gr.Button("Run Decision Tree")
                dt_plot = gr.Plot()


            #==================== K-means tab ====================#
            with gr.Tab("K-means Clustering"):
                with gr.Row():
                    # Cột nhập dữ liệu
                    with gr.Column(scale=1):
                        gender = gr.Dropdown(choices=['Male', 'Female'], label="Gender")
                        age = gr.Number(label="Age")
                        income = gr.Number(label="Annual Income (k$)")
                        spending = gr.Slider(1, 100, label="Spending Score")
                        kmeans_add_btn = gr.Button("Add Data")
                        success_markdown_kmeans = gr.Markdown(visible=False)
                    
                    # Cột hiển thị bảng dữ liệu
                    with gr.Column(scale=2):
                        kmeans_data = gr.DataFrame(value=kmeans_df)
                
                n_clusters = gr.Slider(2, 10, value=3, step=1, label="Number of Clusters")
                kmeans_run_btn = gr.Button("Run K-means")
                kmeans_plot = gr.Plot()
                kmeans_output = gr.DataFrame()
            

            #============================================================#



        # Event handlers
        def show_kmeans_success(_):
            return gr.Markdown("✅ Data successfully added to K-means dataset", visible=True)
        
        def hide_kmeans_success(_):
            return gr.Markdown("", visible=False)

        def show_dt_success(_):
            return gr.Markdown("✅ Data successfully added to Decision Tree dataset", visible=True)
        
        def hide_dt_success(_):
            return gr.Markdown("", visible=False)

        def create_new_sample_text(outlook, temperature, humidity, wind):
            return f"Outlook: {outlook}, Temperature: {temperature}, Humidity: {humidity}, Wind: {wind}"

        # K-means tab event handlers
        kmeans_add_btn.click(
            add_kmeans_data,
            inputs=[kmeans_data, gender, age, income, spending],
            outputs=[kmeans_data]
        ).then(
            show_kmeans_success,
            inputs=[],
            outputs=[success_markdown_kmeans]
        ).then(
            hide_kmeans_success,
            inputs=[],
            outputs=[success_markdown_kmeans],
            js="() => new Promise(resolve => setTimeout(resolve, 1000))"
        )

        # Decision Tree tab event handlers
        dt_add_btn.click(
            add_decision_tree_data,
            inputs=[dt_data, age_group, income_level, region, credit_rating, buy_mobile],
            outputs=[dt_data]
        ).then(
            show_dt_success,
            inputs=[],
            outputs=[success_markdown_dt]
        ).then(
            hide_dt_success,
            inputs=[],
            outputs=[success_markdown_dt],
            js="() => new Promise(resolve => setTimeout(resolve, 1000))"
        )
        
        # Run buttons event handlers
        kmeans_run_btn.click(
            run_kmeans_interactive_3d,
            inputs=[kmeans_data, n_clusters],
            outputs=[kmeans_plot, kmeans_output]
        )

        dt_run_btn.click(
            run_decision_tree,
            inputs=[dt_data],
            outputs=[dt_plot]
        )

        # Naive Bayes event handlers
        nb_new_sample_btn.click(
            create_new_sample_text,
            inputs=[outlook, temperature, humidity, wind],
            outputs=[nb_new_sample]
        )

        nb_run_btn.click(
            run_naive_bayes,
            inputs=[nb_data, outlook, temperature, humidity, wind, use_laplace],
            outputs=[nb_plot]
        )

        # Association Rules event handler
        ar_run_btn.click(
            run_association_rules,
            inputs=[transaction_data, min_support],
            outputs=[ar_plot, fi_output, ar_output]
        )

    return interface