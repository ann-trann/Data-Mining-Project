import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from algorithms.pre_processing import generate_descriptive_stats
from algorithms.association_rules import run_association_rules
from algorithms.reduct import perform_rough_set_analysis
from algorithms.naive_bayes import predict_naive_bayes
from algorithms.decision_tree import run_decision_tree_analysis
from algorithms.kmeans import run_kmeans_3d_clustering


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#============================ Routes ============================#
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/preprocess')
def pre_processing_page():
    return render_template('pre_processing.html')

@app.route('/association-rules')
def association_rules_page():
    return render_template('association_rules.html')

@app.route('/reduct')
def reduct_page():
    return render_template('reduct.html')

@app.route('/naive-bayes')
def naive_bayes_page():
    return render_template('naive_bayes.html')

@app.route('/decision-tree')
def decision_tree_page():
    return render_template('decision_tree.html')

@app.route('/k-means')
def kmeans_page():
    return render_template('kmeans.html')



#============================ Upload File ============================#
@app.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "Không có file được tải lên"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Không có file được chọn"}), 400
    
    try:
        # Lưu file với đường dẫn đầy đủ
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Improved file reading with multiple parsing strategies
        try:
            # Try reading with default settings
            df = pd.read_csv(filepath)
        except Exception:
            try:
                # Try reading without headers
                df = pd.read_csv(filepath, header=None)
            except Exception:
                # Try reading with different delimiters
                df = pd.read_csv(filepath, sep=r'\s+')
        
        # Ensure we have data
        if df.empty:
            return jsonify({"error": "File rỗng hoặc không thể đọc"}), 400
        
        # Clean column names (remove any special characters)
        df.columns = [str(col).strip().replace(' ', '_') for col in df.columns]
        
        # Convert to records, handling potential non-serializable types
        records = df.apply(lambda row: row.apply(lambda x: str(x) if pd.notna(x) else '').to_dict(), axis=1).tolist()
        
        return jsonify({
            "data": records,
            "columns": list(df.columns),
            "filepath": filepath  # Trả về đường dẫn đầy đủ
        })
    
    except Exception as e:
        return jsonify({"error": f"Lỗi xử lý file: {str(e)}"}), 500



#============================ Preprocess Data ============================#
@app.route('/run-preprocess', methods=['POST'])
def preprocess_data():
    try:
        # Receive parameters from frontend
        data = request.json
        
        # Validate input
        if not data or 'filepath' not in data:
            return jsonify({"error": "Missing required filepath"}), 400
        
        # Read the CSV file
        df = pd.read_csv(data['filepath'])
        
        # Generate descriptive statistics
        descriptive_stats = generate_descriptive_stats(df)
        
        # Handle missing values
        missing_values = {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()}
        
        # 2. Imputation strategy
        # Numeric columns: fill with mean
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            df[col].fillna(df[col].mean(), inplace=True)
        
        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)
        
        # One-hot encoding for categorical variables
        df_encoded = pd.get_dummies(df, columns=list(categorical_cols))
        
        return jsonify({
            "descriptive_stats": descriptive_stats,
            "missing_values": missing_values,
            "preprocessed_columns": list(df_encoded.columns),
            "preprocessed_data": df_encoded.head(10).to_dict(orient='records')
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Lỗi xử lý tiền xử lý: {str(e)}"}), 500


#============================ Association Rules ============================#
@app.route('/run-association-rules', methods=['POST'])
def run_association_rules_route():
    try:
        # Receive parameters from frontend
        data = request.json
        
        # Validate input
        if not data or 'filepath' not in data:
            return jsonify({"error": "Missing required filepath"}), 400
        
        # Read the CSV file
        df = pd.read_csv(data['filepath'])
        
        # Get min_support, min_confidence, default to 0.4 if not provided
        min_support = data.get('min_support', 0.4)
        min_confidence = data.get('min_confidence', 0.4)
        
        # Run association rules analysis
        plot_base64, frequent_itemsets, association_rules, maximal_frequent_itemsets = run_association_rules(
            df, 
            min_support=min_support,
            min_confidence=min_confidence
        )
        
        return jsonify({
            "plot": plot_base64,
            "frequent_itemsets": frequent_itemsets,
            "association_rules": association_rules,
            "maximal_frequent_itemsets": maximal_frequent_itemsets
        })
    
    except ValueError as ve:
        # Handle specific value errors (like no transactions)
        return jsonify({"error": str(ve)}), 400
    
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({"error": "Unexpected error processing association rules"}), 500





#============================ Reduct ============================#
@app.route('/run-reduct', methods=['POST'])
def run_reduct_analysis():
    try:
        # Receive parameters from frontend
        data = request.json
        
        # Validate input
        if not data or 'filepath' not in data:
            return jsonify({"error": "Missing required filepath"}), 400
        
        # Read the CSV file
        df = pd.read_csv(data['filepath'])
        
        # Run Rough Set Theory analysis
        result = perform_rough_set_analysis(df)
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({"error": str(e)}), 500
    



#============================ Predict Naive Bayes ============================#
@app.route('/predict-naive-bayes', methods=['POST'])
def predict_naive_bayes_route():
    try:
        features = request.form.getlist('features[]')
        target = request.form.get('target')
        filepath = request.form.get('filepath')
        
        # Get Laplace smoothing parameter
        use_laplace = request.form.get('use_laplace', 'false').lower() == 'true'
        
        # Kiểm tra tồn tại file
        if not os.path.exists(filepath):
            return jsonify({"error": f"File không tồn tại: {filepath}"}), 404
        
        result = predict_naive_bayes(
            filepath, 
            features, 
            target, 
            use_laplace=use_laplace
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    



#============================ Decision Tree ============================#
@app.route('/run-decision-tree', methods=['POST'])
def decision_tree_route():
    try:
        # Receive parameters from frontend
        data = request.json
        
        # Validate input
        if not data or 'filepath' not in data or 'features' not in data or 'target' not in data:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Get data from filepath or use selected data
        if 'selected_data' in data and data['selected_data']:
            selected_data = data['selected_data']
        else:
            # If no selected data, load from file
            df = pd.read_csv(data['filepath'])
            selected_data = df.to_dict('records')
        
        # Get criterion (default to entropy if not provided)
        criterion = data.get('criterion', 'entropy')
        
        # Run analysis
        results = run_decision_tree_analysis(
            selected_data, 
            data['features'], 
            data['target'],
            criterion
        )
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({"error": str(e)}), 500
    



#============================ K-Means ============================#
@app.route('/run-kmeans', methods=['POST'])
def kmeans_route():
    try:
        # Receive parameters from frontend
        data = request.json
        
        # Validate input
        if not data or 'filepath' not in data or 'n_clusters' not in data:
            return jsonify({"error": "Missing required parameters"}), 400
        
        # Run 3D K-means clustering
        results = run_kmeans_3d_clustering(
            data['filepath'], 
            int(data['n_clusters'])
        )
        
        return jsonify(results)
    
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return jsonify({"error": str(e)}), 500
    

if __name__ == '__main__':
    app.run(debug=True)