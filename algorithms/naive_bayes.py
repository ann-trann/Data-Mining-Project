import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import io
import base64

def plot_to_base64(plt):
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    return plot_data

def create_feature_distribution_plot(df, target):
    # Set up the matplotlib figure
    plt.figure(figsize=(16, 10))
    plt.suptitle(f'Feature Distribution by {target}', fontsize=16)
    
    # Get all features except the target
    features = [col for col in df.columns if col != target]
    
    # Create subplots for each feature
    for i, feature in enumerate(features, 1):
        plt.subplot(2, 2, i)
        
        # Use seaborn for a more aesthetic plot
        sns.countplot(
            data=df, 
            x=feature, 
            hue=target, 
            palette='Set2'
        )
        
        plt.title(f'{feature} Distribution')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Only show legend for the first subplot to avoid repetition
        if i == 1:
            plt.legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend([],[], frameon=False)
    
    plt.tight_layout()
    return plot_to_base64(plt)

def predict_naive_bayes(filepath, features, target, use_laplace=False):
    # Read the CSV file
    df = pd.read_csv(filepath)
    
    # Normalize column names (remove underscores if added during upload)
    df.columns = [col.replace('_', ' ') for col in df.columns]
    
    # Normalize target name
    target = target.replace('_', ' ')
    
    # Parse selected features
    selected_features = {}
    for feature_str in features:
        feature, value = feature_str.split(':')
        # Normalize feature name
        feature = feature.replace('_', ' ')
        selected_features[feature] = value
    
    # Prepare the subset of data with selected features
    X = df[[feature for feature in selected_features.keys()]]
    y = df[target]
    
    # Prepare label encoder for target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Prepare feature encoders
    encoders = {}
    X_encoded = []
    
    # Encode each selected feature
    for feature in X.columns:
        le = LabelEncoder()
        encoded_feature = le.fit_transform(X[feature])
        encoders[feature] = le
        X_encoded.append(encoded_feature)
    
    # Transpose to get correct shape for Naive Bayes
    X_encoded = np.array(X_encoded).T
    
    # Train Naive Bayes with optional Laplace smoothing
    nb_model = MultinomialNB(alpha=1.0 if use_laplace else 0.0)
    nb_model.fit(X_encoded, y_encoded)
    
    # Prepare prediction data
    prediction_input = []
    for feature in X.columns:
        # Find the encoded value for the selected feature
        encoded_value = encoders[feature].transform([selected_features[feature]])[0]
        prediction_input.append(encoded_value)
    
    # Reshape prediction input
    prediction_input = np.array(prediction_input).reshape(1, -1)
    
    # Predict
    prediction_encoded = nb_model.predict(prediction_input)
    prediction = le_target.inverse_transform(prediction_encoded)[0]
    
    # Get probabilities
    probabilities = nb_model.predict_proba(prediction_input)
    max_prob_index = np.argmax(probabilities)
    max_probability = probabilities[0][max_prob_index]
    
    # Calculation steps for explanation
    def format_calculation_steps(X_encoded, y_encoded, prediction_input, nb_model, le_target):
        steps = []
        
        # Prior probabilities
        class_counts = np.bincount(y_encoded)
        total_samples = len(y_encoded)
        prior_probs = class_counts / total_samples
        
        steps.append("Prior Probabilities:")
        for cls, prob in zip(le_target.classes_, prior_probs):
            steps.append(f"P({target} = {cls}) = {prob:.4f}")
        
        # Likelihood calculation
        steps.append("\nLikelihood Calculation:")
        for i, input_val in enumerate(prediction_input[0]):
            feature_name = X.columns[i]
            feature_val = selected_features[feature_name]
            
            for cls_idx, cls in enumerate(le_target.classes_):
                # Count occurrences of this feature value for this class
                class_mask = y_encoded == cls_idx
                feature_mask = X_encoded[:, i] == input_val
                feature_class_count = np.sum(class_mask & feature_mask)
                total_class_count = np.sum(class_mask)
                
                # Optional Laplace smoothing
                if use_laplace:
                    unique_feature_vals = len(np.unique(X_encoded[:, i]))
                    likelihood = (feature_class_count + 1) / (total_class_count + unique_feature_vals)
                    steps.append(f"  P({feature_name} = {feature_val} | {target} = {cls}) = ({feature_class_count} + 1) / ({total_class_count} + {unique_feature_vals}) = {likelihood:.4f}")
                else:
                    likelihood = feature_class_count / total_class_count
                    steps.append(f"  P({feature_name} = {feature_val} | {target} = {cls}) = {feature_class_count} / {total_class_count} = {likelihood:.4f}")
        
        # Final prediction probability
        steps.append("\nFinal Prediction Probability:")
        steps.append(f"Predicted {target}: {prediction}")
        steps.append(f"Probability: {max_probability:.4f}")
        
        return "\n".join(steps)
    
    # Create visualization
    plot_data = create_feature_distribution_plot(df, target)
    calculation_text = format_calculation_steps(X_encoded, y_encoded, prediction_input, nb_model, le_target)
    
    return {
        "prediction": prediction,
        "probability": float(max_probability),
        "plot": plot_data,
        "calculation_steps": calculation_text
    }