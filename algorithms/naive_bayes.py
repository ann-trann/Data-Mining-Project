from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Naive Bayes Algorithm
def manual_naive_bayes_calculation(df, new_sample, use_laplace=False):
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
    total_samples = len(y)
    
    for class_name in class_names:
        class_index = le_dict[target_column].transform([class_name])[0]
        class_count = (y == class_index).sum()
        class_prob = class_count / total_samples
        class_probs[class_name] = class_prob
        class_counts[class_name] = class_count
        
        calculation_steps.append(f"P(C{class_name.lower()}) = {class_count}/{total_samples} = {class_prob:.3f}")
    
    # Calculate feature probabilities for each class
    feature_probs = {}
    for class_name in class_names:
        class_index = le_dict[target_column].transform([class_name])[0]
        class_data = df_encoded[y == class_index]
        
        class_feature_probs = {}
        for col in X.columns:
            feature_val = new_sample_encoded[col]
            
            # Laplace smoothing
            if use_laplace:
                unique_feature_count = len(le_dict[col].classes_)
                feature_count = (class_data[col] == feature_val).sum() + 1
                feature_prob = feature_count / (class_counts[class_name] + unique_feature_count)
                calculation_steps.append(
                    f"P({col} = {le_dict[col].inverse_transform([feature_val])[0]} | C{class_name.lower()}) = "
                    f"({(class_data[col] == feature_val).sum()} + 1) / "
                    f"({class_counts[class_name]} + {unique_feature_count}) = {feature_prob:.3f}"
                )
            else:
                # Without Laplace smoothing
                feature_count = (class_data[col] == feature_val).sum()
                feature_prob = feature_count / class_counts[class_name]
                calculation_steps.append(
                    f"P({col} = {le_dict[col].inverse_transform([feature_val])[0]} | C{class_name.lower()}) = "
                    f"{feature_count}/{class_counts[class_name]} = {feature_prob:.3f}"
                )
            
            class_feature_probs[col] = feature_prob
        
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

def run_naive_bayes(df, outlook=None, temperature=None, humidity=None, wind=None, use_laplace=False):
    # If a new sample is provided, process it
    if all([outlook, temperature, humidity, wind]):
        return process_new_sample(df, outlook, temperature, humidity, wind, use_laplace)
    
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

# Modified to handle new sample input with Laplace smoothing
def process_new_sample(df, outlook, temperature, humidity, wind, use_laplace=False):
    new_sample = {
        'Outlook': outlook,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind': wind
    }
    
    # Perform Naive Bayes calculation
    calculation_steps, prediction = manual_naive_bayes_calculation(df, new_sample, use_laplace)
    
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
    plt.title(f'Naive Bayes Calculation Steps\n{"(Laplace Smoothing)" if use_laplace else ""}')
    plt.axis('off')
    
    # Subplot for visualization
    plt.subplot(1, 2, 2)
    # Basic visualization of data distribution
    feature_counts = df.groupby('Play ball').size()
    plt.bar(feature_counts.index, feature_counts.values)
    plt.title(f'Class Distribution\n{"(Laplace Smoothing)" if use_laplace else ""}')
    plt.ylabel('Count')
    
    plt.tight_layout()
    return plt.gcf()
