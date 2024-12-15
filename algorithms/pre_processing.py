def generate_descriptive_stats(df):
    """
    Generate comprehensive descriptive statistics for a DataFrame
    
    Args:
        df (pandas.DataFrame): Input DataFrame to analyze
    
    Returns:
        dict: Descriptive statistics for numeric and categorical columns
    """
    descriptive_stats = {}
    
    # For numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        descriptive_stats['Numeric Columns'] = {
            col: {
                'Count': int(df[col].count()),  # Convert to int
                'Mean': float(round(df[col].mean(), 2)),  # Convert to float
                'Std Dev': float(round(df[col].std(), 2)),
                'Min': float(round(df[col].min(), 2)),
                '25%': float(round(df[col].quantile(0.25), 2)),
                'Median': float(round(df[col].median(), 2)),
                '75%': float(round(df[col].quantile(0.75), 2)),
                'Max': float(round(df[col].max(), 2))
            } for col in numeric_cols
        }
    
    # For categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        descriptive_stats['Categorical Columns'] = {
            col: {
                'Count': int(df[col].count()),  # Convert to int
                'Unique Values': int(df[col].nunique()),
                'Most Common': str(df[col].mode().tolist()[0]),
                'Value Frequencies': {str(k): int(v) for k, v in dict(df[col].value_counts().head()).items()}
            } for col in categorical_cols
        }
    
    return descriptive_stats

