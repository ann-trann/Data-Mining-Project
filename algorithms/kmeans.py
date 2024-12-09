from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objs as go

# K-means Clustering Algorithm
def run_kmeans_interactive_3d(df, n_clusters=3):
    # Select three features for clustering
    features = ['Annual Income (k$)', 'Spending Score (1-100)', 'Age']
    
    # Prepare the data
    X = df[features].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Create results DataFrame
    results_df = df.copy()
    results_df['Cluster'] = clusters
    
    # Create interactive 3D scatter plot with Plotly
    fig = px.scatter_3d(
        results_df, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)', 
        z='Age',
        color='Cluster',
        title='Interactive 3D Customer Segmentation',
        labels={'Cluster': 'Cluster Group'},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Add cluster centers
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_trace = go.Scatter3d(
        x=cluster_centers[:, 0], 
        y=cluster_centers[:, 1], 
        z=cluster_centers[:, 2],
        mode='markers',
        marker=dict(
            color='red', 
            size=10, 
            symbol='cross',
            line=dict(color='red', width=3)
        ),
        name='Centroids'
    )
    fig.add_trace(center_trace)
    
    # Customize layout for better centering and interactivity
    fig.update_layout(
        scene=dict(
            xaxis_title='Annual Income (k$)',
            yaxis_title='Spending Score',
            zaxis_title='Age',
            # Center the camera
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, t=30, b=0),  # Reduce margins
        height=750,
        width=1000,
        # Ensure plot is centered
        autosize=True
    )
    
    return fig, results_df
