import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.io as pio
import json

def run_kmeans_3d_clustering(filepath, n_clusters):
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        
        # Select numeric columns for clustering
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Determine if we can do 3D or need to fall back to 2D
        if len(numeric_columns) < 2:
            raise ValueError(f"Need at least 2 numeric columns for clustering. Found: {numeric_columns}")
        
        # Prepare the data
        if len(numeric_columns) >= 3:
            X = df[numeric_columns[:3]]  # Use first 3 numeric columns for 3D
            is_3d = True
        else:
            X = df[numeric_columns[:2]]  # Use first 2 numeric columns for 2D
            is_3d = False
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Create interactive scatter plot
        colors = [
            'red', 'blue', 'green', 'purple', 'orange', 
            'cyan', 'magenta', 'yellow', 'pink', 'brown'
        ]
        
        # Prepare data for Plotly
        plot_data = []
        for i in range(n_clusters):
            cluster_data = X_scaled[df['Cluster'] == i]
            
            if is_3d:
                plot_data.append({
                    'x': cluster_data[:, 0].tolist(),
                    'y': cluster_data[:, 1].tolist(),
                    'z': cluster_data[:, 2].tolist(),
                    'mode': 'markers',
                    'type': 'scatter3d',
                    'name': f'Cluster {i}',
                    'marker': {
                        'size': 5,
                        'color': colors[i % len(colors)],
                        'opacity': 0.8
                    }
                })
            else:
                plot_data.append({
                    'x': cluster_data[:, 0].tolist(),
                    'y': cluster_data[:, 1].tolist(),
                    'mode': 'markers',
                    'type': 'scatter',
                    'name': f'Cluster {i}',
                    'marker': {
                        'size': 8,
                        'color': colors[i % len(colors)],
                        'opacity': 0.8
                    }
                })
        
        # Centroids
        centroids_scaled = kmeans.cluster_centers_
        for i, centroid in enumerate(centroids_scaled):
            if is_3d:
                plot_data.append({
                    'x': [centroid[0]],
                    'y': [centroid[1]],
                    'z': [centroid[2]],
                    'mode': 'markers',
                    'type': 'scatter3d',
                    'name': f'Centroid {i}',
                    'marker': {
                        'size': 10,
                        'color': colors[i % len(colors)],
                        'symbol': 'diamond',
                        'opacity': 1
                    }
                })
            else:
                plot_data.append({
                    'x': [centroid[0]],
                    'y': [centroid[1]],
                    'mode': 'markers',
                    'type': 'scatter',
                    'name': f'Centroid {i}',
                    'marker': {
                        'size': 12,
                        'color': colors[i % len(colors)],
                        'symbol': 'diamond',
                        'opacity': 1
                    }
                })
        
        # Layout
        if is_3d:
            layout = {
                'title': f'3D K-means Clustering (n_clusters={n_clusters})',
                'scene': {
                    'xaxis': {'title': numeric_columns[0]},
                    'yaxis': {'title': numeric_columns[1]},
                    'zaxis': {'title': numeric_columns[2]}
                }
            }
        else:
            layout = {
                'title': f'2D K-means Clustering (n_clusters={n_clusters})',
                'xaxis': {'title': numeric_columns[0]},
                'yaxis': {'title': numeric_columns[1]}
            }
        
        # Prepare cluster summary
        centroids = scaler.inverse_transform(centroids_scaled)
        cluster_summary = []
        for i in range(n_clusters):
            cluster_data = df[df['Cluster'] == i]
            summary_cols = numeric_columns[:3] if is_3d else numeric_columns[:2]
            cluster_summary.append({
                'Cluster': i,
                'Size': len(cluster_data),
                'Centroids': dict(zip(summary_cols, centroids[i]))
            })
        
        # Convert DataFrame to records for JSON serialization
        df_records = df.to_dict('records')
        
        return {
            'plot_data': plot_data,
            'plot_layout': layout,
            'data': df_records,
            'cluster_summary': cluster_summary,
            'columns': list(df.columns),
            'is_3d': is_3d
        }
    
    except Exception as e:
        print(f"Error in clustering: {str(e)}")
        raise