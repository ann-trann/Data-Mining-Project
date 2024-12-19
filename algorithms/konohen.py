import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd

class KohonenSOM:
    def __init__(self, map_size=(5,5), input_dim=2, learning_rate=0.1, radius=2.0):
        self.map_size = map_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.initial_radius = radius
        
    def initialize_weights(self, data):
        # Khởi tạo ngẫu nhiên trọng số cho các neuron dựa trên phạm vi của dữ liệu
        self.weights = np.random.uniform(
            low=np.min(data), 
            high=np.max(data), 
            size=(self.map_size[0], self.map_size[1], self.input_dim)
        )
        
    def find_bmu(self, sample):
        """Tìm Best Matching Unit (BMU)"""
        distances = np.sum((self.weights - sample) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def get_neighborhood(self, center, sigma):
        """Tính toán hàm láng giềng"""
        y_grid, x_grid = np.ogrid[0:self.map_size[0], 0:self.map_size[1]]
        distance_matrix = np.sqrt((x_grid - center[1])**2 + (y_grid - center[0])**2)
        neighborhood = np.exp(-(distance_matrix ** 2) / (2 * sigma ** 2))
        return neighborhood[..., np.newaxis]
    
    def train(self, data, epochs=100):
        """Huấn luyện mạng SOM"""
        self.initialize_weights(data)  # Khởi tạo trọng số dựa trên dữ liệu
        
        for epoch in range(epochs):
            current_lr = self.initial_learning_rate * np.exp(-epoch/epochs)
            current_radius = self.initial_radius * np.exp(-epoch/epochs)
            
            np.random.shuffle(data)
            
            for sample in data:
                bmu_idx = self.find_bmu(sample)
                neighborhood = self.get_neighborhood(bmu_idx, current_radius)
                self.weights += neighborhood * current_lr * (sample - self.weights)

    def predict(self, data):
        """Dự đoán cluster cho dữ liệu mới"""
        predictions = []
        for sample in data:
            bmu_idx = self.find_bmu(sample)
            predictions.append(bmu_idx)
        return predictions

def run_konohen_clustering(data_path, map_size=(5,5)):
    # Đọc dữ liệu
    data = pd.read_csv(data_path).values
    
    # Khởi tạo và huấn luyện mô hình
    som = KohonenSOM(map_size=map_size, input_dim=2)
    som.train(data, epochs=200)
    
    # Dự đoán clusters
    clusters = som.predict(data)
    
    # Tạo plot
    plt.figure(figsize=(10, 10))
    
    # Vẽ dữ liệu gốc
    plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o', label='Input data')
    
    # Vẽ các neuron
    weights = som.weights.reshape(-1, 2)
    plt.scatter(weights[:, 0], weights[:, 1], c='red', marker='x', label='Neurons')
    
    # Vẽ lưới kết nối
    for i in range(som.map_size[0]):
        for j in range(som.map_size[1]):
            if j < som.map_size[1]-1:
                plt.plot([som.weights[i,j,0], som.weights[i,j+1,0]],
                        [som.weights[i,j,1], som.weights[i,j+1,1]], 'gray', alpha=0.5)
            if i < som.map_size[0]-1:
                plt.plot([som.weights[i,j,0], som.weights[i+1,j,0]],
                        [som.weights[i,j,1], som.weights[i+1,j,1]], 'gray', alpha=0.5)
    
    plt.title('Kohonen Self-Organizing Map')
    plt.legend()
    plt.grid(True)
    
    # Chuyển plot thành base64 string
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Tạo kết quả phân cụm
    cluster_results = []
    for i, (point, cluster) in enumerate(zip(data, clusters)):
        cluster_results.append({
            'point_id': f'x{i+1}',
            'coordinates': f'[{point[0]:.1f}, {point[1]:.1f}]',
            'cluster': f'({cluster[0]}, {cluster[1]})'
        })
    
    return {
        'plot': plot_base64,
        'cluster_results': cluster_results
    }