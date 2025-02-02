import numpy as np

def calculate_inertia(data, centroids, labels):
    inertia = 0
    for i in range(len(data)):
        # Squared distance: (x1 - x2)^2 + (y1 - y2)^2 + ...
        distance_squared = sum((data[i][j] - centroids[labels[i]][j]) ** 2 for j in range(data.shape[1]))
        inertia += distance_squared
    return inertia

def find_optimal_clusters(inertia_values):
    n = len(inertia_values)
    if n < 3:
        raise ValueError("At least three values are required to compute the elbow point.")
    
    # Start and end points of the line
    x_start, y_start = 1, inertia_values[0]
    x_end, y_end = n, inertia_values[-1]
    
    # Calculate distances from each point to the line
    distances = []
    for k in range(1, n + 1):
        # Current point
        x_point, y_point = k, inertia_values[k - 1]
        
        # Area of the triangle formed by the line and the point
        numerator = abs((y_end - y_start) * x_point - (x_end - x_start) * y_point + x_end * y_start - y_end * x_start)
        denominator = ((y_end - y_start) ** 2 + (x_end - x_start) ** 2) ** 0.5
        distance = numerator / denominator
        
        distances.append(distance)
    
    # The optimal k corresponds to the maximum distance
    optimal_k = distances.index(max(distances)) + 1
    return optimal_k

# Example Usage
if __name__ == "__main__":
    # Example dataset (5 data points, 2 features)
    data = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4]])
    
    # Example centroids (for k=2)
    centroids = np.array([[1, 2], [10, 4]])
    
    # Example labels (which cluster each point belongs to)
    labels = np.array([0, 0, 0, 1, 1])  # Cluster assignments for data points
    
    # Calculate inertia
    inertia = calculate_inertia(data, centroids, labels)
    print("Inertia:", inertia)
    
    # Example list of inertia values for k = 1 to 6
    inertia_values = [1000, 800, 500, 300, 250, 200]
    
    # Find the optimal number of clusters
    optimal_k = find_optimal_clusters(inertia_values)
    print("Optimal number of clusters (k):", optimal_k)
