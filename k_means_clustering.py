import random
import math

def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    """
    distance = 0
    for p1, p2 in zip(point1, point2):
        distance += (p1 - p2) ** 2
    return math.sqrt(distance)

def calculate_mean(points):
    """
    Calculate the mean of a list of points.
    """
    if len(points) == 0:
        return []  # Handle empty clusters explicitly
    num_points = len(points)
    mean = []
    for dim in zip(*points):
        mean.append(sum(dim) / num_points)
    return mean

def assign_clusters(data, centroids):
    """
    Assign each data point to the nearest centroid.
    """
    clusters = []
    for _ in centroids:
        clusters.append([])  # Initialize an empty list for each cluster

    for point in data:
        distances = []
        for centroid in centroids:
            distances.append(calculate_distance(point, centroid))
        min_distance_index = 0
        for i in range(1, len(distances)):
            if distances[i] < distances[min_distance_index]:
                min_distance_index = i
        clusters[min_distance_index].append(point)
    return clusters

def kmeans_clustering(data, k, max_iters=100, tol=1e-4):
    # Randomly initialize k centroids from the data points
    centroids = random.sample(data, k)
    
    for iteration in range(max_iters):
        # Step 1: Assign clusters
        clusters = assign_clusters(data, centroids)
        
        # Step 2: Update centroids
        new_centroids = []
        for cluster in clusters:
            if len(cluster) > 0:
                new_centroids.append(calculate_mean(cluster))
            else:
                new_centroids.append([])  # Handle empty cluster explicitly
        
        # Check for convergence
        centroid_shift = 0
        for old, new in zip(centroids, new_centroids):
            centroid_shift += calculate_distance(old, new)
        
        if centroid_shift < tol:
            break
        
        centroids = new_centroids
    
    # Return cluster assignments and centroids
    return {
        "clusters": clusters,
        "centroids": centroids
    }

# Example usage
if __name__ == "__main__":
    data = [
        [1, 2], [2, 1], [3, 2], [8, 8], [9, 10], [10, 9], 
        [1, 0], [2, 3], [10, 8], [8, 9]
    ]
    k = 2  # Number of clusters

    result = kmeans_clustering(data, k)
    
    print("Centroids:")
    for centroid in result["centroids"]:
        print(centroid)
    
    print("\nClusters:")
    for i, cluster in enumerate(result["clusters"]):
        print(f"Cluster {i}: {cluster}")
