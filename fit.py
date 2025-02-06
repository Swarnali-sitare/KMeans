import random

class KMeans:
    def __init__(self, n_clusters, max_iters=100):
        self.k = n_clusters
        self.max_iters = max_iters
        self.centroids = []

    def initialize_centroids(self, X):
        """Randomly selects k points from X as initial centroids"""
        self.centroids = random.sample(X, self.k)

    def assign_clusters(self, X):
        """Assigns each point to the nearest centroid"""
        clusters = [[] for _ in range(self.k)]
        for point in X:
            distances = [self.euclidean_distance(point, centroid) for centroid in self.centroids]
            closest_index = distances.index(min(distances))
            clusters[closest_index].append(point)
        return clusters

    def update_centroids(self, clusters):
        """Updates centroids by computing the mean of assigned points"""
        new_centroids = []
        for cluster in clusters:
            if cluster:
                new_centroids.append([sum(dim) / len(cluster) for dim in zip(*cluster)])
            else:
                new_centroids.append(random.choice(self.centroids))
        return new_centroids

    def euclidean_distance(self, p1, p2):
        """Computes Euclidean distance between two points"""
        return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5

    def fit(self, X):
        """Fits the K-Means model to the dataset"""
        self.initialize_centroids(X)
        
        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X)
            new_centroids = self.update_centroids(clusters)

            if new_centroids == self.centroids:
                break
            
            self.centroids = new_centroids
