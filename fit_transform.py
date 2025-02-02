import random                                                                   # Import the random module to randomly select initial centroids
import math                                                                     # Import math module to calculate Euclidean distance

class SimpleKMeans:
    def __init__(self, n_clusters=2):
        """Initialize the number of clusters and centroids."""
        self.n_clusters = n_clusters                                            # Store the number of clusters to create
        self.centroids = None                                                   # This will hold the centroids of the clusters after fitting

    def fit(self, X):
        """Find cluster centroids using simple k-means clustering."""
        random.seed(42)                                                         # Set a fixed seed for random number generation (to get the same result each time) 
        self.centroids = random.sample(X, self.n_clusters)                      # Randomly choose `n_clusters` data points from X as initial centroids
        for _ in range(10):                                                     # Run the clustering algorithm for 10 iterations to refine centroids
            labels = []                                                         # Create an empty list to store the cluster labels for each data point        
            for point in X:                                                     # Loop through each data point in X
                distances = []                                                  # List to store distances from the point to each centroid  
                for centroid in self.centroids:                                 # Loop through each centroid and calculate the distance to the current point            
                    distance = self.euclidean_distance(point, centroid)         # Calculate the Euclidean distance between the point and the centroid
                    distances.append(distance)                                  # Store the distance in the list           
                closest_centroid = min(distances)                               # Find the closest centroid by picking the minimum distance             
                closest_centroid_index = distances.index(closest_centroid)      # Find the index of the closest centroid             
                labels.append(closest_centroid_index)                           # Append the index of the closest centroid to the labels list
                                                                                # Now that we know which points belong to which clusters, we update centroids
            new_centroids = []                                                  # List to store the new centroids
            for i in range(self.n_clusters):                                    # Loop through each cluster (0 to n_clusters-1)
                cluster_points = []                                             # List to store points belonging to the current cluster            
                for j in range(len(X)):                                         # Loop through all points and check which cluster they belong to
                    if labels[j] == i:                                          # If the point belongs to the current cluster
                        cluster_points.append(X[j])                             # Add the point to the cluster                
                if cluster_points:                                              # If there are points in this cluster, compute the new centroid
                    new_centroid = []                                           # List to store the new centroid of the current cluster                                                                               # For each dimension (e.g., x or y), calculate the mean of the points
                    for dim in zip(*cluster_points):                            # zip(*cluster_points) transposes the points
                        new_centroid.append(sum(dim) / len(cluster_points))     # Average each dimension
                    new_centroids.append(new_centroid)                          # Append the new centroid to the list of centroids
                else:                 
                    new_centroids.append(self.centroids[i])                     # If no points were assigned to this cluster (empty cluster), keep the old centroid   
            self.centroids = new_centroids                                      # Update the centroids with the new values after all clusters have been updated

    def transform(self, X):
        """Convert X into cluster-distance space (distance to each centroid)."""
        distances = []                                                          # List to store the distances of each point to each centroid
       
        for point in X:                                                         # Loop through each data point in X
            point_distances = []                                                # List to store the distances of the current point to each centroid   
            for centroid in self.centroids:                                     # Loop through each centroid and calculate the distance to the current point          
                distance = self.euclidean_distance(point, centroid)             # Calculate the Euclidean distance between the point and the centroid               
                point_distances.append(distance)                                # Append the distance to the list of distances for this point           
            distances.append(point_distances)                                   # Append the list of distances for this point to the final list
        return distances                                                        # Return the distances of all points to all centroids

    def fit_transform(self, X):
        """Fit the model (find clusters) and transform X (convert to distances)."""
        self.fit(X)                                                             # First, find the cluster centroids (fit the model)
        return self.transform(X)                                                # Then, return the distance of each point to each centroid

    def euclidean_distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        squared_diff = []                                                       # List to store squared differences between corresponding dimensions    
        for a, b in zip(point1, point2):                                        # Loop through each dimension of the two points
            diff = a - b                                                        # Calculate the difference in the current dimension
            squared_diff.append(diff ** 2)                                      # Square the difference and store it        
        return math.sqrt(sum(squared_diff))                                     # Return the square root of the sum of squared differences (Euclidean distance formula)

X = [[1, 2], [2, 3], [3, 4], [8, 9], [9, 10]]                                   # List of sample data points
kmeans = SimpleKMeans(n_clusters=2)                                             # Create a k-means object with 2 clusters
X_transformed = kmeans.fit_transform(X)                                         # Fit the model and transform X into distance space
print(X_transformed)                                                            # Print the transformed data (distances to centroids)