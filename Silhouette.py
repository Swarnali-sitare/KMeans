import random

# Function to calculate Euclidean distance between two points
def euclide_dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

# Function to compute the silhouette score
def silhouette_score(X, clusters, k):
    scores = []                                                               # List to store silhouette scores for each point

    for i, x in enumerate(X):                                                 # Iterate over each data point
        same_cluster = []                                                     # List to store points in the same cluster

        for j in range(len(X)):                                               # Iterate over all points
            if clusters[j] == clusters[i]:                                    # Check if the point belongs to the same cluster
                same_cluster.append(X[j])                                     # Add to same cluster list

        if len(same_cluster) > 1:                                             # Compute average intra-cluster distance (a) if there are multiple points
            a_values = []                                                     # Store distances to other points in the same cluster

            for p in same_cluster:
                if x != p:                                                    # Avoid calculating distance to itself
                    a_values.append(euclide_dist(x, p))

            a = sum(a_values) / (len(same_cluster) - 1)                       # Compute mean intra-cluster distance
        else:
            a = 0                                                             # If only one point in cluster, a = 0

        b_values = []                                                         # List to store inter-cluster distances

        for j in range(k):                                                    # Iterate over all clusters
            if j != clusters[i]:                                              # Skip the current cluster
                other_cluster = []                                            # List to store points in another cluster

                for m in range(len(X)):                                       # Iterate over all points
                    if clusters[m] == j:                                      # Check if point belongs to another cluster
                        other_cluster.append(X[m])                            # Add point to other cluster list

                if len(other_cluster) > 0:                                    # Compute average distance to points in another cluster
                    b_values.append(sum(euclide_dist(x, p) for p in other_cluster) / len(other_cluster))

        b = min(b_values)                                                     # Find the smallest inter-cluster distance

        scores.append((b - a) / max(a, b))                                    # Compute silhouette score for the point

    return sum(scores) / len(scores)                                          # Compute overall silhouette score


random.seed(42)                                                               # Set seed for reproducibility
X = []                                                                        # List to store random points

for _ in range(100):                                                          # Generate 100 random points
    X.append((random.uniform(0, 10), random.uniform(0, 10)))                  # Each point is a tuple (x, y)

clusters = []                                                                 # List to store cluster assignments

for _ in range(len(X)):
    clusters.append(random.randint(0, 2))                                     # Assign each point randomly to one of three clusters

k = len(set(clusters))                                                        # Find the number of unique clusters
score = silhouette_score(X, clusters, k)                                      # Compute silhouette score
print(f'Silhouette Score: {score:.4f}')                                       # Print result