K-Means Clustering: Manual Implementation
This repository contains a manual implementation of the K-Means clustering algorithm from scratch using Python. The implementation avoids any predefined machine learning libraries, providing a clear understanding of the algorithm's inner workings.

Features

K-Means Algorithm: Iteratively assigns points to clusters and updates centroids.

Inertia Calculation: Computes the sum of squared distances for cluster evaluation.

Elbow Method: Determines the optimal number of clusters by identifying the "elbow point" in the inertia curve.

Silhouette method: It measures how well each point fits within its assigned K-Means cluster by comparing intra-cluster cohesion to inter-cluster separation, with scores ranging from -1 (misclassified) to 1 (well-clustered).

The fit_transform method combines the steps of fitting the K-means model (finding centroids) and transforming the data into distances from those centroids in one go.
