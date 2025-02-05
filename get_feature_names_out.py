# Define the class first
class KMeansTransformer:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.class_name = "kmeans"
    
    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            if len(input_features) != self.n_clusters:
                raise ValueError("Number of input features does not match number of clusters.")
        
        feature_names = []
        for i in range(self.n_clusters):
            feature_names.append(self.class_name + str(i))
        
        return feature_names

# Create an instance
kmeans_transformer = KMeansTransformer(n_clusters=3)

# Call the method
print(kmeans_transformer.get_feature_names_out())