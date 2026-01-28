import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean', p=3):
        """
        Initialize the KNN Classifier.
        
        Args:
            k (int): Number of neighbors to consider.
            distance_metric (str): The distance metric to use. 
                                   Options: 'euclidean', 'manhattan', 'minkowski', 'cosine', 'hamming'.
            p (int): The power parameter for Minkowski distance. Default is 3.
        """
        self.k = k
        self.distance_metric = distance_metric.lower()
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        
        Args:
            X (np.ndarray): Training features.
            y (np.ndarray): Training labels.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        """
        Predict labels for the given test data.
        
        Args:
            X (np.ndarray): Test features.
            
        Returns:
            np.ndarray: Predicted labels.
        """
        X = np.array(X)
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Predict label for a single data point.
        """
        # Compute distances between x and all examples in the training set
        distances = self._compute_distances(x)
        
        # Get the indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Get the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def _compute_distances(self, x):
        """
        Compute distances based on the selected metric.
        """
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1)
        
        elif self.distance_metric == 'minkowski':
            return np.power(np.sum(np.power(np.abs(self.X_train - x), self.p), axis=1), 1 / self.p)
        
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - Cosine Similarity
            # Similarity = (A . B) / (||A|| * ||B||)
            
            dot_product = np.dot(self.X_train, x)
            norm_train = np.linalg.norm(self.X_train, axis=1)
            norm_x = np.linalg.norm(x)
            
            # Avoid division by zero
            if norm_x == 0:
                return np.ones(len(self.X_train)) # Max distance if zero vector
                
            similarity = dot_product / (norm_train * norm_x + 1e-10) # Add epsilon for stability
            return 1 - similarity
            
        elif self.distance_metric == 'hamming':
            # Hamming distance: proportion of differing components
            # For continuous data, this checks strict inequality
            # Creates a boolean array where elements differ, then sums them up
            return np.sum(self.X_train != x, axis=1) / self.X_train.shape[1]
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
