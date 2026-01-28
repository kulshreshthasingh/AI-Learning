import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
# import seaborn as sns # Removed due to missing dependency
from knn_classifier import KNNClassifier
import os

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(filepath):
    """
    Load data, clean it, and perform preprocessing.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Drop 'id' column as it's not a feature
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
        
    # Drop rows with missing values (if any - simple handling)
    # df = df.dropna() # Inspection of file 1 showed no nans but good practice generally.
    # The last column in the viewed file appeared empty/weird "Unnamed: 32" often happens in CSVs with trailing commas
    if df.columns[-1].startswith('Unnamed'):
        df = df.iloc[:, :-1]

    # Encode target variable 'diagnosis' (M=1, B=0)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop('diagnosis', axis=1).values
    y = df['diagnosis'].values
    
    # Normalize features (Z-score standardization)
    # Formula: (x - mean) / std
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    X_normalized = (X - mean) / std
    
    return X_normalized, y, df.drop('diagnosis', axis=1).columns

def train_test_split_manual(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    """
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and confusion matrix.
    """
    # Confusion Matrix
    # TP: True (1) and Pred (1)
    # TN: True (0) and Pred (0)
    # FP: True (0) and Pred (1)
    # FN: True (1) and Pred (0)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    confusion_matrix = np.array([[tn, fp], [fn, tp]])
    
    return accuracy, precision, recall, confusion_matrix

def run_experiments():
    # File path - adjust if necessary
    data_path = r'c:/Users/Dell/Desktop/AI_LEARNING/data (1).csv'
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        return

    X, y, feature_names = load_and_preprocess_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y, test_size=0.2)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    k_values = [3, 4, 9, 20, 47]
    distances = ['euclidean', 'manhattan', 'minkowski', 'cosine', 'hamming']
    
    results = []
    
    best_accuracy = 0
    best_config = {}
    
    print("\nStarting Experiments...")
    print(f"{'K':<5} {'Metric':<15} {'Accuracy':<10}")
    print("-" * 35)
    
    for metric in distances:
        metric_accuracies = []
        for k in k_values:
            knn = KNNClassifier(k=k, distance_metric=metric)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            
            accuracy, precision, recall, cm = calculate_metrics(y_test, predictions)
            
            results.append({
                'k': k,
                'metric': metric,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'confusion_matrix': cm
            })
            
            print(f"{k:<5} {metric:<15} {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_config = {
                    'k': k,
                    'metric': metric,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'confusion_matrix': cm
                }
        
    print("\nExperiments Completed.")
    print("\n" + "="*30)
    print("BEST CONFIGURATION FOUND")
    print("="*30)
    print(f"K: {best_config['k']}")
    print(f"Distance Metric: {best_config['metric']}")
    print(f"Accuracy: {best_config['accuracy']:.4f}")
    print(f"Precision: {best_config['precision']:.4f}")
    print(f"Recall: {best_config['recall']:.4f}")
    print("Confusion Matrix:")
    print(best_config['confusion_matrix'])
    
    # Plotting
    plot_results(results, k_values, distances)

def plot_results(results, k_values, distances):
    plt.figure(figsize=(12, 8))
    
    for metric in distances:
        accuracies = [res['accuracy'] for res in results if res['metric'] == metric]
        plt.plot(k_values, accuracies, marker='o', label=metric.capitalize())
        
    plt.title('KNN Accuracy vs K for Different Distance Metrics')
    plt.xlabel('K (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(k_values)
    
    output_path = r'c:/Users/Dell/Desktop/AI_LEARNING/Assignment1/task1_results.png'
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    run_experiments()
