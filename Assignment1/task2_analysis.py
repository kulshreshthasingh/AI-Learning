import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from knn_classifier import KNNClassifier

# Set random seed
np.random.seed(42)

def load_cifar_batch(file):
    """Load a single batch of CIFAR-10 data."""
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(data_dir):
    """
    Load all valid batches from the CIFAR-10 directory.
    Returns training and test sets.
    """
    # Look for the extracted folder 'cifar-10-batches-py' or just the directory provided
    if os.path.exists(os.path.join(data_dir, 'cifar-10-batches-py')):
        data_dir = os.path.join(data_dir, 'cifar-10-batches-py')
        
    print(f"Looking for data in: {data_dir}")
    
    xs = []
    ys = []
    
    # Load training batches
    for i in range(1, 6):
        f = os.path.join(data_dir, f'data_batch_{i}')
        if not os.path.exists(f):
            print(f"Warning: {f} not found. Skipping.")
            continue
            
        print(f"Loading {f}...")
        try:
            batch = load_cifar_batch(f)
            X = batch[b'data']
            y = batch[b'labels']
            xs.append(X)
            ys.append(y)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not xs:
        print("No training data found.")
        return None, None, None, None

    X_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    
    # Load test batch
    f_test = os.path.join(data_dir, 'test_batch')
    if os.path.exists(f_test):
        print(f"Loading {f_test}...")
        batch = load_cifar_batch(f_test)
        X_test = batch[b'data']
        y_test = np.array(batch[b'labels'])
    else:
        print("Warning: Test batch not found.")
        X_test = np.empty((0, 3072))
        y_test = np.empty((0,))
        
    return X_train, y_train, X_test, y_test

def normalize_data(X_train, X_test):
    """
    Normalize data using Min-Max scaling (0-1) or Mean Subtraction.
    Here we use simple division by 255.0 to map to [0, 1].
    """
    return X_train.astype('float32') / 255.0, X_test.astype('float32') / 255.0

def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall (Macro-averaged for multi-class).
    """
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    classes = np.unique(y_true)
    precisions = []
    recalls = []
    
    # Simple macro average calculation
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precisions.append(prec)
        recalls.append(rec)
        
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    
    return accuracy, precision_macro, recall_macro

def run_experiments(subset_size=500, test_subset_size=100):
    base_dir = r"c:/Users/Dell/Desktop/AI_LEARNING/Assignment1"
    
    X_train, y_train, X_test, y_test = load_cifar10_data(base_dir)
    
    if X_train is None:
        print("Required data not found. Please ensure 'cifar-10-batches-py' is in the Assignment1 folder.")
        # Create dummy data for code verification if user chose Option 3
        print("Generating DUMMY data for verification...")
        X_train = np.random.rand(100, 3072) * 255
        y_train = np.random.randint(0, 10, 100)
        X_test = np.random.rand(20, 3072) * 255
        y_test = np.random.randint(0, 10, 20)
        subset_size = 100
        test_subset_size = 20
    
    print(f"Total Training Data: {X_train.shape}")
    print(f"Total Test Data: {X_test.shape}")
    
    # Use subset for faster experimentation
    if subset_size:
        print(f"Using subset of {subset_size} training samples and {test_subset_size} test samples.")
        X_train = X_train[:subset_size]
        y_train = y_train[:subset_size]
        X_test = X_test[:test_subset_size]
        y_test = y_test[:test_subset_size]
        
    # Preprocessing
    X_train, X_test = normalize_data(X_train, X_test)
    
    k_values = [3, 4, 9, 20, 47]
    distances = ['euclidean', 'manhattan', 'cosine'] # Removed hamming/minkowski to save time for this verify run
    # distances = ['euclidean', 'manhattan', 'minkowski', 'cosine', 'hamming'] 
    
    results = []
    
    print(f"\n{'K':<5} {'Metric':<15} {'Accuracy':<10}")
    print("-" * 35)
    
    for metric in distances:
        for k in k_values:
            knn = KNNClassifier(k=k, distance_metric=metric)
            knn.fit(X_train, y_train)
            predictions = knn.predict(X_test)
            
            acc, prec, rec = calculate_metrics(y_test, predictions)
            
            results.append({
                'k': k,
                'metric': metric,
                'accuracy': acc,
                'precision': prec,
                'recall': rec
            })
            
            print(f"{k:<5} {metric:<15} {acc:.4f}")
            
    plot_results(results, k_values, distances)

def plot_results(results, k_values, distances):
    plt.figure(figsize=(12, 8))
    
    for metric in distances:
        accuracies = [res['accuracy'] for res in results if res['metric'] == metric]
        if accuracies:
            plt.plot(k_values, accuracies, marker='o', label=metric.capitalize())
        
    plt.title('CIFAR-10 Classification: Accuracy vs K')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    output_path = r'c:/Users/Dell/Desktop/AI_LEARNING/Assignment1/task2_results.png'
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")

if __name__ == "__main__":
    # Run with a small subset initially to verify; user can increase this later
    run_experiments(subset_size=1000, test_subset_size=100)
