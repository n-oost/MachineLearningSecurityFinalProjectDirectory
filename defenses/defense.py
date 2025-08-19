# Defense Mechanisms Against Data Poisoning Attacks
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocessing import load_dataset, preprocess_data, split_dataset

class LabelCleaningDefense:
    """
    Defense mechanism to detect and correct poisoned labels.
    This defense is effective against label flipping attacks.
    """
    
    def __init__(self, method='knn', params=None):
        """
        Initialize the defense.
        
        Parameters:
        -----------
        method : str, default='knn'
            Method to use for label cleaning ('knn', 'kmeans')
        params : dict, optional
            Parameters for the defense method
        """
        self.method = method
        self.params = params if params else {}
        self.name = f"Label Cleaning Defense ({method})"
        
    def detect_and_clean(self, X, y):
        """
        Detect potentially poisoned samples and clean their labels.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training features
        y : array-like, shape (n_samples,)
            Training labels (expected to be class indices, not one-hot encoded)
            
        Returns:
        --------
        X_cleaned, y_cleaned : Cleaned data
        """
        # Handle one-hot encoded labels if necessary
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_idx = np.argmax(y, axis=1)
            is_one_hot = True
            num_classes = y.shape[1]
        else:
            y_idx = y.copy()
            is_one_hot = False
            num_classes = len(np.unique(y_idx))
        
        # Create copies to avoid modifying originals
        X_cleaned = X.copy()
        y_cleaned = y.copy() if is_one_hot else y_idx.copy()
        
        if self.method == 'knn':
            cleaned_labels = self._knn_cleaning(X, y_idx)
        elif self.method == 'kmeans':
            cleaned_labels = self._kmeans_cleaning(X, y_idx, num_classes)
        else:
            raise ValueError(f"Unknown method: {self.method}. Available methods: 'knn', 'kmeans'")
        
        # Find samples with different labels after cleaning
        changed_indices = np.where(y_idx != cleaned_labels)[0]
        num_changed = len(changed_indices)
        
        # Update labels with cleaned versions
        if is_one_hot:
            for idx in changed_indices:
                y_cleaned[idx] = np.zeros(num_classes)
                y_cleaned[idx, cleaned_labels[idx]] = 1
        else:
            y_cleaned[changed_indices] = cleaned_labels[changed_indices]
        
        print(f"Applied {self.name}: Corrected {num_changed} labels ({(num_changed/len(y_idx))*100:.1f}% of data)")
        return X_cleaned, y_cleaned
    
    def _knn_cleaning(self, X, y):
        """
        Use KNN to detect and clean mislabeled samples.
        
        The idea is to train a KNN classifier on the whole dataset, then predict
        labels for each sample using its neighbors. If the predicted label differs
        from the original, it might be poisoned.
        """
        n_neighbors = self.params.get('n_neighbors', 5)
        confidence_threshold = self.params.get('confidence_threshold', 0.5)
        
        # Initialize KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
        
        # Fit on the entire dataset
        knn.fit(X, y)
        
        # Get predicted probabilities for each sample
        y_proba = knn.predict_proba(X)
        
        # Get predicted labels
        y_pred = knn.predict(X)
        
        # Find maximum probability for each prediction
        max_proba = np.max(y_proba, axis=1)
        
        # Create cleaned labels: keep original if confidence is low
        cleaned_labels = np.copy(y)
        high_confidence = max_proba >= confidence_threshold
        
        # Only change labels that have high confidence and differ from original
        to_change = high_confidence & (y_pred != y)
        cleaned_labels[to_change] = y_pred[to_change]
        
        return cleaned_labels
    
    def _kmeans_cleaning(self, X, y, num_classes):
        """
        Use K-means clustering to detect and clean mislabeled samples.
        
        The idea is to cluster the data into k clusters (where k is the number of classes),
        then check if samples are in the 'right' cluster based on majority voting.
        """
        # Initialize KMeans with the number of classes
        kmeans = KMeans(n_clusters=num_classes, random_state=42)
        
        # Fit on the data
        kmeans.fit(X)
        
        # Get cluster assignments
        clusters = kmeans.labels_
        
        # Create cleaned labels (starting with original)
        cleaned_labels = np.copy(y)
        
        # For each cluster, find the majority class
        for cluster_id in range(num_classes):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
                
            # Get the labels of samples in this cluster
            cluster_labels = y[cluster_indices]
            
            # Find the majority class
            unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(label_counts)]
            
            # Find samples in this cluster with different labels
            mislabeled_indices = cluster_indices[cluster_labels != majority_label]
            
            # Correct the labels
            cleaned_labels[mislabeled_indices] = majority_label
        
        return cleaned_labels
    
    def evaluate_defense(self, X_test, y_test, model_original, model_defended):
        """
        Evaluate the effectiveness of the defense.
        
        Parameters:
        -----------
        X_test, y_test : array-like
            Clean test data
        model_original : trained model
            Model trained on poisoned data without defense
        model_defended : trained model
            Model trained on data after applying defense
            
        Returns:
        --------
        results : dict
            Dictionary of defense metrics
        """
        # Evaluate original (poisoned) model
        original_acc = model_original.evaluate(X_test, y_test, verbose=0)[1]
        
        # Evaluate defended model
        defended_acc = model_defended.evaluate(X_test, y_test, verbose=0)[1]
        
        # Calculate defense effectiveness
        improvement = defended_acc - original_acc
        
        return {
            'poisoned_accuracy': original_acc,
            'defended_accuracy': defended_acc,
            'improvement': improvement,
            'defense_name': self.name
        }


def defend_model(model, X_train, y_train, defense_method='knn', defense_params=None):
    """
    Apply defense strategies to protect against data poisoning attacks.
    
    Parameters:
    -----------
    model : model object
        The model to defend
    X_train : array-like
        Training features that might be poisoned
    y_train : array-like
        Training labels that might be poisoned
    defense_method : str, default='knn'
        Method to use for label cleaning ('knn', 'kmeans')
    defense_params : dict, optional
        Parameters for the defense method
        
    Returns:
    --------
    model : defended model
    """
    # Create a defense instance
    defense = LabelCleaningDefense(method=defense_method, params=defense_params)
    
    # Apply the defense to clean the data
    X_cleaned, y_cleaned = defense.detect_and_clean(X_train, y_train)
    
    # Train the model on the cleaned data
    print("Training model on cleaned data...")
    model.fit(X_cleaned, y_cleaned, epochs=10, batch_size=128, verbose=0)
    
    return model


if __name__ == "__main__":
    # Import attack for demonstration
    from scripts.data_poisoning import LabelFlippingAttack
    from scripts.train import build_model
    
    # Load and preprocess the data
    print("Loading and preprocessing MNIST dataset...")
    X, y = load_dataset()
    X = preprocess_data(X)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3, random_state=42)
    
    # Apply a label flipping attack
    attack = LabelFlippingAttack(flip_ratio=0.2, target_label=9)
    X_poisoned, y_poisoned = attack.poison(X_train, y_train)
    
    # Apply defense
    defense = LabelCleaningDefense(method='knn', params={'n_neighbors': 7, 'confidence_threshold': 0.7})
    X_defended, y_defended = defense.detect_and_clean(X_poisoned, y_poisoned)
    
    # Convert labels to indices for comparison
    if len(y_train.shape) > 1:  # One-hot encoded
        y_train_idx = np.argmax(y_train, axis=1)
        y_poisoned_idx = np.argmax(y_poisoned, axis=1)
        y_defended_idx = np.argmax(y_defended, axis=1)
    else:
        y_train_idx = y_train
        y_poisoned_idx = y_poisoned
        y_defended_idx = y_defended
    
    # Calculate how many poisoned labels were corrected
    num_poisoned = np.sum(y_train_idx != y_poisoned_idx)
    correct_fixes = np.sum((y_train_idx != y_poisoned_idx) & (y_train_idx == y_defended_idx))
    incorrect_fixes = np.sum((y_train_idx == y_poisoned_idx) & (y_train_idx != y_defended_idx))
    
    # Print results
    print("\nDefense results:")
    print(f"Total poisoned labels: {num_poisoned}")
    print(f"Correctly fixed poisoned labels: {correct_fixes} ({(correct_fixes/num_poisoned)*100:.1f}%)")
    print(f"Incorrectly changed clean labels: {incorrect_fixes}")
    
    print("\nNext steps:")
    print("1. Train a model on the defended data")
    print("2. Compare with models trained on clean and poisoned data")
    print("3. Evaluate the defense effectiveness")
