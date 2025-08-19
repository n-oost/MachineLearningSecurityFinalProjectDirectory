# Dataset selection and preprocessing pipeline
# Replace 'mnist' with your dataset if needed
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

"""
Select a suitable dataset for your project. It could be a standard dataset like MNIST, CIFAR-10, 
or one relevant to your domain. 

Implement preprocessing pipeline: Normalization, train-test splits (70-30)
"""

# Load MNIST dataset using TensorFlow Keras
def load_dataset():
    from tensorflow.keras.datasets import mnist # type: ignore
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Flatten images to vectors
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    return X, y

def preprocess_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def split_dataset(X, y, test_size, val_size, random_state=42):
    """
    Split dataset into train, validation, and test sets.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix to be split
    y : array-like, shape (n_samples,)
        Target vector to be split
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split
    val_size : float, default=0.2
        Proportion of the dataset to include in the validation split
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    X_train, y_train, X_val, y_val, X_test, y_test : split datasets
    """
    # First split: train+validation vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    
    
    # Determine validation ratio relative to train+val set
    val_ratio = val_size / (1.0 - test_size)
    # Second split: train vs validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state)
    save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_preprocessed_data():
    """
    Load preprocessed data if it exists.
    
    Returns:
    --------
    data : tuple or None
        (X_train, y_train, X_val, y_val, X_test, y_test) if files exist, None otherwise
    """
    try:
        # Check if all required files exist
        data_files = ['data/X_train.npy', 'data/X_val.npy', 'data/X_test.npy', 
                     'data/y_train.npy', 'data/y_val.npy', 'data/y_test.npy']
        if all(os.path.exists(f) for f in data_files):
            X_train = np.load('data/X_train.npy')
            X_val = np.load('data/X_val.npy')
            X_test = np.load('data/X_test.npy')
            y_train = np.load('data/y_train.npy')
            y_val = np.load('data/y_val.npy')
            y_test = np.load('data/y_test.npy')
            return X_train, y_train, X_val, y_val, X_test, y_test
        return None
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        return None

def save_preprocessed_data(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Save preprocessed data to .npy files.
    
    Parameters:
    -----------
    X_train, y_train, X_test, y_test : numpy arrays
        Preprocessed and split datasets to save
    """
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_val.npy', X_val)
    np.save('data/y_val.npy', y_val)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    print("Preprocessed data saved to disk.")
    
if __name__ == "__main__":
    """
    # Check if preprocessed data already exists
    print("Checking for existing preprocessed data...")
    data = load_preprocessed_data()
    
    if data is not None:
        X_train, X_test, y_train, y_test = data
        print("Preprocessed data loaded from disk.")
        print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    else:
        print("No preprocessed data found. Creating new data...")
        
        # Load dataset
        print("Loading MNIST dataset...")
        X, y = load_dataset()
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Apply normalization
        print("Normalizing data...")
        X = preprocess_data(X)
        print(f"Data normalized to range [{X.min():.1f}, {X.max():.1f}]")
        
        # Split into train (70%) and test (30%) sets
        print("Splitting into train and test sets (70-30)...")
        X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3, random_state=42)
        print(f"Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
        
        # Print sample statistics
        print("\nPreprocessing Complete!")
        print(f"Feature value range: [{X.min():.4f}, {X.max():.4f}]")
        
        # Save preprocessed data
        save_preprocessed_data(X_train, y_train, X_test, y_test)
    save_preprocessed_data(X_train, y_train, X_test, y_test)
    
    """
