# Model definition and training
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Import the preprocessing functions
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.preprocessing import load_dataset, preprocess_data, split_dataset
from scripts.visualize import plot_confusion_matrix, plot_training_history

# Placeholder for model definition and training
def build_model(input_shape, num_classes):
    """
    Build a deep neural network model for classification.
    
    Parameters:
    -----------
    input_shape : int
        Number of input features
    num_classes : int
        Number of output classes
        
    Returns:
    --------
    model : keras.Sequential
        Compiled neural network model
    """
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_dropout_model(input_dim, num_classes, dropout_rate=0.3):
    """Build a neural network with dropout regularization"""
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(dropout_rate),  # Dropout after first hidden layer
        Dense(64, activation='relu'),
        Dropout(dropout_rate),  # Dropout after second hidden layer
        Dense(32, activation='relu'),
        Dropout(dropout_rate),  # Dropout after third hidden layer
        Dense(num_classes, activation='softmax')  # No dropout before output
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128):
    """
    Train model with early stopping based on validation loss.
    
    Parameters:
    -----------
    model : keras.Sequential
        Model to train
    X_train, y_train : numpy arrays
        Training data and labels
    X_val, y_val : numpy arrays
        Validation data and labels
    epochs : int, default=50
        Maximum number of epochs to train
    batch_size : int, default=128
        Batch size for training
        
    Returns:
    --------
    history : dict
        Training history
    """
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and print performance metrics.
    
    Parameters:
    -----------
    model : keras.Sequential
        Trained model
    X_test, y_test : numpy arrays
        Test data and labels
        
    Returns:
    --------
    results : dict
        Dictionary containing accuracy and confusion matrix
    """
    # Convert one-hot encoded y_test back to class indices for sklearn metrics
    y_test_classes = np.argmax(y_test, axis=1)
    
    # Predict classes
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate metrics
    acc = accuracy_score(y_test_classes, y_pred_classes)
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    report = classification_report(y_test_classes, y_pred_classes)
    
    # Print results
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Use visualization utility for confusion matrix
    class_names = [str(i) for i in range(len(np.unique(y_test_classes)))]
    plot_confusion_matrix(
        y_test_classes, 
        y_pred_classes,
        class_names=class_names,
        title='Model Confusion Matrix',
        normalize=True,
        save_path='confusion_matrix.png'
    )
    
    return {
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_test_classes,
        'y_pred': y_pred_classes
    }

if __name__ == "__main__":
    # Load and preprocess the dataset
    """
    print("Loading and preprocessing dataset...")
    X, y = load_dataset()
    X = preprocess_data(X)
    
    # Split the data
    X_train_val, X_test, y_train_val, y_test = split_dataset(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = split_dataset(X_train_val, y_train_val, test_size=0.2, random_state=42)
    
    # One-hot encode the labels
    num_classes = len(np.unique(y))
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build the model
    print("\nBuilding neural network model...")
    model = build_model(X_train.shape[1], num_classes)
    model.summary()
    
    # Train the model with early stopping
    print("\nTraining model with early stopping...")
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Plot training history
    print("\nGenerating training history visualization...")
    plot_training_history(
        history, 
        metrics=['loss', 'accuracy'],
        save_path='training_history.png'
    )
    
    # Evaluate the model
    print("\nEvaluating model on test data...")
    results = evaluate_model(model, X_test, y_test)
    
    # Save the model
    model.save('mnist_model.h5')
    print("\nModel saved as 'mnist_model.h5'")
    print("\nModel saved as 'mnist_model.h5'")
"""
