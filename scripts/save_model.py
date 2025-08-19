"""
Utility to save Keras models and training history with a customizable name.
"""
import os
import pickle


def save_model_and_history(model, history, save_name, save_dir="data/models/saved"):
    """
    Save a Keras model and its training history to disk.

    Parameters:
    -----------
    model : keras.Model
        Trained Keras model to save.
    history : History or dict
        Training history; if a Keras History object, its .history dict is used.
    save_name : str
        Base name for saved files (no extension).
    save_dir : str
        Directory path (relative to project root) where files will be saved.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Construct file paths
    model_path = os.path.join(save_dir, f"{save_name}.keras")
    history_path = os.path.join(save_dir, f"{save_name}_history.pkl")

    # Save the Keras model (architecture + weights + optimizer state)
    model.save(model_path)

    # Extract history data
    hist_data = history.history if hasattr(history, 'history') else history

    # Save history dict as pickle
    with open(history_path, 'wb') as f:
        pickle.dump(hist_data, f)

    print(f"Model saved to {model_path}")
    print(f"Training history saved to {history_path}")
