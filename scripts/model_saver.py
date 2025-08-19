"""
Utility module to save Keras models and training history files.

Usage:
    from scripts.model_saver import save_model_and_history
    save_model_and_history(model, history, base_name='my_model')

This saves:
  models/saved/{base_name}.keras
  models/saved/{base_name}_history.pkl
"""
import os
import pickle


def save_model_and_history(model, history, base_name, save_dir=None):
    """
    Save a Keras model and its training history to disk under models/saved.

    Parameters:
    -----------
    model : keras.Model
        The trained Keras model to save.
    history : keras.callbacks.History or dict
        Training history object returned by model.fit or a dict of metrics.
    base_name : str
        Base filename (without extension) for saving the model and history.
    save_dir : str or None
        Directory under which to save. Defaults to './models/saved'.

    Returns:
    --------
    model_path : str
        Path to the saved model file.
    history_path : str
        Path to the saved history file.
    """
    # Default save directory
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), 'models', 'saved')
    os.makedirs(save_dir, exist_ok=True)

    # Save model (architecture + weights + optimizer state)
    model_filename = f"{base_name}.keras"
    model_path = os.path.join(save_dir, model_filename)
    model.save(model_path)

    # Save history
    history_filename = f"{base_name}_history.pkl"
    history_path = os.path.join(save_dir, history_filename)
    # Extract history dict
    hist_data = history.history if hasattr(history, 'history') else history
    with open(history_path, 'wb') as f:
        pickle.dump(hist_data, f)

    print(f"Model saved to {model_path}")
    print(f"History saved to {history_path}")
    return model_path, history_path
