"""

"""

import numpy as np

def create_label_flip_poison(X, y, poison_fraction=0.13, target_label=0, seed=None):
    """
    Randomly select a fraction of training samples and flip their labels to target_label.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Training features.
    y : array-like, shape (n_samples,)
        Integer class labels.
    poison_fraction : float
        Fraction of samples to poison (0 < fraction < 1).
    target_label : int
        Label to assign to poisoned samples.
    seed : int or None
        Random seed for reproducibility.

    Returns:
    --------
    X_poisoned : ndarray
        Copy of X (unchanged features).
    y_poisoned : ndarray
        Copy of y with selected labels flipped.
    poisoned_indices : ndarray
        Indices of poisoned samples.
    """
    if seed is not None:
        np.random.seed(seed)
    X_poisoned = np.copy(X)
    y_poisoned = np.copy(y)
    n_samples = y_poisoned.shape[0]
    n_poison = int(np.floor(poison_fraction * n_samples))
    poisoned_indices = np.random.choice(n_samples, size=n_poison, replace=False)
    y_poisoned[poisoned_indices] = target_label
    return X_poisoned, y_poisoned, poisoned_indices
