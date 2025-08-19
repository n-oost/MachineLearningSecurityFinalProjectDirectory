# Plotting utilities for Machine Learning Security Project
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
import pandas as pd

def set_style():
    """Set the default style for all plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("muted")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_confusion_matrix(y_true, y_pred, class_names=None, title='Confusion Matrix', 
                          normalize=False, save_path=None, cmap='Blues'):
    """
    Plot confusion matrix with optional normalization.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    class_names : list, optional
        Names of classes for axis labels
    title : str, default='Confusion Matrix'
        Plot title
    normalize : bool, default=False
        Whether to normalize confusion matrix values
    save_path : str, optional
        Path to save the plot
    cmap : str, default='Blues'
        Colormap for the plot
    """
    set_style()
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(10, 8))
    
    # Plot with seaborn for better styling
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_training_history(history, metrics=None, save_path=None):
    """
    Plot training history for specified metrics.
    
    Parameters:
    -----------
    history : dict or keras.callbacks.History
        Training history from model.fit()
    metrics : list, optional
        Metrics to plot (defaults to ['loss', 'accuracy'])
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    if hasattr(history, 'history'):
        history = history.history
    
    if metrics is None:
        metrics = ['loss', 'accuracy']
    
    # Determine number of subplots needed
    n_plots = len(metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 5))
    if n_plots == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot training metric
        ax.plot(history[metric], label=f'Training {metric}')
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}')
        
        ax.set_title(f'{metric.capitalize()} Over Epochs', fontsize=14)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_score, class_idx=None, multi_class='ovr', save_path=None):
    """
    Plot ROC curve for binary or multi-class classification.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels or label indicators
    y_score : array-like
        Target scores (probabilities)
    class_idx : int, optional
        For multi-class, specify class index to plot (plots all if None)
    multi_class : {'ovr', 'ovo'}, default='ovr'
        'ovr' for One-vs-Rest, 'ovo' for One-vs-One
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    # Check if binary or multi-class
    n_classes = len(np.unique(y_true))
    binary = n_classes <= 2
    
    plt.figure(figsize=(10, 8))
    
    if binary:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic', fontsize=16)
        plt.legend(loc="lower right")
        
    else:
        # Multi-class
        if class_idx is not None:
            # Plot for single class
            y_true_bin = (y_true == class_idx).astype(int)
            fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, class_idx])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'Class {class_idx} (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            
        else:
            # Plot for all classes
            for i in range(n_classes):
                y_true_bin = (y_true == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_bin, y_score[:, i])
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('Receiver Operating Characteristic (Multi-class)', fontsize=16)
        plt.legend(loc="lower right")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()

def visualize_data_distribution(X, y, method='tsne', perplexity=30, n_samples=1000, 
                                save_path=None):
    """
    Visualize data distribution using dimension reduction.
    
    Parameters:
    -----------
    X : array-like
        Features to visualize
    y : array-like
        Labels for coloring
    method : str, default='tsne'
        Dimension reduction method ('tsne', 'pca', etc.)
    perplexity : int, default=30
        Perplexity parameter for t-SNE
    n_samples : int, default=1000
        Number of samples to visualize (for large datasets)
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    # Subsample if needed
    if n_samples and X.shape[0] > n_samples:
        indices = np.random.choice(X.shape[0], n_samples, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Reduce dimensions
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedding = reducer.fit_transform(X_sample)
    else:
        # Fallback to t-SNE
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embedding = reducer.fit_transform(X_sample)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'label': y_sample
    })
    
    plt.figure(figsize=(12, 10))
    
    # Plot with Seaborn for better styling
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='viridis', 
                   alpha=0.7, s=50, edgecolor='w', linewidth=0.5)
    
    plt.title(f'Data Distribution Visualization ({method.upper()})', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Data distribution plot saved to {save_path}")
    
    plt.show()

def plot_decision_boundary(model, X, y, feature_indices=None, mesh_step=0.02, 
                          save_path=None):
    """
    Plot decision boundary for a classifier (works for 2D feature space).
    
    Parameters:
    -----------
    model : sklearn-like model with predict method
        Trained classifier
    X : array-like
        Features
    y : array-like
        Labels
    feature_indices : tuple, optional
        Which two features to use for visualization (uses PCA if None)
    mesh_step : float, default=0.02
        Step size for mesh grid
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    # If more than 2 features, reduce dimensions or use specified features
    if X.shape[1] > 2:
        if feature_indices is not None:
            # Use specified features
            X_vis = X[:, feature_indices]
        else:
            # Use t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_vis = tsne.fit_transform(X)
            print("Using t-SNE to visualize decision boundary")
    else:
        X_vis = X
    
    # Create mesh grid
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                         np.arange(y_min, y_max, mesh_step))
    
    # Make predictions on mesh grid
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 10))
    
    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdBu')
    
    # Plot training points
    scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors='k',
                         s=50, cmap='RdBu', alpha=0.8)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision Boundary", fontsize=16)
    plt.xlabel("Feature 1", fontsize=14)
    plt.ylabel("Feature 2", fontsize=14)
    plt.colorbar(scatter)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Decision boundary plot saved to {save_path}")
    
    plt.show()

def compare_original_vs_poisoned(X_orig, X_poisoned, indices=None, n_samples=5, 
                               save_path=None):
    """
    Compare original vs poisoned samples.
    
    Parameters:
    -----------
    X_orig : array-like
        Original clean data
    X_poisoned : array-like
        Poisoned data
    indices : array-like, optional
        Indices of samples to visualize (random if None)
    n_samples : int, default=5
        Number of samples to visualize if indices is None
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    # Default to MNIST-like data shape if needed
    if len(X_orig.shape) == 2:
        # Try to reshape to image (assume square)
        side = int(np.sqrt(X_orig.shape[1]))
        if side * side == X_orig.shape[1]:
            X_orig_reshaped = X_orig.reshape(-1, side, side)
            X_poisoned_reshaped = X_poisoned.reshape(-1, side, side)
        else:
            # Can't reshape, use original
            X_orig_reshaped = X_orig
            X_poisoned_reshaped = X_poisoned
            print("Warning: Could not reshape data to images")
    else:
        X_orig_reshaped = X_orig
        X_poisoned_reshaped = X_poisoned
    
    # Get indices to visualize
    if indices is None:
        indices = np.random.choice(len(X_orig), n_samples, replace=False)
    
    # Create plot
    fig, axes = plt.subplots(len(indices), 2, figsize=(8, 2*len(indices)))
    
    for i, idx in enumerate(indices):
        # Original sample
        if len(X_orig_reshaped.shape) == 3:
            # For image data
            axes[i, 0].imshow(X_orig_reshaped[idx], cmap='gray')
            axes[i, 1].imshow(X_poisoned_reshaped[idx], cmap='gray')
        else:
            # For non-image data
            axes[i, 0].plot(X_orig_reshaped[idx])
            axes[i, 1].plot(X_poisoned_reshaped[idx])
        
        axes[i, 0].set_title(f"Original {idx}")
        axes[i, 1].set_title(f"Poisoned {idx}")
        axes[i, 0].axis('off')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Original vs Poisoned Samples", fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

def plot_attack_success_rate(model_results, attack_types, metrics=None, save_path=None):
    """
    Plot attack success rates across different attack types.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing model performance metrics under different attacks
    attack_types : list
        List of attack types to compare
    metrics : list, optional
        Metrics to plot (defaults to ['accuracy', 'attack_success_rate'])
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    if metrics is None:
        metrics = ['accuracy', 'attack_success_rate']
    
    # Create DataFrame for easier plotting
    data = []
    for attack in attack_types:
        for metric in metrics:
            if attack in model_results and metric in model_results[attack]:
                data.append({
                    'Attack': attack,
                    'Metric': metric,
                    'Value': model_results[attack][metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar plot
    plt.figure(figsize=(12, 8))
    
    # Use Seaborn for nicer styling
    ax = sns.barplot(x='Attack', y='Value', hue='Metric', data=df, palette='viridis')
    
    # Add value labels on bars
    for i, container in enumerate(ax.containers):
        labels = [f"{v:.2f}" for v in df.loc[df['Metric'] == metrics[i%len(metrics)], 'Value']]
        ax.bar_label(container, labels=labels, fontsize=10)
    
    plt.title('Model Performance Under Different Attacks', fontsize=16)
    plt.xlabel('Attack Type', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.ylim(0, 1.1)  # Assuming metrics are between 0 and 1
    plt.legend(title='Metric')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attack success rate plot saved to {save_path}")
    
    plt.show()

def plot_defense_effectiveness(attack_results, defense_methods, attack_types=None,
                              metric='accuracy', save_path=None):
    """
    Plot effectiveness of different defense methods against attacks.
    
    Parameters:
    -----------
    attack_results : dict
        Nested dictionary with defense methods, attack types, and metrics
    defense_methods : list
        List of defense methods to compare
    attack_types : list, optional
        List of attack types to compare (all if None)
    metric : str, default='accuracy'
        Metric to plot
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    # Get all attack types if not specified
    if attack_types is None:
        attack_types = set()
        for defense in defense_methods:
            if defense in attack_results:
                attack_types.update(attack_results[defense].keys())
        attack_types = list(attack_types)
    
    # Create DataFrame for easier plotting
    data = []
    for defense in defense_methods:
        for attack in attack_types:
            if (defense in attack_results and 
                attack in attack_results[defense] and
                metric in attack_results[defense][attack]):
                data.append({
                    'Defense': defense,
                    'Attack': attack,
                    'Value': attack_results[defense][attack][metric]
                })
    
    df = pd.DataFrame(data)
    
    # Create grouped bar plot
    plt.figure(figsize=(14, 8))
    
    # Use Seaborn for nicer styling
    ax = sns.barplot(x='Defense', y='Value', hue='Attack', data=df, palette='viridis')
    
    # Add value labels on bars
    for i, container in enumerate(ax.containers):
        labels = [f"{v:.2f}" for v in df.loc[df['Attack'] == attack_types[i%len(attack_types)], 'Value']]
        if len(labels) > 0:
            ax.bar_label(container, labels=labels, fontsize=10)
    
    plt.title(f'Defense Effectiveness ({metric.capitalize()})', fontsize=16)
    plt.xlabel('Defense Method', fontsize=14)
    plt.ylabel(metric.capitalize(), fontsize=14)
    plt.ylim(0, 1.1)  # Assuming metrics are between 0 and 1
    plt.legend(title='Attack Type')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Defense effectiveness plot saved to {save_path}")
    
    plt.show()

def plot_adversarial_examples(X_orig, X_adv, y_true, model, n_samples=5, random_seed=42, save_path=None):
    """
    Visualize original samples and their adversarial counterparts with model predictions.
    
    Parameters:
    -----------
    X_orig : array-like
        Original clean samples
    X_adv : array-like
        Adversarially perturbed samples
    y_true : array-like
        True labels for the samples
    model : keras.Model or similar
        Model to use for predictions
    n_samples : int, default=5
        Number of sample pairs to visualize
    random_seed : int, default=42
        Random seed for sample selection
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    np.random.seed(random_seed)
    
    # Randomly select samples
    if n_samples < len(X_orig):
        indices = np.random.choice(len(X_orig), n_samples, replace=False)
    else:
        indices = np.arange(len(X_orig))
    
    # Get predictions
    y_pred_orig = np.argmax(model.predict(X_orig[indices]), axis=1)
    y_pred_adv = np.argmax(model.predict(X_adv[indices]), axis=1)
    
    # Calculate perturbation magnitude
    diffs = X_adv[indices] - X_orig[indices]
    l2_norms = np.sqrt(np.sum(diffs**2, axis=1))
    linf_norms = np.max(np.abs(diffs), axis=1)
    
    # Create figure
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 3*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    # Reshape if data is flat (assumes MNIST-like square images)
    if len(X_orig.shape) == 2:
        side = int(np.sqrt(X_orig.shape[1]))
        if side * side == X_orig.shape[1]:
            X_orig_display = X_orig[indices].reshape(-1, side, side)
            X_adv_display = X_adv[indices].reshape(-1, side, side)
            perturbation_display = diffs.reshape(-1, side, side)
        else:
            X_orig_display = X_orig[indices]
            X_adv_display = X_adv[indices]
            perturbation_display = diffs
            print("Warning: Could not reshape to square image")
    else:
        X_orig_display = X_orig[indices]
        X_adv_display = X_adv[indices]
        perturbation_display = diffs
    
    # Plot original, adversarial, and perturbation
    for i in range(n_samples):
        # Original image
        if len(X_orig_display.shape) == 3:  # If reshaped to 2D image
            axes[i, 0].imshow(X_orig_display[i], cmap='gray')
            axes[i, 1].imshow(X_adv_display[i], cmap='gray')
            axes[i, 2].imshow(perturbation_display[i], cmap='coolwarm', vmin=-0.5, vmax=0.5)
        else:  # If still 1D
            axes[i, 0].plot(X_orig_display[i])
            axes[i, 1].plot(X_adv_display[i])
            axes[i, 2].plot(perturbation_display[i])
        
        # Set titles
        orig_correct = y_pred_orig[i] == y_true[indices[i]]
        adv_correct = y_pred_adv[i] == y_true[indices[i]]
        
        axes[i, 0].set_title(f"Original: True={y_true[indices[i]]}, Pred={y_pred_orig[i]}\n{'✓' if orig_correct else '✗'}", 
                           color='green' if orig_correct else 'red')
        axes[i, 1].set_title(f"Adversarial: True={y_true[indices[i]]}, Pred={y_pred_adv[i]}\n{'✓' if adv_correct else '✗'}", 
                           color='green' if adv_correct else 'red')
        axes[i, 2].set_title(f"Perturbation\nL2={l2_norms[i]:.4f}, L∞={linf_norms[i]:.4f}")
        
        # Turn off axis
        for j in range(3):
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Original vs Adversarial Examples", fontsize=16, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Adversarial examples plot saved to {save_path}")
    
    plt.show()

def plot_confidence_distributions(model, X_clean, X_adv, y_true, save_path=None):
    """
    Plot confidence score distributions for clean vs. adversarial examples.
    
    Parameters:
    -----------
    model : keras.Model or similar
        Model to use for predictions
    X_clean : array-like
        Clean samples
    X_adv : array-like
        Adversarially perturbed samples
    y_true : array-like
        True labels
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    # Get predictions and confidence scores
    clean_probs = model.predict(X_clean)
    adv_probs = model.predict(X_adv)
    
    # Get confidence in true class and predicted class
    clean_true_conf = np.array([clean_probs[i, y_true[i]] for i in range(len(y_true))])
    adv_true_conf = np.array([adv_probs[i, y_true[i]] for i in range(len(y_true))])
    
    clean_pred_conf = np.max(clean_probs, axis=1)
    adv_pred_conf = np.max(adv_probs, axis=1)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot confidence in true class
    sns.histplot(clean_true_conf, kde=True, label='Clean', ax=axes[0], color='blue', alpha=0.5)
    sns.histplot(adv_true_conf, kde=True, label='Adversarial', ax=axes[0], color='red', alpha=0.5)
    axes[0].set_title('Confidence in True Class', fontsize=14)
    axes[0].set_xlabel('Confidence Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend()
    
    # Plot max confidence (usually in predicted class)
    sns.histplot(clean_pred_conf, kde=True, label='Clean', ax=axes[1], color='blue', alpha=0.5)
    sns.histplot(adv_pred_conf, kde=True, label='Adversarial', ax=axes[1], color='red', alpha=0.5)
    axes[1].set_title('Confidence in Predicted Class', fontsize=14)
    axes[1].set_xlabel('Confidence Score', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confidence distributions plot saved to {save_path}")
    
    plt.show()

def plot_per_class_vulnerability(model, X_clean, X_adv, y_true, class_names=None, save_path=None):
    """
    Analyze and plot vulnerability by class.
    
    Parameters:
    -----------
    model : keras.Model or similar
        Model to use for predictions
    X_clean : array-like
        Clean samples
    X_adv : array-like
        Adversarially perturbed samples
    y_true : array-like
        True labels
    class_names : list, optional
        Names of classes for axis labels
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    # Get predictions
    y_pred_clean = np.argmax(model.predict(X_clean), axis=1)
    y_pred_adv = np.argmax(model.predict(X_adv), axis=1)
    
    # Calculate per-class accuracy
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    clean_acc_by_class = np.zeros(n_classes)
    adv_acc_by_class = np.zeros(n_classes)
    vulnerability_by_class = np.zeros(n_classes)
    samples_by_class = np.zeros(n_classes)
    
    for i, c in enumerate(classes):
        # Get indices for this class
        idx = (y_true == c)
        samples_by_class[i] = np.sum(idx)
        
        # Calculate accuracy for clean and adversarial examples
        clean_correct = y_pred_clean[idx] == c
        adv_correct = y_pred_adv[idx] == c
        
        clean_acc_by_class[i] = np.mean(clean_correct) if len(clean_correct) > 0 else 0
        adv_acc_by_class[i] = np.mean(adv_correct) if len(adv_correct) > 0 else 0
        
        # Calculate vulnerability (accuracy drop)
        vulnerability_by_class[i] = clean_acc_by_class[i] - adv_acc_by_class[i]
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'Class': class_names if class_names else [f"Class {c}" for c in classes],
        'Clean Accuracy': clean_acc_by_class,
        'Adversarial Accuracy': adv_acc_by_class,
        'Vulnerability': vulnerability_by_class,
        'Samples': samples_by_class
    })
    
    # Sort by vulnerability for better visualization
    df = df.sort_values('Vulnerability', ascending=False)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot accuracy comparison
    df_melted = pd.melt(df, id_vars=['Class'], value_vars=['Clean Accuracy', 'Adversarial Accuracy'],
                        var_name='Type', value_name='Accuracy')
    sns.barplot(x='Class', y='Accuracy', hue='Type', data=df_melted, ax=axes[0])
    axes[0].set_title('Accuracy by Class: Clean vs. Adversarial', fontsize=14)
    axes[0].set_xlabel('Class', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
    
    # Plot vulnerability
    bars = sns.barplot(x='Class', y='Vulnerability', data=df, ax=axes[1], palette='rocket')
    axes[1].set_title('Vulnerability by Class (Accuracy Drop)', fontsize=14)
    axes[1].set_xlabel('Class', fontsize=12)
    axes[1].set_ylabel('Vulnerability (Clean Acc - Adv Acc)', fontsize=12)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
    
    # Add values on top of bars
    for i, p in enumerate(bars.patches):
        axes[1].annotate(f"{p.get_height():.2f}",
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha='center', va='center', xytext=(0, 10),
                      textcoords='offset points')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class vulnerability plot saved to {save_path}")
    
    plt.show()

def plot_robustness_summary(clean_model, poisoned_model, X_test, y_test, X_adv_dict, 
                           attack_names=None, save_path=None):
    """
    Create a comprehensive robustness summary visualization with multiple panels.
    
    Parameters:
    -----------
    clean_model : keras.Model
        Clean/baseline model
    poisoned_model : keras.Model
        Poisoned model to compare
    X_test : array-like
        Clean test samples
    y_test : array-like
        True test labels
    X_adv_dict : dict
        Dictionary of adversarial examples {attack_name: X_adv}
    attack_names : list, optional
        Names of attacks for labeling
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    if attack_names is None:
        attack_names = list(X_adv_dict.keys())
    
    # Calculate metrics for all scenarios
    results = {}
    
    # Clean performance
    clean_pred_clean = np.argmax(clean_model.predict(X_test), axis=1)
    clean_pred_poisoned = np.argmax(poisoned_model.predict(X_test), axis=1)
    
    results['Clean'] = {
        'clean_model_acc': np.mean(clean_pred_clean == y_test),
        'poisoned_model_acc': np.mean(clean_pred_poisoned == y_test),
        'clean_conf': np.max(clean_model.predict(X_test), axis=1),
        'poisoned_conf': np.max(poisoned_model.predict(X_test), axis=1)
    }
    
    # Adversarial performance
    for attack_name, X_adv in X_adv_dict.items():
        clean_pred_adv = np.argmax(clean_model.predict(X_adv), axis=1)
        poisoned_pred_adv = np.argmax(poisoned_model.predict(X_adv), axis=1)
        
        results[attack_name] = {
            'clean_model_acc': np.mean(clean_pred_adv == y_test),
            'poisoned_model_acc': np.mean(poisoned_pred_adv == y_test),
            'clean_conf': np.max(clean_model.predict(X_adv), axis=1),
            'poisoned_conf': np.max(poisoned_model.predict(X_adv), axis=1),
            'attack_success_clean': 1 - np.mean(clean_pred_adv == y_test),
            'attack_success_poisoned': 1 - np.mean(poisoned_pred_adv == y_test)
        }
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
    
    # 1. Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    scenarios = list(results.keys())
    clean_accs = [results[s]['clean_model_acc'] for s in scenarios]
    poisoned_accs = [results[s]['poisoned_model_acc'] for s in scenarios]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, clean_accs, width, label='Clean Model', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, poisoned_accs, width, label='Poisoned Model', color='salmon', alpha=0.8)
    
    ax1.set_xlabel('Attack Scenario', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Model Accuracy Across Attack Scenarios', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 2. Attack Success Rates
    ax2 = fig.add_subplot(gs[0, 1])
    attack_scenarios = [s for s in scenarios if s != 'Clean']
    
    if attack_scenarios:
        clean_success = [results[s]['attack_success_clean'] for s in attack_scenarios]
        poisoned_success = [results[s]['attack_success_poisoned'] for s in attack_scenarios]
        
        x_attack = np.arange(len(attack_scenarios))
        bars3 = ax2.bar(x_attack - width/2, clean_success, width, 
                       label='vs Clean Model', color='lightcoral', alpha=0.8)
        bars4 = ax2.bar(x_attack + width/2, poisoned_success, width, 
                       label='vs Poisoned Model', color='darkred', alpha=0.8)
        
        ax2.set_xlabel('Attack Type', fontsize=12)
        ax2.set_ylabel('Attack Success Rate', fontsize=12)
        ax2.set_title('Attack Success Rates', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_attack)
        ax2.set_xticklabels(attack_scenarios, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 3. Confidence Distribution Comparison
    ax3 = fig.add_subplot(gs[1, :])
    
    # Plot confidence distributions for clean vs first adversarial attack
    clean_conf_clean = results['Clean']['clean_conf']
    clean_conf_poisoned = results['Clean']['poisoned_conf']
    
    if attack_scenarios:
        first_attack = attack_scenarios[0]
        adv_conf_clean = results[first_attack]['clean_conf']
        adv_conf_poisoned = results[first_attack]['poisoned_conf']
        
        # Create bins
        bins = np.linspace(0, 1, 30).tolist()
        
        ax3.hist(clean_conf_clean, bins=bins, alpha=0.5, label='Clean Model on Clean Data', 
                color='blue', density=True)
        ax3.hist(clean_conf_poisoned, bins=bins, alpha=0.5, label='Poisoned Model on Clean Data', 
                color='red', density=True)
        ax3.hist(adv_conf_clean, bins=bins, alpha=0.5, label=f'Clean Model on {first_attack}', 
                color='lightblue', density=True)
        ax3.hist(adv_conf_poisoned, bins=bins, alpha=0.5, label=f'Poisoned Model on {first_attack}', 
                color='pink', density=True)
        
        ax3.set_xlabel('Confidence Score', fontsize=12)
        ax3.set_ylabel('Density', fontsize=12)
        ax3.set_title('Confidence Score Distributions', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)
    
    # 4. Robustness Metrics Summary Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Scenario', 'Clean Model Acc', 'Poisoned Model Acc', 'Accuracy Drop', 
               'Avg Confidence (Clean)', 'Avg Confidence (Poisoned)']
    
    for scenario in scenarios:
        clean_acc = results[scenario]['clean_model_acc']
        poisoned_acc = results[scenario]['poisoned_model_acc']
        acc_drop = clean_acc - poisoned_acc
        clean_conf_avg = np.mean(results[scenario]['clean_conf'])
        poisoned_conf_avg = np.mean(results[scenario]['poisoned_conf'])
        
        table_data.append([
            scenario,
            f"{clean_acc:.3f}",
            f"{poisoned_acc:.3f}",
            f"{acc_drop:.3f}",
            f"{clean_conf_avg:.3f}",
            f"{poisoned_conf_avg:.3f}"
        ])
    
    # Create table
    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color-code accuracy drops
    for i, row in enumerate(table_data):
        acc_drop = float(row[3])
        if acc_drop > 0.1:  # Significant degradation
            table[(i+1, 3)].set_facecolor('#ffcccb')  # Light red
        elif acc_drop < 0:  # Improvement (unexpected but possible)
            table[(i+1, 3)].set_facecolor('#ccffcc')  # Light green
    
    ax4.set_title('Robustness Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Add overall summary text
    overall_clean_acc = results['Clean']['clean_model_acc']
    overall_poisoned_acc = results['Clean']['poisoned_model_acc']
    overall_degradation = overall_clean_acc - overall_poisoned_acc
    
    if attack_scenarios:
        avg_attack_success_clean = np.mean([results[s]['attack_success_clean'] for s in attack_scenarios])
        avg_attack_success_poisoned = np.mean([results[s]['attack_success_poisoned'] for s in attack_scenarios])
        
        summary_text = (
            f"Overall Analysis:\n"
            f"• Clean data accuracy: Clean Model {overall_clean_acc:.1%}, Poisoned Model {overall_poisoned_acc:.1%}\n"
            f"• Poisoning impact: {overall_degradation:.1%} accuracy drop\n"
            f"• Average attack success: {avg_attack_success_clean:.1%} (vs Clean), {avg_attack_success_poisoned:.1%} (vs Poisoned)\n"
            f"• Robustness assessment: {'VULNERABLE' if overall_degradation > 0.1 or avg_attack_success_clean > 0.5 else 'ROBUST'}"
        )
    else:
        summary_text = (
            f"Overall Analysis:\n"
            f"• Clean data accuracy: Clean Model {overall_clean_acc:.1%}, Poisoned Model {overall_poisoned_acc:.1%}\n"
            f"• Poisoning impact: {overall_degradation:.1%} accuracy drop\n"
            f"• Robustness assessment: {'VULNERABLE' if overall_degradation > 0.1 else 'ROBUST'}"
        )
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=11, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle('Machine Learning Security: Comprehensive Robustness Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness summary plot saved to {save_path}")
    
    plt.show()
    
    # Return summary statistics for programmatic use
    return {
        'overall_degradation': overall_degradation,
        'clean_model_clean_acc': overall_clean_acc,
        'poisoned_model_clean_acc': overall_poisoned_acc,
        'attack_results': results
    }

def compare_model_training_history(history_clean, history_poisoned, save_path=None):
    """
    Compare training history of clean vs poisoned models.
    
    Parameters:
    -----------
    history_clean : dict
        Training history of the clean model
    history_poisoned : dict
        Training history of the poisoned model
    save_path : str, optional
        Path to save the plot
    """
    set_style()
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot accuracy
    axes[0].plot(history_clean['accuracy'], label='Clean Model', color='blue')
    axes[0].plot(history_poisoned['accuracy'], label='Poisoned Model', color='red')
    axes[0].set_title('Model Accuracy Comparison', fontsize=14)
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    
    # Plot loss
    axes[1].plot(history_clean['loss'], label='Clean Model', color='blue')
    axes[1].plot(history_poisoned['loss'], label='Poisoned Model', color='red')
    axes[1].set_title('Model Loss Comparison', fontsize=14)
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    
    # Plot 3: Dropout rate comparison
    plt.subplot(2, 3, 3)
    rates = list(dropout_test_accs.keys())
    accuracies = list(dropout_test_accs.values())
    baseline_line = [baseline_test_acc] * len(rates)

    plt.plot(rates, accuracies, 'bo-', label='Dropout Models', linewidth=2, markersize=8)
    plt.plot(rates, baseline_line, 'r--', label='Baseline Model', linewidth=2)
    plt.title('Dropout Rate vs. Test Accuracy')
    plt.xlabel('Dropout Rate')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 4: Model architecture comparison
    plt.subplot(2, 3, 4)
    models = ['Baseline', f'Dropout {best_dropout_rate}']
    test_accs = [baseline_test_acc, dropout_test_accs[best_dropout_rate]]
    colors = ['lightcoral', 'lightblue']

    bars = plt.bar(models, test_accs, color=colors, alpha=0.8)
    plt.title('Model Performance Comparison')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, test_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.4f}', ha='center', va='bottom')

    # Plot 5: Overfitting analysis
    plt.subplot(2, 3, 5)
    baseline_gap = np.array(baseline_history.history['accuracy']) - np.array(baseline_history.history['val_accuracy'])
    dropout_gap = np.array(best_dropout_history.history['accuracy']) - np.array(best_dropout_history.history['val_accuracy'])

    plt.plot(baseline_gap, label='Baseline Gap', linewidth=2)
    plt.plot(dropout_gap, label=f'Dropout {best_dropout_rate} Gap', linewidth=2)
    plt.title('Overfitting Analysis (Train-Val Gap)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 6: Regularization effectiveness
    plt.subplot(2, 3, 6)
    methods = ['No Regularization', 'Dropout Regularization']
    final_gaps = [baseline_gap[-1], dropout_gap[-1]]
    colors = ['lightcoral', 'lightgreen']

    bars = plt.bar(methods, final_gaps, color=colors, alpha=0.8)
    plt.title('Regularization Effectiveness')
    plt.ylabel('Final Train-Val Gap')

    # Add value labels on bars
    for bar, gap in zip(bars, final_gaps):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{gap:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history comparison plot saved to {save_path}")
    
    plt.show()