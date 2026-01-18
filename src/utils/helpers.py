"""
Utility functions for WSI Classification with MoE Token Compression.
Includes seeding, metric calculation, and logging helpers.
"""

import random
import numpy as np
import torch
import logging
from typing import Optional


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str = "WSI_MoE", log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def compute_metrics(y_true, y_pred, y_probs):
    """
    Compute classification metrics: Accuracy and AUC using PyTorch.
    Supports both binary and multi-class (One-vs-Rest Micro/Macro avg).

    Args:
        y_true: Ground truth labels (numpy array or list)
        y_pred: Predicted class labels (numpy array or list)
        y_probs: Predicted probabilities (numpy array or list).
                 Shape [N] for binary (prob of positive class),
                 Shape [N, C] for multi-class.

    Returns:
        Dictionary containing accuracy and AUC scores
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate accuracy
    accuracy = np.mean(y_true == y_pred)

    # Calculate AUC
    auc = 0.0
    try:
        classes = np.unique(y_true)
        num_classes = len(classes)
        
        # Check if we have probability scores
        if y_probs.ndim == 1 or (y_probs.ndim == 2 and y_probs.shape[1] == 1):
            # Binary classification case
            # Assumes y_true has 0 and 1, or needs mapping if not
            if num_classes <= 2:
                # Convert to tensors for helper
                y_true_tensor = torch.tensor(y_true, dtype=torch.long)
                y_probs_tensor = torch.tensor(y_probs.flatten(), dtype=torch.float32)
                auc = compute_auc_pytorch(y_true_tensor, y_probs_tensor)
        
        elif y_probs.ndim == 2 and y_probs.shape[1] > 1:
            # Multi-class case: Calculate Macro-Average AUC (One-vs-Rest)
            aucs = []
            # We iterate over all possible classes (based on prob shape, not just present labels)
            # This handles cases where a batch might miss a class, but model outputs probs for it
            num_model_classes = y_probs.shape[1]
            
            for c in range(num_model_classes):
                # Create binary labels for class c (1 vs Rest 0)
                y_true_binary = (y_true == c).astype(int)
                
                # If a class is not present in y_true, we cannot compute AUC for it correctly
                # (it would have 0 positives). Skippping or handling gracefully.
                if np.sum(y_true_binary == 1) > 0 and np.sum(y_true_binary == 0) > 0:
                    y_probs_binary = y_probs[:, c]
                    
                    y_true_tensor = torch.tensor(y_true_binary, dtype=torch.long)
                    y_probs_tensor = torch.tensor(y_probs_binary, dtype=torch.float32)
                    
                    class_auc = compute_auc_pytorch(y_true_tensor, y_probs_tensor)
                    aucs.append(class_auc)
            
            if aucs:
                auc = np.mean(aucs)
    
    except Exception as e:
        print(f"Metrics computation error: {e}")
        auc = 0.0

    return {
        'accuracy': float(accuracy),
        'auc': float(auc)
    }


def compute_auc_pytorch(y_true, y_probs):
    """
    Compute AUC score using PyTorch (manual implementation).

    Args:
        y_true: True labels (torch.Tensor)
        y_probs: Predicted probabilities for positive class (torch.Tensor)

    Returns:
        AUC score (float)
    """
    # Sort by predicted probabilities
    sorted_indices = torch.argsort(y_probs, descending=True)
    y_true_sorted = y_true[sorted_indices]

    # Count positives and negatives
    n_pos = torch.sum(y_true_sorted == 1).item()
    n_neg = torch.sum(y_true_sorted == 0).item()

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Calculate AUC using trapezoidal rule
    tpr_list = []
    fpr_list = []

    tp = 0
    fp = 0

    for label in y_true_sorted:
        if label == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / n_pos
        fpr = fp / n_neg

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Convert to tensors
    tpr_tensor = torch.tensor(tpr_list)
    fpr_tensor = torch.tensor(fpr_list)

    # Calculate area using trapezoidal rule
    # Sort by fpr to ensure proper integration
    sorted_indices = torch.argsort(fpr_tensor)
    fpr_sorted = fpr_tensor[sorted_indices]
    tpr_sorted = tpr_tensor[sorted_indices]

    # Compute AUC
    auc = torch.trapz(tpr_sorted, fpr_sorted).item()

    return auc


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='checkpoint.pth'):
    """
    Save model checkpoint.

    Args:
        state: Dictionary containing model state, optimizer state, etc.
        filename: Path to save checkpoint
    """
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """
    Load model checkpoint.

    Args:
        filename: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into

    Returns:
        Loaded checkpoint dictionary
    """
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint
