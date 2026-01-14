"""
Utility functions for WSI Classification with MoE Token Compression.
Includes seeding, metric calculation, and logging helpers.
"""

import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
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
    Compute classification metrics: Accuracy and AUC.

    Args:
        y_true: Ground truth labels (numpy array or list)
        y_pred: Predicted class labels (numpy array or list)
        y_probs: Predicted probabilities for positive class (numpy array or list)

    Returns:
        Dictionary containing accuracy and AUC scores
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # AUC requires at least two classes in y_true
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_probs)
        else:
            auc = 0.0  # Cannot compute AUC with single class
    except ValueError:
        auc = 0.0

    return {
        'accuracy': accuracy,
        'auc': auc
    }


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
