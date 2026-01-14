"""
Evaluation script for WSI Classification with MoE Token Compression.

This script evaluates a trained model on a test dataset and generates:
- Overall metrics (Accuracy, AUC, Precision, Recall, F1)
- Per-slide predictions CSV
- Confusion matrix
- ROC curve (optional)
"""

import argparse
import os
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.data import WSIFeatureDataset, collate_fn_variable_length
from src.models import build_model
from src.utils import set_seed, setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate WSI Classifier with MoE Token Compression'
    )

    # Data parameters
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing .pt feature files')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='moe',
                        choices=['moe', 'mil_baseline'],
                        help='Model architecture type')

    # System parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory for evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save per-slide predictions to CSV')

    return parser.parse_args()


@torch.no_grad()
def evaluate_model(model, dataloader, device, logger):
    """
    Evaluate the model on test data.

    Args:
        model: PyTorch model
        dataloader: Test dataloader
        device: Device to use
        logger: Logger instance

    Returns:
        Dictionary with predictions and metrics
    """
    model.eval()

    all_slide_ids = []
    all_labels = []
    all_preds = []
    all_probs = []

    logger.info("Running evaluation...")

    for batch_idx, (features_list, labels, slide_ids) in enumerate(dataloader):
        labels = labels.to(device)

        for i, features in enumerate(features_list):
            features = features.unsqueeze(0).to(device)  # [1, N, feature_dim]
            label = labels[i]

            # Forward pass
            logits, _ = model(features)

            # Get predictions
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)

            # Store results
            all_slide_ids.append(slide_ids[i])
            all_labels.append(label.cpu().item())
            all_preds.append(pred_class.cpu().item())
            all_probs.append(probs.cpu().numpy()[0])  # [num_classes]

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)  # [num_samples, num_classes]

    return {
        'slide_ids': all_slide_ids,
        'labels': all_labels,
        'preds': all_preds,
        'probs': all_probs
    }


def compute_detailed_metrics(results, num_classes, logger):
    """
    Compute detailed evaluation metrics.

    Args:
        results: Dictionary with predictions
        num_classes: Number of classes
        logger: Logger instance

    Returns:
        Dictionary with metrics
    """
    labels = results['labels']
    preds = results['preds']
    probs = results['probs']

    # Overall accuracy
    accuracy = accuracy_score(labels, preds)

    # AUC (for binary and multi-class)
    if num_classes == 2:
        # Binary classification: use probability of positive class
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        # Multi-class: use one-vs-rest
        try:
            auc = roc_auc_score(labels, probs, multi_class='ovr')
        except ValueError:
            auc = 0.0
            logger.warning("Could not compute AUC (possibly missing classes in predictions)")

    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average='weighted', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = \
        precision_recall_fscore_support(labels, preds, average=None, zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': {
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1': per_class_f1.tolist(),
            'support': per_class_support.tolist()
        }
    }

    return metrics


def print_metrics(metrics, logger):
    """Print metrics in a formatted way."""
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"AUC:       {metrics['auc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info("-" * 60)

    # Confusion matrix
    logger.info("Confusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    for row in cm:
        logger.info("  " + "  ".join(f"{val:>6}" for val in row))
    logger.info("-" * 60)

    # Per-class metrics
    logger.info("Per-Class Metrics:")
    per_class = metrics['per_class_metrics']
    for i in range(len(per_class['precision'])):
        logger.info(
            f"  Class {i}: "
            f"Precision={per_class['precision'][i]:.4f}, "
            f"Recall={per_class['recall'][i]:.4f}, "
            f"F1={per_class['f1'][i]:.4f}, "
            f"Support={per_class['support'][i]}"
        )
    logger.info("=" * 60)


def save_predictions(results, output_path, logger):
    """
    Save per-slide predictions to CSV.

    Args:
        results: Dictionary with predictions
        output_path: Path to save CSV
        logger: Logger instance
    """
    df = pd.DataFrame({
        'slide_id': results['slide_ids'],
        'true_label': results['labels'],
        'pred_label': results['preds'],
        **{f'prob_class_{i}': results['probs'][:, i]
           for i in range(results['probs'].shape[1])}
    })

    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup
    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(args.output_dir, 'eval.log'))

    # Log arguments
    logger.info("Evaluation Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load checkpoint to get model configuration
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Get model parameters from checkpoint args
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        feature_dim = ckpt_args.get('feature_dim', 1024)
        num_slots = ckpt_args.get('num_slots', 64)
        num_classes = ckpt_args.get('num_classes', 2)
        hidden_dim = ckpt_args.get('hidden_dim', 512)
        dropout = ckpt_args.get('dropout', 0.25)
        temperature = ckpt_args.get('temperature', 1.0)
        model_type = ckpt_args.get('model_type', args.model_type)
    else:
        # Use default values if not in checkpoint
        logger.warning("Model config not found in checkpoint, using defaults")
        feature_dim = 1024
        num_slots = 64
        num_classes = 2
        hidden_dim = 512
        dropout = 0.25
        temperature = 1.0
        model_type = args.model_type

    # Test dataset
    logger.info("Loading test dataset...")
    test_dataset = WSIFeatureDataset(
        csv_path=args.test_csv,
        features_dir=args.features_dir,
        feature_dim=feature_dim
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_variable_length,
        pin_memory=True
    )

    # Build model
    logger.info("Building model...")
    model = build_model(
        model_type=model_type,
        input_dim=feature_dim,
        num_slots=num_slots,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        temperature=temperature
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {num_params:,} parameters")

    # Evaluate
    results = evaluate_model(model, test_loader, device, logger)

    # Compute metrics
    metrics = compute_detailed_metrics(results, num_classes, logger)

    # Print metrics
    print_metrics(metrics, logger)

    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Save predictions if requested
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        save_predictions(results, predictions_path, logger)

    logger.info("Evaluation completed!")


if __name__ == '__main__':
    main()
