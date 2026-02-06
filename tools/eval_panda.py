"""
PANDA ISUP Grading Evaluation Script

Evaluate a trained MoE model on the PANDA test set.

Usage:
    python tools/eval_panda.py \
        --checkpoint outputs/panda_isup_32slots/best_model.pth \
        --test_csv outputs/panda_isup_32slots/test_split.csv \
        --features_dir data/CPathPatchFeature/panda/uni/pt_files \
        --output_dir outputs/panda_isup_32slots/eval
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src.data import WSIFeatureDataset, collate_fn_variable_length
from src.models import build_model
from src.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate PANDA ISUP grading model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test split CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing .pt feature files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as checkpoint)')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of ISUP classes (default: 6)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap with both counts and percentages
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix - ISUP Grade Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_per_class_metrics(report_dict, output_path):
    """Plot per-class precision, recall, F1 scores."""
    classes = [f'ISUP {i}' for i in range(6)]
    
    precision = [report_dict[c]['precision'] for c in classes]
    recall = [report_dict[c]['recall'] for c in classes]
    f1 = [report_dict[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')
    
    ax.set_xlabel('ISUP Grade', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Classification Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics to {output_path}")


@torch.no_grad()
def evaluate(model, dataloader, device, use_amp=False, is_coral=False, num_classes=6):
    """Evaluate model on a dataset."""
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_slide_ids = []
    
    for features_list, labels, slide_ids in dataloader:
        labels = labels.to(device)
        
        for i, features in enumerate(features_list):
            features = features.unsqueeze(0).to(device)
            label = labels[i].unsqueeze(0)
            
            if use_amp:
                with autocast():
                    logits, _ = model(features)
            else:
                logits, _ = model(features)
            
            # Handle CORAL vs standard classification
            if is_coral:
                # CORAL: sigmoid + count
                sigmoid_proba = torch.sigmoid(logits)
                pred_class = (sigmoid_proba > 0.5).sum(dim=1)
                
                # Convert to class probabilities
                probs = torch.zeros(1, num_classes, device=logits.device)
                for k in range(num_classes):
                    if k == 0:
                        probs[:, k] = 1 - sigmoid_proba[:, 0]
                    elif k == num_classes - 1:
                        probs[:, k] = sigmoid_proba[:, k-1]
                    else:
                        probs[:, k] = sigmoid_proba[:, k-1] - sigmoid_proba[:, k]
                probs = torch.clamp(probs, min=0)
                probs = probs / probs.sum(dim=1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1)
            
            all_labels.append(label.cpu().numpy())
            all_preds.append(pred_class.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_slide_ids.append(slide_ids[i])
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    
    return all_labels, all_preds, all_probs, all_slide_ids


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    if 'args' in checkpoint:
        ckpt_args = checkpoint['args']
        model_type = ckpt_args.get('model_type', 'moe')
        num_slots = ckpt_args.get('num_slots', 32)
        feature_dim = ckpt_args.get('feature_dim', 1024)
        num_classes = ckpt_args.get('num_classes', args.num_classes)
        loss_type = ckpt_args.get('loss_type', 'combined')
        print(f"Model config from checkpoint:")
        print(f"  model_type: {model_type}")
        print(f"  num_slots: {num_slots}")
        print(f"  feature_dim: {feature_dim}")
        print(f"  num_classes: {num_classes}")
        print(f"  loss_type: {loss_type}")
    else:
        model_type = 'moe'
        num_slots = 32
        feature_dim = 1024
        num_classes = args.num_classes
        loss_type = 'combined'
    
    # For CORAL, model output dimension is num_classes - 1
    is_coral = (loss_type == 'coral')
    model_output_dim = num_classes - 1 if is_coral else num_classes
    actual_num_classes = num_classes  # Keep original for evaluation
    
    # Build model
    print("\nBuilding model...")
    if is_coral:
        print(f"CORAL model detected: using output dim = {model_output_dim}")
    model = build_model(
        model_type=model_type,
        input_dim=feature_dim,
        num_slots=num_slots,
        num_classes=model_output_dim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    
    # Load test data
    print(f"\nLoading test data from {args.test_csv}")
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
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate
    print("\nEvaluating...")
    if is_coral:
        print("Using CORAL prediction (ordinal regression)")
    labels, preds, probs, slide_ids = evaluate(
        model, test_loader, device, args.use_amp, 
        is_coral=is_coral, num_classes=actual_num_classes
    )
    
    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Quadratic Weighted Kappa: {kappa:.4f}")
    print("=" * 60)
    
    # Classification report
    class_names = [f'ISUP {i}' for i in range(actual_num_classes)]
    report = classification_report(labels, preds, target_names=class_names, output_dict=True)
    report_str = classification_report(labels, preds, target_names=class_names)
    
    print("\nClassification Report:")
    print(report_str)
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'quadratic_kappa': float(kappa),
        'per_class_metrics': {k: v for k, v in report.items() if k in class_names},
        'macro_avg': report.get('macro avg', {}),
        'weighted_avg': report.get('weighted avg', {}),
        'confusion_matrix': cm.tolist(),
        'num_samples': len(labels),
        'checkpoint': args.checkpoint,
        'test_csv': args.test_csv
    }
    
    results_path = os.path.join(args.output_dir, 'eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'slide_id': slide_ids,
        'true_label': labels,
        'predicted_label': preds,
        'correct': (labels == preds).astype(int)
    })
    
    # Add probability columns
    for i in range(num_classes):
        predictions_df[f'prob_ISUP_{i}'] = probs[:, i]
    
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved predictions to {predictions_path}")
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)
    
    # Plot per-class metrics
    metrics_path = os.path.join(args.output_dir, 'per_class_metrics.png')
    plot_per_class_metrics(report, metrics_path)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Samples: {len(labels)}")
    print(f"Correct: {(labels == preds).sum()}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Quadratic Kappa: {kappa:.4f}")
    print("=" * 60)
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(num_classes):
        mask = labels == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == i).mean()
            print(f"  ISUP {i}: {class_acc:.4f} ({mask.sum()} samples)")
    
    print(f"\nAll results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
