"""
Test script for PANDA Prostate Cancer Grade Assessment.

提供完整的测试和评估功能，包括：
- 模型性能评估（准确率、F1分数、混淆矩阵等）
- 类别级别的详细分析
- 可视化结果生成
- 预测结果保存

Usage:
    python tools/test_panda.py \
        --test_csv data/panda/test.csv \
        --features_dir data/CPathPatchFeature/panda/uni/pt_files \
        --checkpoint outputs/panda/best_model.pth \
        --output_dir results/panda \
        --num_classes 6
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score,
    cohen_kappa_score, matthews_corrcoef
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import WSIFeatureDataset, collate_fn_variable_length
from src.models import build_model
from src.utils import set_seed, setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test PANDA WSI Classifier'
    )

    # Data parameters
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing .pt feature files')
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Feature dimension (default: 1024)')

    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='moe',
                        choices=['moe', 'mil_baseline'],
                        help='Model architecture type')
    parser.add_argument('--num_slots', type=int, default=64,
                        help='Number of MoE expert slots (default: 64)')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of output classes for PANDA (default: 6)')

    # System parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results/panda',
                        help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='Save prediction results to CSV')

    return parser.parse_args()


def load_model(args, device):
    """Load model from checkpoint."""
    model = build_model(
        model_type=args.model_type,
        input_dim=args.feature_dim,
        num_slots=args.num_slots,
        num_classes=args.num_classes
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def test_model(model, dataloader, device, args):
    """
    Test model on dataset.
    
    Returns:
        Dictionary containing predictions, labels, probabilities, and slide IDs
    """
    all_labels = []
    all_preds = []
    all_probs = []
    all_slide_ids = []

    for features_list, labels, slide_ids in dataloader:
        labels = labels.to(device)

        for i, features in enumerate(features_list):
            features = features.unsqueeze(0).to(device)
            label = labels[i].unsqueeze(0)

            if args.use_amp:
                with autocast():
                    logits, _ = model(features)
            else:
                logits, _ = model(features)

            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)

            all_labels.append(label.cpu().numpy())
            all_preds.append(pred_class.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_slide_ids.append(slide_ids[i])

    return {
        'labels': np.concatenate(all_labels),
        'predictions': np.concatenate(all_preds),
        'probabilities': np.concatenate(all_probs),
        'slide_ids': all_slide_ids
    }


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_probs: np.ndarray, num_classes: int) -> Dict:
    """
    计算全面的评估指标。
    """
    # 基础指标
    accuracy = np.mean(y_true == y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # 分类报告
    report = classification_report(
        y_true, y_pred, 
        output_dict=True, 
        zero_division=0,
        target_names=[f'Class_{i}' for i in range(num_classes)]
    )

    # 每个类别的详细指标
    class_metrics = {}
    for c in range(num_classes):
        class_name = f'Class_{c}'
        if class_name in report:
            class_metrics[c] = {
                'precision': report[class_name]['precision'],
                'recall': report[class_name]['recall'],
                'f1-score': report[class_name]['f1-score'],
                'support': report[class_name]['support']
            }

    # 计算每个类别的准确率
    class_accuracies = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            class_accuracies[c] = np.mean(y_pred[mask] == c)
        else:
            class_accuracies[c] = 0.0

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc,
        'confusion_matrix': cm,
        'classification_report': report,
        'class_metrics': class_metrics,
        'class_accuracies': class_accuracies,
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_precision': report['weighted avg']['precision'],
        'weighted_avg_recall': report['weighted avg']['recall'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: str):
    """绘制并保存混淆矩阵。"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def plot_normalized_confusion_matrix(cm: np.ndarray, class_names: List[str], output_path: str):
    """绘制并保存归一化混淆矩阵。"""
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=1,
        cbar_kws={'label': 'Proportion'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized confusion matrix saved to: {output_path}")


def plot_class_metrics(class_metrics: Dict, output_path: str):
    """绘制各类别指标对比图。"""
    classes = list(class_metrics.keys())
    precision = [class_metrics[c]['precision'] for c in classes]
    recall = [class_metrics[c]['recall'] for c in classes]
    f1 = [class_metrics[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {c}' for c in classes])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Class metrics plot saved to: {output_path}")


def save_predictions(results: Dict, output_path: str):
    """Save prediction results to CSV."""
    df = pd.DataFrame({
        'slide_id': results['slide_ids'],
        'true_label': results['labels'],
        'predicted_label': results['predictions']
    })

    # Add probability columns
    for c in range(results['probabilities'].shape[1]):
        df[f'prob_class_{c}'] = results['probabilities'][:, c]

    df['correct'] = df['true_label'] == df['predicted_label']
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")


def print_metrics_report(metrics: Dict, num_classes: int):
    """打印详细的评估报告。"""
    print("\n" + "=" * 70)
    print("TEST RESULTS REPORT")
    print("=" * 70)

    print("\n【Overall Metrics】")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"  Cohen's Kappa:      {metrics['cohen_kappa']:.4f}")
    print(f"  Matthews Corr Coef: {metrics['matthews_corrcoef']:.4f}")

    print("\n【Macro Average】")
    print(f"  Precision: {metrics['macro_avg_precision']:.4f}")
    print(f"  Recall:    {metrics['macro_avg_recall']:.4f}")
    print(f"  F1-Score:  {metrics['macro_avg_f1']:.4f}")

    print("\n【Weighted Average】")
    print(f"  Precision: {metrics['weighted_avg_precision']:.4f}")
    print(f"  Recall:    {metrics['weighted_avg_recall']:.4f}")
    print(f"  F1-Score:  {metrics['weighted_avg_f1']:.4f}")

    print("\n【Per-Class Metrics】")
    print(f"{'Class':<10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10} {'Accuracy':>10}")
    print("-" * 70)
    for c in range(num_classes):
        if c in metrics['class_metrics']:
            m = metrics['class_metrics'][c]
            acc = metrics['class_accuracies'][c]
            print(f"{c:<10} {m['precision']:>12.4f} {m['recall']:>12.4f} "
                  f"{m['f1-score']:>12.4f} {int(m['support']):>10} {acc:>10.4f}")

    print("\n【Confusion Matrix】")
    cm = metrics['confusion_matrix']
    header = 'True\\Pred'
    print(f"{header:<10}", end='')
    for c in range(num_classes):
        print(f"{c:>8}", end='')
    print()
    print("-" * (10 + 8 * num_classes))
    for i in range(num_classes):
        print(f"{i:<10}", end='')
        for j in range(num_classes):
            print(f"{cm[i][j]:>8}", end='')
        print()

    print("=" * 70)


def save_metrics_json(metrics: Dict, output_path: str):
    """Save metrics to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif isinstance(value, dict):
            metrics_serializable[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in value.items()
            }
        else:
            metrics_serializable[key] = value

    with open(output_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"Metrics saved to: {output_path}")


def main():
    """Main testing function."""
    args = parse_args()

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(args.output_dir, 'test.log'))

    logger.info("=" * 70)
    logger.info("PANDA Testing Configuration")
    logger.info("=" * 70)
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from: {args.checkpoint}")
    model, checkpoint = load_model(args, device)

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = WSIFeatureDataset(
        csv_path=args.test_csv,
        features_dir=args.features_dir,
        feature_dim=args.feature_dim
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_variable_length,
        pin_memory=True
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Run testing
    logger.info("Running inference...")
    results = test_model(model, test_loader, device, args)

    # Compute metrics
    logger.info("Computing metrics...")
    metrics = compute_comprehensive_metrics(
        results['labels'],
        results['predictions'],
        results['probabilities'],
        args.num_classes
    )

    # Print report
    print_metrics_report(metrics, args.num_classes)

    # Save results
    logger.info("Saving results...")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    save_metrics_json(metrics, metrics_path)

    # Save predictions
    if args.save_predictions:
        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        save_predictions(results, predictions_path)

    # Generate visualizations
    logger.info("Generating visualizations...")
    class_names = [f'Class {i}' for i in range(args.num_classes)]

    # Confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)

    # Normalized confusion matrix
    cm_norm_path = os.path.join(args.output_dir, 'confusion_matrix_normalized.png')
    plot_normalized_confusion_matrix(metrics['confusion_matrix'], class_names, cm_norm_path)

    # Class metrics
    metrics_plot_path = os.path.join(args.output_dir, 'class_metrics.png')
    plot_class_metrics(metrics['class_metrics'], metrics_plot_path)

    logger.info("=" * 70)
    logger.info("Testing completed!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
