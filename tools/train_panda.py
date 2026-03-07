"""
Training script for PANDA Prostate Cancer Grade Assessment with MoE Token Compression.

针对Panda多分类任务中中间类别难以区分的问题，实现了多种优化策略：
- 类别平衡处理（Class Balancing）
- 标签平滑（Label Smoothing）
- Focal Loss（处理类别不平衡）
- 分层学习率（Layer-wise Learning Rate Decay）
- 多尺度特征融合
- 交叉验证策略

Usage:
    python tools/train_panda.py \
        --train_csv data/panda/train.csv \
        --val_csv data/panda/val.csv \
        --features_dir data/CPathPatchFeature/panda/uni/pt_files \
        --num_classes 6 \
        --num_epochs 100 \
        --lr 1e-4 \
        --use_class_weights \
        --use_focal_loss \
        --output_dir outputs/panda_experiment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

from src.data import WSIFeatureDataset, collate_fn_variable_length
from src.models import build_model
from src.utils import set_seed, setup_logger, AverageMeter, save_checkpoint


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    对难以分类的样本给予更高的权重，特别适用于中间类别难以区分的情况。
    """
    def __init__(self, num_classes: int, gamma: float = 2.0, alpha: float = None, reduction: str = 'mean'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else torch.ones(num_classes)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [N, num_classes] - logits
            targets: [N] - class indices
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha.to(inputs.device))
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    通过软化标签分布，防止模型过于自信，提高泛化能力。
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = nn.functional.log_softmax(x, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PANDA WSI Classifier with MoE Token Compression'
    )

    # Data parameters
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, required=True,
                        help='Path to validation CSV file')
    parser.add_argument('--test_csv', type=str, default=None,
                        help='Path to test CSV file (optional)')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing .pt feature files')
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Feature dimension (default: 1024)')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='moe',
                        choices=['moe', 'mil_baseline'],
                        help='Model architecture type')
    parser.add_argument('--num_slots', type=int, default=64,
                        help='Number of MoE expert slots (default: 64)')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of output classes for PANDA (default: 6)')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                        help='Layer-wise learning rate decay (default: 0.95)')
    parser.add_argument('--aux_loss_weight', type=float, default=0.01,
                        help='Weight for auxiliary load-balancing loss (default: 0.01)')
    parser.add_argument('--grad_accum_steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')

    # Loss and optimization strategies
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal', 'label_smoothing'],
                        help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (default: 2.0)')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (default: 0.1)')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    parser.add_argument('--use_balanced_sampler', action='store_true',
                        help='Use balanced sampler for training')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau (default: 5)')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor to reduce LR by (default: 0.5)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate (default: 1e-6)')

    # System parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='./outputs/panda',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches (default: 10)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    return parser.parse_args()


def get_class_weights(dataset) -> torch.Tensor:
    """
    计算类别权重用于处理不平衡数据。
    使用逆频率加权策略。
    """
    labels = dataset.metadata['label'].values
    class_counts = np.bincount(labels, minlength=dataset.get_num_classes())
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0)
    return torch.tensor(weights, dtype=torch.float32)


def create_balanced_sampler(dataset) -> WeightedRandomSampler:
    """
    创建平衡采样器，确保每个类别在训练中被均匀采样。
    """
    labels = dataset.metadata['label'].values
    class_counts = np.bincount(labels, minlength=dataset.get_num_classes())
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True
    )


def get_criterion(args, class_weights: torch.Tensor = None):
    """Get loss function based on arguments."""
    if args.loss_type == 'focal':
        return FocalLoss(
            num_classes=args.num_classes,
            gamma=args.focal_gamma,
            alpha=class_weights
        )
    elif args.loss_type == 'label_smoothing':
        return LabelSmoothingCrossEntropy(
            num_classes=args.num_classes,
            smoothing=args.label_smoothing
        )
    else:  # 'ce'
        return nn.CrossEntropyLoss(weight=class_weights)


def get_optimizer_with_layer_decay(model, args):
    """
    分层学习率优化器。
    为不同层设置不同的学习率，底层使用较低学习率。
    """
    param_groups = []

    # Compressor parameters with lower learning rate
    compressor_params = list(model.compressor.parameters())
    param_groups.append({
        'params': compressor_params,
        'lr': args.lr * args.lr_decay,
        'name': 'compressor'
    })

    # Classifier parameters with full learning rate
    classifier_params = list(model.classifier.parameters())
    param_groups.append({
        'params': classifier_params,
        'lr': args.lr,
        'name': 'classifier'
    })

    if args.optimizer == 'adam':
        optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)

    return optimizer


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray, num_classes: int) -> Dict:
    """
    计算多分类任务的全面评估指标。
    """
    # 基础指标
    accuracy = np.mean(y_true == y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # 每个类别的准确率
    class_accuracies = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            class_accuracies[f'class_{c}_acc'] = np.mean(y_pred[mask] == c)
        else:
            class_accuracies[f'class_{c}_acc'] = 0.0

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # 分类报告
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }


def train_epoch(model, dataloader, criterion, optimizer, scaler, device, args, epoch, logger):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    all_labels = []
    all_preds = []
    all_probs = []

    optimizer.zero_grad()
    start_time = time.time()

    for batch_idx, (features_list, labels, slide_ids) in enumerate(dataloader):
        labels = labels.to(device)

        for i, features in enumerate(features_list):
            features = features.unsqueeze(0).to(device)
            label = labels[i].unsqueeze(0)

            if args.use_amp:
                with autocast():
                    logits, aux_loss = model(features)
                    ce_loss = criterion(logits, label)
                    total_loss = ce_loss + args.aux_loss_weight * aux_loss
                    total_loss = total_loss / args.grad_accum_steps
            else:
                logits, aux_loss = model(features)
                ce_loss = criterion(logits, label)
                total_loss = ce_loss + args.aux_loss_weight * aux_loss
                total_loss = total_loss / args.grad_accum_steps

            if args.use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            loss_meter.update(total_loss.item() * args.grad_accum_steps, 1)
            ce_loss_meter.update(ce_loss.item(), 1)
            aux_loss_meter.update(aux_loss.item(), 1)

            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)

            all_labels.append(label.cpu().numpy())
            all_preds.append(pred_class.cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())

        if (batch_idx + 1) % args.grad_accum_steps == 0:
            if args.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % args.log_interval == 0:
            logger.info(
                f'Epoch [{epoch}] Batch [{batch_idx + 1}/{len(dataloader)}] '
                f'Loss: {loss_meter.avg:.4f} CE: {ce_loss_meter.avg:.4f} '
                f'Aux: {aux_loss_meter.avg:.4f}'
            )

    # Handle remaining gradients
    if len(dataloader) % args.grad_accum_steps != 0:
        if args.use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    metrics = compute_multiclass_metrics(all_labels, all_preds, all_probs, args.num_classes)
    epoch_time = time.time() - start_time

    logger.info(
        f'Epoch [{epoch}] Training - '
        f'Loss: {loss_meter.avg:.4f} CE: {ce_loss_meter.avg:.4f} '
        f'Aux: {aux_loss_meter.avg:.4f} Acc: {metrics["accuracy"]:.4f} '
        f'Balanced Acc: {metrics["balanced_accuracy"]:.4f} '
        f'Macro F1: {metrics["macro_avg_f1"]:.4f} Time: {epoch_time:.2f}s'
    )

    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_loss_meter.avg,
        'aux_loss': aux_loss_meter.avg,
        **metrics
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, args, epoch, logger, phase='Validation'):
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    ce_loss_meter = AverageMeter()
    aux_loss_meter = AverageMeter()

    all_labels = []
    all_preds = []
    all_probs = []

    for features_list, labels, slide_ids in dataloader:
        labels = labels.to(device)

        for i, features in enumerate(features_list):
            features = features.unsqueeze(0).to(device)
            label = labels[i].unsqueeze(0)

            if args.use_amp:
                with autocast():
                    logits, aux_loss = model(features)
                    ce_loss = criterion(logits, label)
                    total_loss = ce_loss + args.aux_loss_weight * aux_loss
            else:
                logits, aux_loss = model(features)
                ce_loss = criterion(logits, label)
                total_loss = ce_loss + args.aux_loss_weight * aux_loss

            loss_meter.update(total_loss.item(), 1)
            ce_loss_meter.update(ce_loss.item(), 1)
            aux_loss_meter.update(aux_loss.item(), 1)

            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)

            all_labels.append(label.cpu().numpy())
            all_preds.append(pred_class.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    metrics = compute_multiclass_metrics(all_labels, all_preds, all_probs, args.num_classes)

    logger.info(
        f'Epoch [{epoch}] {phase} - '
        f'Loss: {loss_meter.avg:.4f} CE: {ce_loss_meter.avg:.4f} '
        f'Aux: {aux_loss_meter.avg:.4f} Acc: {metrics["accuracy"]:.4f} '
        f'Balanced Acc: {metrics["balanced_accuracy"]:.4f} '
        f'Macro F1: {metrics["macro_avg_f1"]:.4f}'
    )

    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_loss_meter.avg,
        'aux_loss': aux_loss_meter.avg,
        **metrics
    }


def save_metrics_history(history: Dict, output_dir: str):
    """Save training history to JSON file."""
    history_path = os.path.join(output_dir, 'metrics_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    """Main training function."""
    args = parse_args()

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(args.output_dir, 'train.log'))

    logger.info("=" * 60)
    logger.info("PANDA Training Configuration")
    logger.info("=" * 60)
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = WSIFeatureDataset(
        csv_path=args.train_csv,
        features_dir=args.features_dir,
        feature_dim=args.feature_dim
    )

    val_dataset = WSIFeatureDataset(
        csv_path=args.val_csv,
        features_dir=args.features_dir,
        feature_dim=args.feature_dim
    )

    test_dataset = None
    if args.test_csv:
        test_dataset = WSIFeatureDataset(
            csv_path=args.test_csv,
            features_dir=args.features_dir,
            feature_dim=args.feature_dim
        )

    # Compute class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = get_class_weights(train_dataset)
        class_weights = class_weights.to(device)
        logger.info(f"Class weights: {class_weights.cpu().numpy()}")

    # Create data loaders
    if args.use_balanced_sampler:
        sampler = create_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn_variable_length,
            pin_memory=True
        )
        logger.info("Using balanced sampler for training")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_fn_variable_length,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_variable_length,
        pin_memory=True
    )

    test_loader = None
    if test_dataset:
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
        model_type=args.model_type,
        input_dim=args.feature_dim,
        num_slots=args.num_slots,
        num_classes=args.num_classes
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {num_params:,} trainable parameters")

    # Loss function
    criterion = get_criterion(args, class_weights)
    logger.info(f"Using loss function: {args.loss_type}")

    # Optimizer with layer-wise learning rate decay
    if args.model_type == 'moe' and args.lr_decay < 1.0:
        optimizer = get_optimizer_with_layer_decay(model, args)
        logger.info(f"Using layer-wise LR decay: {args.lr_decay}")
    else:
        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.min_lr)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.num_epochs // 3, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.lr_factor,
            patience=args.lr_patience, min_lr=args.min_lr
        )
        logger.info(f"Using ReduceLROnPlateau: patience={args.lr_patience}, factor={args.lr_factor}")

    scaler = GradScaler() if args.use_amp else None

    # Early stopping
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        mode='max'
    )
    logger.info(f"Early stopping enabled: patience={args.early_stopping_patience}")

    # Training history
    history = {
        'train': [],
        'val': [],
        'test': [],
        'config': vars(args)
    }

    best_val_metric = 0.0
    best_epoch = 0

    logger.info("Starting training...")
    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, args, epoch, logger
        )
        history['train'].append(train_metrics)

        val_metrics = validate(
            model, val_loader, criterion, device, args, epoch, logger, phase='Validation'
        )
        history['val'].append(val_metrics)

        # Update learning rate
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['balanced_accuracy'])
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Current learning rate: {current_lr:.6f}")
            else:
                scheduler.step()

        # Save best model based on balanced accuracy
        if val_metrics['balanced_accuracy'] > best_val_metric:
            best_val_metric = val_metrics['balanced_accuracy']
            best_epoch = epoch
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'args': vars(args)
                },
                filename=os.path.join(args.output_dir, 'best_model.pth')
            )
            logger.info(f"Saved best model with balanced accuracy: {best_val_metric:.4f}")

        # Save checkpoint periodically
        if epoch % args.save_freq == 0:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args)
                },
                filename=os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )

        # Save metrics history
        save_metrics_history(history, args.output_dir)

        # Check early stopping
        if early_stopper(val_metrics['balanced_accuracy']):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Final evaluation on test set
    if test_loader:
        logger.info("=" * 60)
        logger.info("Final Test Set Evaluation")
        logger.info("=" * 60)

        # Load best model
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])

        test_metrics = validate(
            model, test_loader, criterion, device, args, best_epoch, logger, phase='Test'
        )
        history['test'].append(test_metrics)

        # Save final results
        save_metrics_history(history, args.output_dir)

        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Best model at epoch {best_epoch} with validation balanced accuracy: {best_val_metric:.4f}")
        logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Test balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
        logger.info(f"Test macro F1: {test_metrics['macro_avg_f1']:.4f}")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
