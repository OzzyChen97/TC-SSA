"""
Training script for WSI Classification with MoE Token Compression.

Supports:
- Mixed Precision Training (AMP)
- Gradient Accumulation
- Comprehensive metrics (Accuracy, AUC)
- Auxiliary load-balancing loss
- Checkpointing and logging

python tools/train.py \
    --train_csv data/nsclc/train.csv \
    --val_csv data/nsclc/test.csv \
    --features_dir data/CPathPatchFeature/nsclc/uni/pt_files \
    --feature_dim 1024 \
    --model_type moe \
    --num_slots 128 \
    --num_classes 2 \
    --num_epochs 100 \
    --lr 1e-4 \
    --aux_loss_weight 0.1 \
    --use_amp \
    --output_dir outputs/nsclc_uni_moe_experiment128
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np

from src.data import WSIFeatureDataset, collate_fn_variable_length
from src.models import build_model
from src.utils import (
    set_seed,
    setup_logger,
    compute_metrics,
    AverageMeter,
    save_checkpoint
)


class EarlyStopping:
    """Early stopping to stop training when validation metric doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' (whether lower or higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
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
        description='Train WSI Classifier with MoE Token Compression'
    )

    # Data parameters
    parser.add_argument('--train_csv', type=str, required=True,
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default=None,
                        help='Path to validation CSV file (optional)')
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
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes (default: 2)')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--aux_loss_weight', type=float, default=0.01,
                        help='Weight for auxiliary load-balancing loss (default: 0.01)')
    parser.add_argument('--grad_accum_steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (epochs without improvement, default: 15)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help='Minimum improvement to reset patience (default: 0.0)')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler (default: plateau)')
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
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches (default: 10)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    return parser.parse_args()


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
            all_probs.append(probs[:, 1].detach().cpu().numpy())

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

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    epoch_time = time.time() - start_time

    logger.info(
        f'Epoch [{epoch}] Training - '
        f'Loss: {loss_meter.avg:.4f} CE: {ce_loss_meter.avg:.4f} '
        f'Aux: {aux_loss_meter.avg:.4f} Acc: {metrics["accuracy"]:.4f} '
        f'AUC: {metrics["auc"]:.4f} Time: {epoch_time:.2f}s'
    )

    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_loss_meter.avg,
        'aux_loss': aux_loss_meter.avg,
        'accuracy': metrics['accuracy'],
        'auc': metrics['auc']
    }


def collect_expert_usage(model, dataloader, device, args, logger):
    """
    Collect expert usage statistics across the dataset.
    
    Args:
        model: The MoE model
        dataloader: Data loader
        device: Device to use
        args: Training arguments
        logger: Logger instance
    
    Returns:
        Dict with expert usage statistics
    """
    model.eval()
    
    # Check if model is MoE type
    if args.model_type != 'moe':
        logger.info("Expert usage stats only available for MoE model")
        return None
    
    total_slot_counts = None
    total_importance = None
    total_patches = 0
    num_samples = 0
    
    with torch.no_grad():
        for features_list, labels, slide_ids in dataloader:
            for features in features_list:
                features = features.unsqueeze(0).to(device)
                
                # Get routing statistics
                _, _, routing_stats = model(features, return_routing_stats=True)
                
                if total_slot_counts is None:
                    total_slot_counts = routing_stats['slot_counts'].clone()
                    total_importance = routing_stats['importance'].clone()
                else:
                    total_slot_counts += routing_stats['slot_counts']
                    total_importance += routing_stats['importance']
                
                total_patches += routing_stats['num_patches']
                num_samples += 1
    
    # Compute averages
    avg_slot_counts = total_slot_counts / num_samples
    avg_importance = total_importance / num_samples
    avg_patches_per_sample = total_patches / num_samples
    
    return {
        'avg_slot_counts': avg_slot_counts,
        'avg_importance': avg_importance,
        'avg_patches_per_sample': avg_patches_per_sample,
        'num_samples': num_samples
    }


def print_expert_usage(expert_stats, num_slots, logger):
    """
    Print expert usage statistics in a formatted way.
    
    Args:
        expert_stats: Dict with expert usage statistics
        num_slots: Number of expert slots
        logger: Logger instance
    """
    if expert_stats is None:
        return
    
    avg_counts = expert_stats['avg_slot_counts'].numpy()
    avg_importance = expert_stats['avg_importance'].numpy()
    
    logger.info("=" * 60)
    logger.info("Expert Usage Statistics")
    logger.info("=" * 60)
    logger.info(f"Average patches per sample: {expert_stats['avg_patches_per_sample']:.1f}")
    logger.info(f"Number of samples: {expert_stats['num_samples']}")
    logger.info("-" * 60)
    
    # Sort by usage count
    sorted_indices = np.argsort(avg_counts)[::-1]
    
    # Print top 10 most used experts
    logger.info("Top 10 Most Used Experts:")
    for i, idx in enumerate(sorted_indices[:10]):
        bar_len = int(avg_counts[idx] / avg_counts.max() * 20)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        logger.info(f"  Expert {idx:3d}: {bar} {avg_counts[idx]:6.1f} patches (importance: {avg_importance[idx]:.3f})")
    
    # Print bottom 10 least used experts
    logger.info("Bottom 10 Least Used Experts:")
    for i, idx in enumerate(sorted_indices[-10:]):
        bar_len = int(avg_counts[idx] / avg_counts.max() * 20) if avg_counts.max() > 0 else 0
        bar = '█' * bar_len + '░' * (20 - bar_len)
        logger.info(f"  Expert {idx:3d}: {bar} {avg_counts[idx]:6.1f} patches (importance: {avg_importance[idx]:.3f})")
    
    # Usage statistics summary
    active_experts = np.sum(avg_counts > 0.1)
    usage_std = np.std(avg_counts)
    usage_cv = usage_std / np.mean(avg_counts) if np.mean(avg_counts) > 0 else 0
    
    logger.info("-" * 60)
    logger.info(f"Active Experts (>0.1 patches): {active_experts}/{num_slots}")
    logger.info(f"Usage Std Dev: {usage_std:.2f}")
    logger.info(f"Usage Coefficient of Variation: {usage_cv:.2f}")
    logger.info("=" * 60)


@torch.no_grad()
def validate(model, dataloader, criterion, device, args, epoch, logger):
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
            all_probs.append(probs[:, 1].cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    metrics = compute_metrics(all_labels, all_preds, all_probs)

    logger.info(
        f'Epoch [{epoch}] Validation - '
        f'Loss: {loss_meter.avg:.4f} CE: {ce_loss_meter.avg:.4f} '
        f'Aux: {aux_loss_meter.avg:.4f} Acc: {metrics["accuracy"]:.4f} '
        f'AUC: {metrics["auc"]:.4f}'
    )

    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_loss_meter.avg,
        'aux_loss': aux_loss_meter.avg,
        'accuracy': metrics['accuracy'],
        'auc': metrics['auc']
    }


def main():
    """Main training function."""
    args = parse_args()

    set_seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=os.path.join(args.output_dir, 'train.log'))

    logger.info("Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    train_dataset = WSIFeatureDataset(
        csv_path=args.train_csv,
        features_dir=args.features_dir,
        feature_dim=args.feature_dim
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_variable_length,
        pin_memory=True
    )

    val_loader = None
    if args.val_csv:
        val_dataset = WSIFeatureDataset(
            csv_path=args.val_csv,
            features_dir=args.features_dir,
            feature_dim=args.feature_dim
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_variable_length,
            pin_memory=True
        )

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

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

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
        logger.info(f"Using ReduceLROnPlateau: patience={args.lr_patience}, factor={args.lr_factor}, min_lr={args.min_lr}")

    scaler = GradScaler() if args.use_amp else None

    logger.info("Starting training...")
    best_val_auc = 0.0
    
    # Initialize early stopping
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
        mode='max'  # We want to maximize AUC
    )
    logger.info(f"Early stopping enabled: patience={args.early_stopping_patience}, min_delta={args.early_stopping_min_delta}")

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, args, epoch, logger
        )

        if val_loader:
            val_metrics = validate(model, val_loader, criterion, device, args, epoch, logger)

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
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
                logger.info(f"Saved best model with AUC: {best_val_auc:.4f}")

            # Update learning rate based on validation AUC
            if scheduler:
                if args.scheduler == 'plateau':
                    scheduler.step(val_metrics['auc'])
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Current learning rate: {current_lr:.6f}")
                else:
                    scheduler.step()
            
            # Check early stopping
            if early_stopper(val_metrics['auc']):
                logger.info(f"Early stopping triggered! No improvement for {args.early_stopping_patience} epochs.")
                logger.info(f"Best validation AUC: {best_val_auc:.4f}")
                break
            else:
                logger.info(f"Early stopping counter: {early_stopper.counter}/{args.early_stopping_patience}")
        else:
            # No validation set, use cosine/step scheduler normally
            if scheduler and args.scheduler != 'plateau':
                scheduler.step()
        
        # Collect and print expert usage statistics
        if args.model_type == 'moe':
            expert_stats = collect_expert_usage(
                model, val_loader if val_loader else train_loader,
                device, args, logger
            )
            print_expert_usage(expert_stats, args.num_slots, logger)

        if epoch % args.save_freq == 0:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics if val_loader else None,
                    'args': vars(args)
                },
                filename=os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )

    logger.info("Training completed!")
    if val_loader:
        logger.info(f"Best validation AUC: {best_val_auc:.4f}")


if __name__ == '__main__':
    main()
