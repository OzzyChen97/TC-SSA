"""
Training script for WSI Classification with MoE Token Compression.

Supports:
- Mixed Precision Training (AMP)
- Gradient Accumulation
- Comprehensive metrics (Accuracy, AUC)
- Auxiliary load-balancing loss
- Checkpointing and logging

python tools/train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --features_dir /workspace/moe/CPathPatchFeature/brca/uni/pt_files \
    --model_type moe \
    --num_slots 64 \
    --num_classes 2 \
    --num_epochs 50 \
    --lr 1e-4 \
    --aux_loss_weight 0.01 \
    --use_amp \
    --output_dir outputs/full_experiment
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
        else:
            # No validation set, use cosine/step scheduler normally
            if scheduler and args.scheduler != 'plateau':
                scheduler.step()

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
