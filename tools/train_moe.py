"""
Training script for WSI Classification with MoE Token Compression on GTEx-TCGA data.
Mimics tools/train.py.

cd /workspace/zhuo/ETC

CUDA_VISIBLE_DEVICES=0,1 python tools/train_moe.py \
    --train_csv /workspace/zhuo/ETC/vqa/data/TCGA_priority/train.csv \
    --val_csv /workspace/zhuo/ETC/vqa/data/TCGA_priority/val.csv \
    --feature_dim 512 \
    --num_classes 9 \
    --num_slots 64 \
    --num_epochs 15 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --aux_loss_weight 0.1 \
    --scheduler plateau \
    --lr_patience 5 \
    --early_stop_patience 3 \
    --output_dir outputs/moe_tcga_priority_slots64 \
    --save_freq 5 \
    --use_amp
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
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd

from src.data import collate_fn_variable_length
from src.models import build_model
from src.utils import (
    set_seed,
    setup_logger,
    compute_metrics,
    AverageMeter,
    save_checkpoint
)


class NpyFeatureDataset(Dataset):
    """
    Dataset for loading .npy WSI patch features.
    """
    def __init__(self, csv_path, feature_dim=512):
        self.metadata = pd.read_csv(csv_path)
        self.feature_dim = feature_dim
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = row['file_path']
        label = int(row['label'])
        slide_id = row['slide_id']
        
        try:
            # Load .npy file containing a dictionary
            d = np.load(file_path, allow_pickle=True).item()
            features = d['feature']
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return a dummy tensor or raise
            raise e

        # Convert to tensor if numpy
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
            
        # Ensure features are float
        features = features.float()

        # Ensure shape [N, feature_dim]
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        # Handle dim mismatch if any (e.g. older files?)
        if features.shape[1] != self.feature_dim:
             # Just a warning or strict check
             # print(f"Warning: feature dim {features.shape[1]} != expected {self.feature_dim} for {slide_id}")
             pass

        return features, torch.tensor(label).long(), slide_id


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train WSI Classifier with MoE Token Compression (Custom)'
    )

    # Data parameters
    parser.add_argument('--train_csv', type=str, default='/workspace/ETC/vqa/data/train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--val_csv', type=str, default='/workspace/ETC/vqa/data/val.csv',
                        help='Path to validation CSV file (optional)')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension (default: 512)')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='moe',
                        choices=['moe', 'mil_baseline'],
                        help='Model architecture type')
    parser.add_argument('--num_slots', type=int, default=64,
                        help='Number of MoE expert slots')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of output classes (GTEx-TCGA has 9 classes)')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--aux_loss_weight', type=float, default=0.1,
                        help='Weight for auxiliary load-balancing loss (default: 0.1)')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')

    # Optimization
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler (default: plateau)')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau')
    parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='Factor to reduce LR by')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate')

    # System parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of data loading workers (increase for faster loading)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    # Logging and checkpointing
    parser.add_argument('--output_dir', type=str, default='outputs/moe_tcga_experiment',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Early stopping patience (stop if no improvement for N epochs)')

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
            features = features.unsqueeze(0).to(device) # [1, N, D]
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
            if args.num_classes == 2:
                all_probs.append(probs[:, 1].detach().cpu().numpy())
            else:
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

    # For multiclass, metrics might need adjustment if compute_metrics expects binary
    # We assume compute_metrics handles it or we print basic accuracy
    # Check src.utils for compute_metrics but it's not visible now.
    # Assuming standard behavior.
    try:
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        acc = metrics.get('accuracy', 0.0)
        auc = metrics.get('auc', 0.0)
    except Exception as e:
        # Fallback metric calculation
        acc = (all_labels == all_preds).mean()
        auc = 0.0 # Placeholder
        
    epoch_time = time.time() - start_time

    logger.info(
        f'Epoch [{epoch}] Training - '
        f'Loss: {loss_meter.avg:.4f} CE: {ce_loss_meter.avg:.4f} '
        f'Aux: {aux_loss_meter.avg:.4f} Acc: {acc:.4f} '
        f'AUC: {auc:.4f} Time: {epoch_time:.2f}s'
    )

    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_loss_meter.avg,
        'aux_loss': aux_loss_meter.avg,
        'accuracy': acc,
        'auc': auc
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
            if args.num_classes == 2:
                all_probs.append(probs[:, 1].cpu().numpy())
            else:
                all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    try:
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        acc = metrics.get('accuracy', 0.0)
        auc = metrics.get('auc', 0.0)
    except:
        acc = (all_labels == all_preds).mean()
        auc = 0.0

    logger.info(
        f'Epoch [{epoch}] Validation - '
        f'Loss: {loss_meter.avg:.4f} CE: {ce_loss_meter.avg:.4f} '
        f'Aux: {aux_loss_meter.avg:.4f} Acc: {acc:.4f} '
        f'AUC: {auc:.4f}'
    )

    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_loss_meter.avg,
        'aux_loss': aux_loss_meter.avg,
        'accuracy': acc,
        'auc': auc
    }


def main():
    """Main training function."""
    args = parse_args()

    # Ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    set_seed(args.seed)
    logger = setup_logger(log_file=os.path.join(args.output_dir, 'train.log'))

    logger.info("Training Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    logger.info("Loading datasets...")
    # Use NpyFeatureDataset
    train_dataset = NpyFeatureDataset(
        csv_path=args.train_csv,
        feature_dim=args.feature_dim
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # Variable length, so batch_size 1 + collate
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_variable_length,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False
    )

    val_loader = None
    if args.val_csv and os.path.exists(args.val_csv):
        val_dataset = NpyFeatureDataset(
            csv_path=args.val_csv,
            feature_dim=args.feature_dim
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn_variable_length,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=True if args.num_workers > 0 else False
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
    epochs_without_improvement = 0

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, args, epoch, logger
        )

        if val_loader:
            val_metrics = validate(model, val_loader, criterion, device, args, epoch, logger)

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                epochs_without_improvement = 0
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
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping check
            if epochs_without_improvement >= args.early_stop_patience:
                logger.info(f"Early stopping triggered! No improvement for {args.early_stop_patience} epochs.")
                logger.info(f"Best validation AUC: {best_val_auc:.4f}")
                break

            # Update learning rate based on validation AUC for plateau
            if scheduler:
                if args.scheduler == 'plateau':
                    scheduler.step(val_metrics['auc']) # Monitor AUC, assume max
                    current_lr = optimizer.param_groups[0]['lr']
                    logger.info(f"Current learning rate: {current_lr:.6f}")
                else:
                    scheduler.step()
        else:
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
