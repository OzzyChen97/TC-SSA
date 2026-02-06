"""
Training script for PANDA ISUP Grade Classification.

Features:
- 7:1:2 train/val/test split with stratification
- Multi-class classification (6 ISUP grades: 0-5)
- Comprehensive metrics (Accuracy, AUC, Quadratic Weighted Kappa)
- Class-weighted loss for imbalanced data
- Early stopping and learning rate scheduling

Usage:
    python tools/train_panda.py \
        --csv_path data/panda/train.csv \
        --features_dir data/CPathPatchFeature/panda/uni/pt_files \
        --feature_dim 1024 \
        --model_type moe \
        --num_slots 128 \
        --num_epochs 100 \
        --lr 1e-4 \
        --output_dir outputs/panda_experiment
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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    cohen_kappa_score,
    confusion_matrix,
    classification_report
)

from src.models import build_model
from src.utils import (
    set_seed,
    setup_logger,
    AverageMeter,
    save_checkpoint
)


class PANDADataset(torch.utils.data.Dataset):
    """
    Dataset for PANDA ISUP grading task.
    
    Handles the specific PANDA CSV format:
    - image_id: slide identifier
    - isup_grade: ISUP grade (0-5)
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        features_dir: str,
        feature_dim: int = 1024
    ):
        """
        Initialize PANDA dataset.
        
        Args:
            dataframe: DataFrame with columns [image_id, isup_grade]
            features_dir: Directory containing .pt feature files
            feature_dim: Feature dimension (default: 1024 for UNI)
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.features_dir = features_dir
        self.feature_dim = feature_dim
        
        # Validate files exist
        self._validate_files()
        
    def _validate_files(self):
        """Check that feature files exist."""
        missing = 0
        for idx, row in self.dataframe.iterrows():
            path = os.path.join(self.features_dir, f"{row['image_id']}.pt")
            if not os.path.exists(path):
                missing += 1
        
        if missing > 0:
            print(f"Warning: {missing} feature files missing out of {len(self.dataframe)}")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = row['image_id']
        label = row['isup_grade']
        
        # Load features
        feature_path = os.path.join(self.features_dir, f"{image_id}.pt")
        try:
            data = torch.load(feature_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading {feature_path}: {e}")
        
        # Handle different formats
        if isinstance(data, dict):
            if 'features' in data:
                features = data['features']
            elif 'feat' in data:
                features = data['feat']
            else:
                keys = [k for k, v in data.items() if isinstance(v, torch.Tensor)]
                features = data[keys[0]] if keys else None
        else:
            features = data
        
        # Ensure 2D
        if features.dim() == 1:
            features = features.unsqueeze(0)
        elif features.dim() == 3 and features.size(0) == 1:
            features = features.squeeze(0)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return features, label, image_id


def collate_fn(batch):
    """Collate function for variable-length sequences."""
    features_list = []
    labels_list = []
    ids_list = []
    
    for features, label, slide_id in batch:
        features_list.append(features)
        labels_list.append(label)
        ids_list.append(slide_id)
    
    labels_tensor = torch.stack(labels_list)
    
    return features_list, labels_tensor, ids_list


class EarlyStopping:
    """Early stopping with patience."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='max'):
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


def compute_multiclass_metrics(labels, preds, probs, num_classes=6):
    """
    Compute comprehensive metrics for multi-class classification.
    
    Args:
        labels: Ground truth labels
        preds: Predicted class labels
        probs: Predicted probabilities [N, num_classes]
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    # Accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Quadratic Weighted Kappa (important for ordinal ISUP grades)
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    
    # Multi-class AUC (one-vs-rest)
    try:
        # Handle case where some classes might be missing
        if probs.shape[1] >= 2:
            auc = roc_auc_score(
                labels, probs, 
                multi_class='ovr', 
                average='macro',
                labels=list(range(num_classes))
            )
        else:
            auc = 0.0
    except ValueError:
        auc = 0.0
    
    # Per-class accuracy from confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-10)
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'auc': auc,
        'per_class_acc': per_class_acc,
        'confusion_matrix': cm
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train PANDA ISUP Grade Classifier'
    )
    
    # Data parameters
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to PANDA CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing .pt feature files')
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Feature dimension (default: 1024)')
    
    # Split parameters
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='moe',
                        choices=['moe', 'mil_baseline'],
                        help='Model architecture type')
    parser.add_argument('--num_slots', type=int, default=128,
                        help='Number of MoE expert slots (default: 128)')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of ISUP grades (default: 6)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (default: 1e-5)')
    parser.add_argument('--aux_loss_weight', type=float, default=0.01,
                        help='Auxiliary loss weight (default: 0.01)')
    parser.add_argument('--grad_accum_steps', type=int, default=8,
                        help='Gradient accumulation steps (default: 8)')
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    
    # Optimizer settings
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type (default: adamw)')
    parser.add_argument('--scheduler', type=str, default='plateau',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='LR scheduler (default: plateau)')
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
                        help='Data loading workers (default: 4)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    # Logging
    parser.add_argument('--output_dir', type=str, default='./outputs/panda',
                        help='Output directory')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches (default: 10)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    
    return parser.parse_args()


def create_data_splits(csv_path, features_dir, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    Create stratified train/val/test splits.
    
    Args:
        csv_path: Path to PANDA CSV file
        features_dir: Directory containing feature files
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Load CSV
    df = pd.read_csv(csv_path)
    original_count = len(df)
    
    # Filter out samples without feature files
    print(f"Filtering samples with missing feature files...")
    valid_mask = df['image_id'].apply(
        lambda x: os.path.exists(os.path.join(features_dir, f"{x}.pt"))
    )
    df = df[valid_mask].reset_index(drop=True)
    
    filtered_count = len(df)
    print(f"Filtered: {original_count} -> {filtered_count} samples ({original_count - filtered_count} missing)")
    
    if filtered_count == 0:
        raise ValueError("No samples with valid feature files found!")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=df['isup_grade']
    )
    
    # Second split: val vs test
    val_relative_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_relative_ratio),
        random_state=seed,
        stratify=temp_df['isup_grade']
    )
    
    return train_df, val_df, test_df


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
    all_probs = np.vstack(all_probs)
    
    metrics = compute_multiclass_metrics(all_labels, all_preds, all_probs, args.num_classes)
    epoch_time = time.time() - start_time
    
    logger.info(
        f'Epoch [{epoch}] Training - '
        f'Loss: {loss_meter.avg:.4f} Acc: {metrics["accuracy"]:.4f} '
        f'Kappa: {metrics["kappa"]:.4f} AUC: {metrics["auc"]:.4f} '
        f'Time: {epoch_time:.2f}s'
    )
    
    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_loss_meter.avg,
        'aux_loss': aux_loss_meter.avg,
        **metrics
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device, args, epoch, logger, split_name='Validation'):
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
    all_probs = np.vstack(all_probs)
    
    metrics = compute_multiclass_metrics(all_labels, all_preds, all_probs, args.num_classes)
    
    logger.info(
        f'Epoch [{epoch}] {split_name} - '
        f'Loss: {loss_meter.avg:.4f} Acc: {metrics["accuracy"]:.4f} '
        f'Kappa: {metrics["kappa"]:.4f} AUC: {metrics["auc"]:.4f}'
    )
    
    # Print per-class accuracy
    logger.info(f"Per-class Accuracy: {[f'{acc:.3f}' for acc in metrics['per_class_acc']]}")
    
    return {
        'loss': loss_meter.avg,
        'ce_loss': ce_loss_meter.avg,
        'aux_loss': aux_loss_meter.avg,
        **metrics
    }


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = setup_logger(log_file=os.path.join(args.output_dir, 'train.log'))
    
    logger.info("=" * 60)
    logger.info("PANDA ISUP Grade Classification Training")
    logger.info("=" * 60)
    logger.info("Configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create data splits
    logger.info("\nCreating data splits...")
    train_df, val_df, test_df = create_data_splits(
        args.csv_path,
        args.features_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Val samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    # Print label distributions
    logger.info("\nTrain label distribution:")
    logger.info(f"{train_df['isup_grade'].value_counts().sort_index().to_dict()}")
    logger.info("\nVal label distribution:")
    logger.info(f"{val_df['isup_grade'].value_counts().sort_index().to_dict()}")
    logger.info("\nTest label distribution:")
    logger.info(f"{test_df['isup_grade'].value_counts().sort_index().to_dict()}")
    
    # Save splits to CSV for reproducibility
    train_df.to_csv(os.path.join(args.output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(args.output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(args.output_dir, 'test_split.csv'), index=False)
    logger.info("\nSaved data splits to output directory")
    
    # Create datasets
    train_dataset = PANDADataset(train_df, args.features_dir, args.feature_dim)
    val_dataset = PANDADataset(val_df, args.features_dir, args.feature_dim)
    test_dataset = PANDADataset(test_df, args.features_dir, args.feature_dim)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Build model
    logger.info("\nBuilding model...")
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
    if args.use_class_weights:
        class_counts = train_df['isup_grade'].value_counts().sort_index().values
        class_weights = torch.tensor(
            len(train_df) / (args.num_classes * class_counts),
            dtype=torch.float32
        ).to(device)
        logger.info(f"Using class weights: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Scheduler
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
    
    # Mixed precision scaler
    scaler = GradScaler() if args.use_amp else None
    
    # Early stopping
    early_stopper = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=0.0,
        mode='max'
    )
    logger.info(f"Early stopping: patience={args.early_stopping_patience}")
    
    # Training loop
    logger.info("\n" + "=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    best_val_kappa = -1.0
    best_val_auc = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, args, epoch, logger
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, args, epoch, logger, 'Validation'
        )
        
        # Save best model (using Kappa as primary metric for ordinal classification)
        if val_metrics['kappa'] > best_val_kappa:
            best_val_kappa = val_metrics['kappa']
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
            logger.info(f"Saved best model with Kappa: {best_val_kappa:.4f}")
        
        # Update scheduler
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['kappa'])
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Current learning rate: {current_lr:.6f}")
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopper(val_metrics['kappa']):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
        else:
            logger.info(f"Early stopping counter: {early_stopper.counter}/{args.early_stopping_patience}")
        
        # Save periodic checkpoints
        if epoch % args.save_freq == 0:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                    'args': vars(args)
                },
                filename=os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
    
    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 60)
    
    # Load best model
    best_checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    test_metrics = validate(
        model, test_loader, criterion, device, args, 0, logger, 'Test'
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best Validation Kappa: {best_val_kappa:.4f}")
    logger.info(f"Best Validation AUC: {best_val_auc:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Kappa: {test_metrics['kappa']:.4f}")
    logger.info(f"Test AUC: {test_metrics['auc']:.4f}")
    
    # Print confusion matrix
    logger.info("\nTest Set Confusion Matrix:")
    logger.info(f"\n{test_metrics['confusion_matrix']}")


if __name__ == '__main__':
    main()
