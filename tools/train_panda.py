"""
PANDA (Prostate cANcer graDe Assessment) Training Script

Train a MIL (Multiple Instance Learning) model for ISUP grade classification.
ISUP grades: 0~5 (6 classes)

Gleason to ISUP mapping:
- Gleason 0 (negative) -> ISUP 0
- Gleason 3+3=6 -> ISUP 1  
- Gleason 3+4=7 -> ISUP 2
- Gleason 4+3=7 -> ISUP 3
- Gleason 4+4=8, 3+5=8, 5+3=8 -> ISUP 4
- Gleason 4+5=9, 5+4=9, 5+5=10 -> ISUP 5

Usage:
    python tools/train_panda.py \
        --features_dir data/CPathPatchFeature/panda/uni/pt_files \
        --labels_csv data/panda/train.csv \
        --output_dir outputs/panda_isup

Before running, download the PANDA train.csv from Kaggle:
    https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data
    
Or use the command:
    kaggle competitions download -c prostate-cancer-grade-assessment -f train.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm
import json
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Train PANDA ISUP grading model')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing .pt feature files')
    parser.add_argument('--labels_csv', type=str, required=True,
                        help='Path to train.csv with image_id and isup_grade columns')
    parser.add_argument('--output_dir', type=str, default='outputs/panda_isup',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Feature dimension (UNI uses 1024)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for MIL model')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of ISUP classes (0-5)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    return parser.parse_args()


class AttentionMIL(nn.Module):
    """Attention-based Multiple Instance Learning model for WSI classification."""
    
    def __init__(self, input_dim=1024, hidden_dim=512, num_classes=6, dropout=0.25):
        super().__init__()
        
        # Feature transformation
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_patches, feature_dim) or (num_patches, feature_dim)
        Returns:
            logits: (batch, num_classes) or (num_classes,)
        """
        # Handle single sample case
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = x.size(0)
        
        # Feature transformation
        h = self.feature_extractor(x)  # (batch, num_patches, hidden_dim)
        
        # Attention weights
        a = self.attention(h)  # (batch, num_patches, 1)
        a = F.softmax(a, dim=1)  # Normalize across patches
        
        # Weighted aggregation
        z = torch.sum(a * h, dim=1)  # (batch, hidden_dim)
        
        # Classification
        logits = self.classifier(z)  # (batch, num_classes)
        
        if squeeze_output:
            logits = logits.squeeze(0)
        
        return logits


class PANDADataset(Dataset):
    """Dataset for PANDA features with ISUP labels."""
    
    def __init__(self, slide_ids, labels, features_dir, max_patches=512):
        self.slide_ids = slide_ids
        self.labels = labels
        self.features_dir = Path(features_dir)
        self.max_patches = max_patches
    
    def __len__(self):
        return len(self.slide_ids)
    
    def __getitem__(self, idx):
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        
        # Load features
        pt_path = self.features_dir / f"{slide_id}.pt"
        
        if not pt_path.exists():
            # Return empty tensor if file not found
            return torch.zeros(1, 1024), label, slide_id
        
        features = torch.load(pt_path, map_location='cpu')
        
        if isinstance(features, dict):
            features = features.get('features', features.get('feature', list(features.values())[0]))
        
        # Random sample if too many patches
        if features.size(0) > self.max_patches:
            indices = torch.randperm(features.size(0))[:self.max_patches]
            features = features[indices]
        
        return features, label, slide_id


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    features, labels, slide_ids = zip(*batch)
    
    # Pad sequences to max length in batch
    max_len = max(f.size(0) for f in features)
    feature_dim = features[0].size(1)
    
    padded_features = torch.zeros(len(features), max_len, feature_dim)
    attention_masks = torch.zeros(len(features), max_len)
    
    for i, f in enumerate(features):
        padded_features[i, :f.size(0)] = f
        attention_masks[i, :f.size(0)] = 1
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_features, labels, attention_masks, list(slide_ids)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for features, labels, masks, _ in tqdm(dataloader, desc='Training'):
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, accuracy, kappa


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels, masks, _ in tqdm(dataloader, desc='Evaluating'):
            features = features.to(device)
            labels = labels.to(device)
            
            logits = model(features)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, accuracy, kappa, all_preds, all_labels


def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load labels
    print(f"Loading labels from {args.labels_csv}")
    df = pd.read_csv(args.labels_csv)
    
    # Check required columns
    if 'image_id' not in df.columns:
        raise ValueError("CSV must have 'image_id' column")
    if 'isup_grade' not in df.columns:
        raise ValueError("CSV must have 'isup_grade' column")
    
    # Check which slides have features
    features_dir = Path(args.features_dir)
    available_files = set(f.stem for f in features_dir.glob('*.pt'))
    
    # Filter to slides with available features
    df_filtered = df[df['image_id'].isin(available_files)].copy()
    print(f"Found {len(df_filtered)} slides with features (out of {len(df)})")
    
    if len(df_filtered) == 0:
        raise ValueError("No matching slides found between labels and features!")
    
    # Print class distribution
    print("\nISUP Grade Distribution:")
    print(df_filtered['isup_grade'].value_counts().sort_index())
    
    # Split data: 70% train, 10% val, 20% test (7:1:2 ratio)
    slide_ids = df_filtered['image_id'].values
    labels = df_filtered['isup_grade'].values
    
    # First split: 80% train+val, 20% test
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        slide_ids, labels, test_size=0.2, stratify=labels, random_state=args.seed
    )
    
    # Second split: 87.5% train, 12.5% val (from train+val = 70% + 10%)
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        train_val_ids, train_val_labels, test_size=0.125, 
        stratify=train_val_labels, random_state=args.seed
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train_ids)} slides")
    print(f"  Val:   {len(val_ids)} slides")
    print(f"  Test:  {len(test_ids)} slides")
    
    # Create datasets
    train_dataset = PANDADataset(train_ids, train_labels, args.features_dir)
    val_dataset = PANDADataset(val_ids, val_labels, args.features_dir)
    test_dataset = PANDADataset(test_ids, test_labels, args.features_dir)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    # Create model
    model = AttentionMIL(
        input_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    best_val_kappa = -1
    best_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'train_kappa': [],
               'val_loss': [], 'val_acc': [], 'val_kappa': []}
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_kappa = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_acc, val_kappa, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_kappa'].append(train_kappa)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}")
        
        # Save best model
        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kappa': val_kappa,
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  >> New best model saved! (Kappa: {val_kappa:.4f})")
    
    # Load best model for testing
    print(f"\n{'='*60}")
    print(f"Training complete! Best epoch: {best_epoch}")
    print(f"Loading best model for testing...")
    
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_loss, test_acc, test_kappa, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Quadratic Kappa: {test_kappa:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                                target_names=[f'ISUP {i}' for i in range(args.num_classes)]))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'test_kappa': test_kappa,
        'best_epoch': best_epoch,
        'best_val_kappa': best_val_kappa,
        'history': history,
        'confusion_matrix': cm.tolist(),
        'config': vars(args)
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save split information
    splits = {
        'train': train_ids.tolist(),
        'val': val_ids.tolist(),
        'test': test_ids.tolist()
    }
    with open(os.path.join(args.output_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
