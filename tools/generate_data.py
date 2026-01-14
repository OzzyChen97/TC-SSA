"""
Script to generate dummy WSI feature data for testing the pipeline.

This script creates:
1. Synthetic .pt files containing patch features
2. CSV files with slide IDs and labels
3. Train/val/test splits

Useful for:
- Testing the pipeline without real data
- Quick prototyping and debugging
- Demonstrating the codebase
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm


def generate_dummy_features(num_patches, feature_dim=1024, format='dict'):
    """
    Generate dummy patch features.

    Args:
        num_patches: Number of patches in the slide
        feature_dim: Feature dimension
        format: 'dict' or 'tensor'

    Returns:
        Features in specified format
    """
    # Generate random features
    features = torch.randn(num_patches, feature_dim)

    if format == 'dict':
        return {'features': features}
    else:
        return features


def generate_dataset(
    num_slides,
    num_classes=2,
    min_patches=500,
    max_patches=5000,
    feature_dim=1024,
    output_dir='data',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    format='dict',
    seed=42
):
    """
    Generate a complete dummy dataset.

    Args:
        num_slides: Total number of slides to generate
        num_classes: Number of classes
        min_patches: Minimum patches per slide
        max_patches: Maximum patches per slide
        feature_dim: Feature dimension
        output_dir: Output directory
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        format: Feature format ('dict' or 'tensor')
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create output directories
    output_path = Path(output_dir)
    features_dir = output_path / 'features'
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_slides} dummy slides...")
    print(f"  Classes: {num_classes}")
    print(f"  Patches per slide: {min_patches}-{max_patches}")
    print(f"  Feature dimension: {feature_dim}")
    print(f"  Format: {format}")
    print(f"  Output: {output_path}")
    print()

    # Generate metadata
    slide_ids = []
    labels = []

    for i in tqdm(range(num_slides), desc="Generating features"):
        slide_id = f"slide_{i:05d}"
        label = i % num_classes  # Simple class distribution

        # Random number of patches
        num_patches = np.random.randint(min_patches, max_patches + 1)

        # Generate features
        features = generate_dummy_features(num_patches, feature_dim, format)

        # Save to .pt file
        feature_path = features_dir / f"{slide_id}.pt"
        torch.save(features, feature_path)

        slide_ids.append(slide_id)
        labels.append(label)

    # Create DataFrame
    df = pd.DataFrame({
        'slide_id': slide_ids,
        'label': labels
    })

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split into train/val/test
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    # Save CSVs
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)

    print("\nDataset generation complete!")
    print(f"  Train: {len(train_df)} slides")
    print(f"  Val:   {len(val_df)} slides")
    print(f"  Test:  {len(test_df)} slides")
    print()

    # Print class distribution
    print("Class distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"  {split_name}:")
        for cls in range(num_classes):
            count = (split_df['label'] == cls).sum()
            print(f"    Class {cls}: {count} slides")
    print()

    print(f"Files saved to:")
    print(f"  Features: {features_dir}/")
    print(f"  Train CSV: {output_path / 'train.csv'}")
    print(f"  Val CSV: {output_path / 'val.csv'}")
    print(f"  Test CSV: {output_path / 'test.csv'}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate dummy WSI feature data for testing'
    )

    parser.add_argument('--num_slides', type=int, default=100,
                        help='Total number of slides (default: 100)')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes (default: 2)')
    parser.add_argument('--min_patches', type=int, default=500,
                        help='Minimum patches per slide (default: 500)')
    parser.add_argument('--max_patches', type=int, default=5000,
                        help='Maximum patches per slide (default: 5000)')
    parser.add_argument('--feature_dim', type=int, default=1024,
                        help='Feature dimension (default: 1024)')
    parser.add_argument('--output_dir', type=str, default='data',
                        help='Output directory (default: data)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Test set ratio (default: 0.15)')
    parser.add_argument('--format', type=str, default='dict',
                        choices=['dict', 'tensor'],
                        help='Feature file format (default: dict)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        print(f"Warning: train + val + test ratios = {total_ratio}, not 1.0")
        print("Ratios will be normalized.")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    # Generate dataset
    generate_dataset(
        num_slides=args.num_slides,
        num_classes=args.num_classes,
        min_patches=args.min_patches,
        max_patches=args.max_patches,
        feature_dim=args.feature_dim,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        format=args.format,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
