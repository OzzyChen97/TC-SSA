"""
PANDA Complete Pipeline Script

完整的PANDA数据处理、训练和测试流程：
1. 数据划分（7:1:2比例）
2. 模型训练（含多种优化策略）
3. 模型测试和评估

Usage:
    # 完整流程
    python tools/run_panda_pipeline.py \
        --data_dir /workspace/zhuo/ETC/data/CPathPatchFeature/panda/uni \
        --output_dir /workspace/zhuo/ETC/outputs/panda \
        --run_all

    # 仅数据划分
    python tools/run_panda_pipeline.py \
        --data_dir /workspace/zhuo/ETC/data/CPathPatchFeature/panda/uni \
        --output_dir /workspace/zhuo/ETC/data/CPathPatchFeature/panda \
        --split_only

    # 仅训练
    python tools/run_panda_pipeline.py \
        --data_dir /workspace/zhuo/ETC/data/CPathPatchFeature/panda \
        --output_dir /workspace/zhuo/ETC/outputs/panda \
        --train_only

    # 仅测试
    python tools/run_panda_pipeline.py \
        --data_dir /workspace/zhuo/ETC/data/CPathPatchFeature/panda \
        --output_dir /workspace/zhuo/ETC/results/panda \
        --test_only \
        --checkpoint /workspace/zhuo/ETC/outputs/panda/best_model.pth
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='PANDA Complete Pipeline'
    )

    # Data paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing PANDA feature files')
    parser.add_argument('--labels_csv', type=str, default=None,
                        help='Path to labels CSV file (optional)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    # Pipeline control
    parser.add_argument('--run_all', action='store_true',
                        help='Run complete pipeline: split, train, test')
    parser.add_argument('--split_only', action='store_true',
                        help='Only run data splitting')
    parser.add_argument('--train_only', action='store_true',
                        help='Only run training')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run testing')

    # Split parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')

    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes for PANDA (default: 6)')
    parser.add_argument('--model_type', type=str, default='moe',
                        choices=['moe', 'mil_baseline'],
                        help='Model type')
    parser.add_argument('--num_slots', type=int, default=64,
                        help='Number of MoE slots (default: 64)')

    # Optimization strategies
    parser.add_argument('--use_class_weights', action='store_true',
                        help='Use class weights for imbalanced data')
    parser.add_argument('--use_balanced_sampler', action='store_true',
                        help='Use balanced sampler')
    parser.add_argument('--use_focal_loss', action='store_true',
                        help='Use focal loss')
    parser.add_argument('--use_label_smoothing', action='store_true',
                        help='Use label smoothing')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')

    # Testing parameters
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for testing')

    # System parameters
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')

    return parser.parse_args()


def run_data_split(args):
    """Run data splitting."""
    print("\n" + "=" * 70)
    print("STEP 1: Data Splitting")
    print("=" * 70)

    cmd = [
        'python', 'tools/panda_split.py',
        '--data_dir', args.data_dir,
        '--output_dir', args.output_dir,
        '--seed', str(args.seed),
        '--train_ratio', str(args.train_ratio),
        '--val_ratio', str(args.val_ratio),
        '--test_ratio', str(args.test_ratio),
        '--stratify'
    ]

    if args.labels_csv:
        cmd.extend(['--labels_csv', args.labels_csv])

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd='/workspace/zhuo/ETC')

    if result.returncode != 0:
        raise RuntimeError("Data splitting failed")

    print("Data splitting completed successfully!")
    return os.path.join(args.output_dir, 'train.csv'), \
           os.path.join(args.output_dir, 'val.csv'), \
           os.path.join(args.output_dir, 'test.csv')


def run_training(args, train_csv, val_csv, test_csv):
    """Run model training."""
    print("\n" + "=" * 70)
    print("STEP 2: Model Training")
    print("=" * 70)

    train_output_dir = os.path.join(args.output_dir, 'training')
    Path(train_output_dir).mkdir(parents=True, exist_ok=True)

    # Determine loss type
    loss_type = 'ce'
    if args.use_focal_loss:
        loss_type = 'focal'
    elif args.use_label_smoothing:
        loss_type = 'label_smoothing'

    cmd = [
        'python', 'tools/train_panda.py',
        '--train_csv', train_csv,
        '--val_csv', val_csv,
        '--test_csv', test_csv,
        '--features_dir', args.data_dir,
        '--output_dir', train_output_dir,
        '--num_classes', str(args.num_classes),
        '--model_type', args.model_type,
        '--num_slots', str(args.num_slots),
        '--num_epochs', str(args.num_epochs),
        '--lr', str(args.lr),
        '--seed', str(args.seed),
        '--num_workers', str(args.num_workers),
        '--device', args.device,
        '--loss_type', loss_type,
        '--scheduler', 'plateau',
        '--use_class_weights' if args.use_class_weights else '',
        '--use_balanced_sampler' if args.use_balanced_sampler else '',
        '--use_amp' if args.use_amp else ''
    ]

    # Remove empty strings
    cmd = [c for c in cmd if c]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd='/workspace/zhuo/ETC')

    if result.returncode != 0:
        raise RuntimeError("Training failed")

    print("Training completed successfully!")
    return os.path.join(train_output_dir, 'best_model.pth')


def run_testing(args, test_csv, checkpoint):
    """Run model testing."""
    print("\n" + "=" * 70)
    print("STEP 3: Model Testing")
    print("=" * 70)

    test_output_dir = os.path.join(args.output_dir, 'testing')
    Path(test_output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        'python', 'tools/test_panda.py',
        '--test_csv', test_csv,
        '--features_dir', args.data_dir,
        '--checkpoint', checkpoint,
        '--output_dir', test_output_dir,
        '--num_classes', str(args.num_classes),
        '--model_type', args.model_type,
        '--num_slots', str(args.num_slots),
        '--seed', str(args.seed),
        '--num_workers', str(args.num_workers),
        '--device', args.device,
        '--save_predictions'
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd='/workspace/zhuo/ETC')

    if result.returncode != 0:
        raise RuntimeError("Testing failed")

    print("Testing completed successfully!")


def main():
    """Main pipeline function."""
    args = parse_args()

    print("=" * 70)
    print("PANDA Complete Pipeline")
    print("=" * 70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("=" * 70)

    # Determine which steps to run
    run_split = args.run_all or args.split_only or not (args.train_only or args.test_only)
    run_train = args.run_all or args.train_only
    run_test = args.run_all or args.test_only

    train_csv = None
    val_csv = None
    test_csv = None
    checkpoint = None

    try:
        # Step 1: Data Splitting
        if run_split:
            train_csv, val_csv, test_csv = run_data_split(args)
        else:
            # Use existing split files
            train_csv = os.path.join(args.data_dir, 'train.csv')
            val_csv = os.path.join(args.data_dir, 'val.csv')
            test_csv = os.path.join(args.data_dir, 'test.csv')

            if not all(os.path.exists(f) for f in [train_csv, val_csv, test_csv]):
                raise FileNotFoundError(
                    "Split files not found. Please run with --split_only first."
                )

        # Step 2: Training
        if run_train:
            checkpoint = run_training(args, train_csv, val_csv, test_csv)
        elif args.checkpoint:
            checkpoint = args.checkpoint
        else:
            checkpoint = os.path.join(args.output_dir, 'training', 'best_model.pth')

        # Step 3: Testing
        if run_test:
            if not checkpoint or not os.path.exists(checkpoint):
                raise FileNotFoundError(
                    f"Checkpoint not found: {checkpoint}. Please run training first or provide --checkpoint."
                )
            run_testing(args, test_csv, checkpoint)

        print("\n" + "=" * 70)
        print("Pipeline completed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
