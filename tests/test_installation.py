"""
Installation verification script for WSI MoE Classifier.

This script tests:
1. Dependencies are installed correctly
2. Model can be instantiated
3. Dataset can be loaded
4. Basic forward pass works
5. Training utilities work
"""

import sys
import torch
import numpy as np
from pathlib import Path


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import torch
        import torchvision
        import numpy as np
        import pandas as pd
        import sklearn
        print("  ✓ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_local_imports():
    """Test that local modules can be imported."""
    print("\nTesting local module imports...")
    try:
        from src.data import WSIFeatureDataset, collate_fn_variable_length
        from src.models import WSIClassifier, MoETokenCompressor, build_model
        from src.utils import set_seed, setup_logger, compute_metrics
        print("  ✓ All local modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_model_instantiation():
    """Test that models can be instantiated."""
    print("\nTesting model instantiation...")
    try:
        from src.models import build_model

        # Test MoE model
        model_moe = build_model(
            model_type='moe',
            input_dim=1024,
            num_slots=64,
            num_classes=2
        )
        print(f"  ✓ MoE model created: {sum(p.numel() for p in model_moe.parameters()):,} parameters")

        # Test MIL baseline
        model_mil = build_model(
            model_type='mil_baseline',
            input_dim=1024,
            num_classes=2
        )
        print(f"  ✓ MIL baseline created: {sum(p.numel() for p in model_mil.parameters()):,} parameters")

        return True
    except Exception as e:
        print(f"  ✗ Model instantiation error: {e}")
        return False


def test_forward_pass():
    """Test that a forward pass works."""
    print("\nTesting forward pass...")
    try:
        from src.models import build_model

        model = build_model(
            model_type='moe',
            input_dim=1024,
            num_slots=64,
            num_classes=2
        )
        model.eval()

        # Create dummy input
        batch_size = 1
        num_patches = 1000
        feature_dim = 1024
        x = torch.randn(batch_size, num_patches, feature_dim)

        # Forward pass
        with torch.no_grad():
            logits, aux_loss = model(x)

        # Check outputs
        assert logits.shape == (batch_size, 2), f"Unexpected logits shape: {logits.shape}"
        assert aux_loss.dim() == 0, f"Aux loss should be scalar, got shape: {aux_loss.shape}"

        print(f"  ✓ Forward pass successful")
        print(f"    Input shape: {x.shape}")
        print(f"    Output logits shape: {logits.shape}")
        print(f"    Aux loss: {aux_loss.item():.6f}")

        return True
    except Exception as e:
        print(f"  ✗ Forward pass error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset():
    """Test dataset loading (if dummy data exists)."""
    print("\nTesting dataset loading...")

    # Check if dummy data exists
    data_dir = Path("data")
    if not data_dir.exists():
        print("  ⚠ Data directory not found, skipping dataset test")
        print("    Run: python scripts/generate_dummy_data.py --num_slides 10")
        return True

    train_csv = data_dir / "train.csv"
    features_dir = data_dir / "features"

    if not train_csv.exists():
        print("  ⚠ train.csv not found, skipping dataset test")
        return True

    try:
        from src.data import WSIFeatureDataset

        dataset = WSIFeatureDataset(
            csv_path=str(train_csv),
            features_dir=str(features_dir),
            feature_dim=1024
        )

        print(f"  ✓ Dataset loaded: {len(dataset)} samples")

        # Test __getitem__
        if len(dataset) > 0:
            features, label, slide_id = dataset[0]
            print(f"    Sample 0: {features.shape}, label={label}, id={slide_id}")

        return True
    except Exception as e:
        print(f"  ✗ Dataset loading error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    try:
        from src.utils import set_seed, compute_metrics, AverageMeter

        # Test seeding
        set_seed(42)
        print("  ✓ set_seed() works")

        # Test metrics
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 0, 0, 1]
        y_probs = [0.2, 0.8, 0.4, 0.3, 0.9]

        metrics = compute_metrics(y_true, y_pred, y_probs)
        print(f"  ✓ compute_metrics() works: accuracy={metrics['accuracy']:.3f}, auc={metrics['auc']:.3f}")

        # Test AverageMeter
        meter = AverageMeter()
        meter.update(1.0)
        meter.update(2.0)
        assert meter.avg == 1.5, "AverageMeter calculation error"
        print("  ✓ AverageMeter works")

        return True
    except Exception as e:
        print(f"  ✗ Utils error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA...")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")
        print(f"    PyTorch version: {torch.__version__}")

        # Test model on GPU
        try:
            from src.models import build_model
            model = build_model(model_type='moe', num_slots=32)
            model = model.cuda()
            x = torch.randn(1, 100, 1024).cuda()
            with torch.no_grad():
                logits, _ = model(x)
            print("  ✓ GPU forward pass successful")
        except Exception as e:
            print(f"  ⚠ GPU forward pass failed: {e}")
    else:
        print("  ⚠ CUDA not available (CPU only)")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("WSI MoE Classifier - Installation Verification")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Local Modules", test_local_imports),
        ("Model Instantiation", test_model_instantiation),
        ("Forward Pass", test_forward_pass),
        ("Dataset Loading", test_dataset),
        ("Utility Functions", test_utils),
        ("CUDA", test_cuda),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("  1. Generate test data: python scripts/generate_dummy_data.py --num_slides 50")
        print("  2. Train a model: python train.py --train_csv data/train.csv --features_dir data/features")
        print("  3. See QUICKSTART.md for more examples")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check that you're in the correct directory")
        print("  3. Verify Python version >= 3.8")
        return 1


if __name__ == '__main__':
    sys.exit(main())
