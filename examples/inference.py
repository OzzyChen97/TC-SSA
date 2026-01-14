"""
Example inference script demonstrating how to use a trained WSI classifier.

This script shows:
1. How to load a trained model
2. How to perform inference on a single slide
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

3. How to batch process multiple slides
"""

import torch
import numpy as np
from pathlib import Path

from src.models import build_model


def load_trained_model(checkpoint_path, device='cuda'):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Loaded model in eval mode
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model configuration from checkpoint
    if 'args' in checkpoint:
        args = checkpoint['args']
        feature_dim = args.get('feature_dim', 1024)
        num_slots = args.get('num_slots', 64)
        num_classes = args.get('num_classes', 2)
        hidden_dim = args.get('hidden_dim', 512)
        dropout = args.get('dropout', 0.25)
        temperature = args.get('temperature', 1.0)
        model_type = args.get('model_type', 'moe')
    else:
        print("Warning: Model config not found in checkpoint, using defaults")
        feature_dim = 1024
        num_slots = 64
        num_classes = 2
        hidden_dim = 512
        dropout = 0.25
        temperature = 1.0
        model_type = 'moe'

    # Build model
    model = build_model(
        model_type=model_type,
        input_dim=feature_dim,
        num_slots=num_slots,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        temperature=temperature
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_type} model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Model configuration: {num_slots} slots, {num_classes} classes")

    return model


def predict_single_slide(model, features, device='cuda'):
    """
    Predict class for a single slide.

    Args:
        model: Trained model
        features: Patch features tensor [N, feature_dim]
        device: Device to use

    Returns:
        Dictionary with prediction results
    """
    # Ensure features are 2D
    if features.dim() == 1:
        features = features.unsqueeze(0)

    # Add batch dimension
    features = features.unsqueeze(0)  # [1, N, feature_dim]

    # Move to device
    features = features.to(device)

    # Inference
    with torch.no_grad():
        logits, aux_loss = model(features)

    # Get predictions
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1)

    # Convert to numpy
    probs_np = probs.cpu().numpy()[0]
    pred_class_np = pred_class.cpu().item()

    return {
        'predicted_class': pred_class_np,
        'probabilities': probs_np,
        'confidence': probs_np[pred_class_np],
        'logits': logits.cpu().numpy()[0]
    }


def batch_predict(model, feature_list, device='cuda'):
    """
    Predict classes for multiple slides.

    Args:
        model: Trained model
        feature_list: List of feature tensors, each [N_i, feature_dim]
        device: Device to use

    Returns:
        List of prediction dictionaries
    """
    predictions = []

    for features in feature_list:
        pred = predict_single_slide(model, features, device)
        predictions.append(pred)

    return predictions


def example_single_inference():
    """Example: Inference on a single slide."""
    print("=" * 60)
    print("Example 1: Single Slide Inference")
    print("=" * 60)

    # Path to trained model checkpoint
    checkpoint_path = "outputs/best_model.pth"

    # Check if checkpoint exists
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using train.py")
        return

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_trained_model(checkpoint_path, device=device)

    # Load features for a single slide (example)
    # In practice, you would load this from a .pt file
    feature_path = "data/features/example_slide.pt"

    if Path(feature_path).exists():
        # Load actual features
        data = torch.load(feature_path, map_location='cpu')

        # Handle both dictionary and direct tensor formats
        if isinstance(data, dict):
            features = data.get('features', data.get('feat'))
        else:
            features = data

        print(f"Loaded features: {features.shape}")
    else:
        # Generate dummy features for demonstration
        print(f"Feature file not found: {feature_path}")
        print("Generating dummy features for demonstration...")
        num_patches = 1000
        feature_dim = 1024
        features = torch.randn(num_patches, feature_dim)

    # Perform inference
    result = predict_single_slide(model, features, device=device)

    # Display results
    print("\nPrediction Results:")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    print(f"  Class Probabilities:")
    for i, prob in enumerate(result['probabilities']):
        print(f"    Class {i}: {prob:.4f}")
    print()


def example_batch_inference():
    """Example: Batch inference on multiple slides."""
    print("=" * 60)
    print("Example 2: Batch Inference on Multiple Slides")
    print("=" * 60)

    checkpoint_path = "outputs/best_model.pth"

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_trained_model(checkpoint_path, device=device)

    # Generate dummy features for multiple slides
    print("Generating dummy features for 5 slides...")
    feature_list = [
        torch.randn(np.random.randint(500, 2000), 1024)  # Variable N
        for _ in range(5)
    ]

    # Batch inference
    predictions = batch_predict(model, feature_list, device=device)

    # Display results
    print("\nBatch Prediction Results:")
    for i, pred in enumerate(predictions):
        print(f"  Slide {i+1}:")
        print(f"    Predicted Class: {pred['predicted_class']}")
        print(f"    Confidence: {pred['confidence']:.4f}")
    print()


def example_load_from_directory():
    """Example: Load and predict all slides in a directory."""
    print("=" * 60)
    print("Example 3: Predict All Slides in Directory")
    print("=" * 60)

    checkpoint_path = "outputs/best_model.pth"
    features_dir = "data/features"

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    if not Path(features_dir).exists():
        print(f"Features directory not found: {features_dir}")
        print("Creating dummy directory structure...")
        Path(features_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = load_trained_model(checkpoint_path, device=device)

    # Find all .pt files in directory
    feature_files = list(Path(features_dir).glob("*.pt"))

    if len(feature_files) == 0:
        print(f"No .pt files found in {features_dir}")
        return

    print(f"Found {len(feature_files)} feature files")

    # Process each file
    results = []
    for feature_file in feature_files[:5]:  # Limit to first 5 for demo
        # Load features
        data = torch.load(feature_file, map_location='cpu')

        if isinstance(data, dict):
            features = data.get('features', data.get('feat'))
        else:
            features = data

        # Predict
        pred = predict_single_slide(model, features, device=device)

        results.append({
            'slide_id': feature_file.stem,
            'prediction': pred
        })

    # Display results
    print("\nPrediction Results:")
    for res in results:
        print(f"  {res['slide_id']}:")
        print(f"    Class: {res['prediction']['predicted_class']} "
              f"(confidence: {res['prediction']['confidence']:.4f})")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("WSI Classification - Inference Examples")
    print("=" * 60 + "\n")

    # Run examples
    example_single_inference()
    # example_batch_inference()
    # example_load_from_directory()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)
