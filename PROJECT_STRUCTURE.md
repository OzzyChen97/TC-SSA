# Project Structure

Complete organization of the WSI MoE Classifier codebase.

## Directory Tree

```
wsi-moe-classifier/
├── README.md                    # Main documentation
├── QUICKSTART.md               # Quick start guide
├── PROJECT_STRUCTURE.md        # This file
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation script
├── config.yaml                 # Example configurations
├── .gitignore                  # Git ignore patterns
│
├── Core Modules
├── dataset.py                  # WSI feature dataset loader
├── model.py                    # MoE Token Compressor + Classifier
├── train.py                    # Training script with AMP
├── eval.py                     # Evaluation script
├── utils.py                    # Helper functions
│
├── Examples & Tools
├── inference_example.py        # Inference examples
├── test_installation.py        # Installation verification
│
├── scripts/
│   └── generate_dummy_data.py  # Generate test data
│
└── Data Structure (not included in repo)
    ├── data/
    │   ├── features/           # .pt feature files
    │   │   ├── slide_001.pt
    │   │   ├── slide_002.pt
    │   │   └── ...
    │   ├── train.csv           # Training metadata
    │   ├── val.csv             # Validation metadata
    │   └── test.csv            # Test metadata
    │
    └── outputs/                # Training outputs
        ├── train.log
        ├── best_model.pth
        └── checkpoint_epoch_*.pth
```

## Core Modules

### dataset.py
**Purpose**: Load pre-extracted WSI patch features

**Key Classes**:
- `WSIFeatureDataset`: PyTorch Dataset for loading .pt files
- `collate_fn_variable_length`: Custom collate function for variable-length sequences

**Features**:
- Handles both dictionary and tensor .pt formats
- Supports variable number of patches per slide
- Validates feature dimensions and file formats
- Provides class distribution statistics

**Usage**:
```python
from dataset import WSIFeatureDataset

dataset = WSIFeatureDataset(
    csv_path="data/train.csv",
    features_dir="data/features",
    feature_dim=1024
)
```

### model.py
**Purpose**: MoE-based architecture for WSI classification

**Key Classes**:
- `MoETokenCompressor`: Mixture of Experts token compression module
  - Gating network for routing patches to expert slots
  - Top-1 routing mechanism
  - Weighted aggregation per slot
  - Auxiliary load-balancing loss (CV²)

- `WSIClassifier`: Complete classification model
  - MoE Token Compressor
  - Mean pooling layer
  - MLP classifier

- `SimpleMILBaseline`: Attention-based MIL baseline for comparison

**Architecture Flow**:
```
Input [Batch, N, 1024]
  ↓
MoE Compressor [Batch, K, 1024]
  ↓
Mean Pool [Batch, 1024]
  ↓
MLP Classifier [Batch, num_classes]
```

**Usage**:
```python
from model import build_model

model = build_model(
    model_type='moe',
    num_slots=64,
    num_classes=2
)
```

### train.py
**Purpose**: Training loop with advanced features

**Features**:
- Mixed Precision Training (AMP)
- Gradient Accumulation
- Multiple optimizers (Adam, AdamW, SGD)
- Learning rate schedulers (Cosine, Step)
- Comprehensive metrics (Accuracy, AUC)
- Checkpointing (best model + periodic saves)
- Detailed logging

**Total Loss**:
```
Total Loss = CrossEntropyLoss + λ × aux_loss
```

**Usage**:
```bash
python train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --features_dir data/features \
    --use_amp \
    --num_epochs 50
```

### eval.py
**Purpose**: Model evaluation and inference

**Features**:
- Comprehensive metrics (Accuracy, AUC, Precision, Recall, F1)
- Confusion matrix
- Per-class performance
- Per-slide predictions export
- Automatic model config loading from checkpoint

**Usage**:
```bash
python eval.py \
    --test_csv data/test.csv \
    --features_dir data/features \
    --checkpoint outputs/best_model.pth \
    --save_predictions
```

### utils.py
**Purpose**: Shared utility functions

**Functions**:
- `set_seed()`: Reproducibility across all libraries
- `setup_logger()`: Logging configuration
- `compute_metrics()`: Calculate classification metrics
- `AverageMeter`: Track running averages
- `save_checkpoint()` / `load_checkpoint()`: Model persistence

## Configuration Files

### config.yaml
**Purpose**: Example configurations for different scenarios

**Configurations**:
1. **basic**: Standard training setup
2. **high_compression**: Fewer slots (32), higher compression
3. **low_compression**: More slots (128), lower compression
4. **multiclass**: Multi-class classification (>2 classes)
5. **mil_baseline**: Attention-based MIL for comparison
6. **fine_tuning**: Transfer learning on small datasets
7. **large_scale**: Large dataset configuration
8. **cpu_test**: CPU-only testing

### requirements.txt
**Purpose**: Python package dependencies

**Core Dependencies**:
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- tqdm >= 4.65.0

## Scripts & Tools

### scripts/generate_dummy_data.py
**Purpose**: Generate synthetic data for testing

**Features**:
- Generates .pt feature files with random embeddings
- Creates train/val/test CSV splits
- Configurable number of slides, patches, classes
- Supports both dictionary and tensor formats

**Usage**:
```bash
python scripts/generate_dummy_data.py \
    --num_slides 100 \
    --num_classes 2 \
    --output_dir data
```

### inference_example.py
**Purpose**: Demonstrate inference usage

**Examples**:
1. Single slide inference
2. Batch inference on multiple slides
3. Load and predict from directory

**Usage**:
```bash
python inference_example.py
```

### test_installation.py
**Purpose**: Verify installation and dependencies

**Tests**:
- Dependency imports
- Local module imports
- Model instantiation
- Forward pass
- Dataset loading (if data exists)
- Utility functions
- CUDA availability

**Usage**:
```bash
python test_installation.py
```

## Data Format

### CSV Files (train.csv, val.csv, test.csv)
```csv
slide_id,label
slide_001,0
slide_002,1
slide_003,0
```

**Columns**:
- `slide_id`: Unique identifier (matches .pt filename)
- `label`: Integer class label (0, 1, 2, ...)

### Feature Files (.pt)

**Format 1: Dictionary**
```python
{
    'features': torch.Tensor([N, 1024])  # N patches, 1024-dim features
}
```

**Format 2: Direct Tensor**
```python
torch.Tensor([N, 1024])  # N patches, 1024-dim features
```

Where:
- N: Number of patches (variable per slide, typically 100-10,000)
- 1024: Feature dimension (from foundation models like UNI, CTransPath)

## Output Structure

### Training Outputs (outputs/)
```
outputs/experiment_name/
├── train.log                    # Training logs
├── best_model.pth              # Best model (highest val AUC)
├── checkpoint_epoch_10.pth     # Periodic checkpoints
├── checkpoint_epoch_20.pth
└── ...
```

### Checkpoint Contents
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `epoch`: Current epoch
- `train_metrics`: Training metrics
- `val_metrics`: Validation metrics
- `args`: All training arguments

### Evaluation Outputs (eval_results/)
```
eval_results/
├── eval.log                    # Evaluation logs
├── metrics.json               # Detailed metrics
└── predictions.csv            # Per-slide predictions
```

## Model Architecture Details

### MoE Token Compressor

**Input**: [Batch, N, input_dim]
- Batch: Always 1 (due to variable N)
- N: Variable number of patches per slide
- input_dim: Feature dimension (e.g., 1024)

**Components**:
1. **Layer Normalization**: Stabilize input features
2. **Gating Network**: Linear(input_dim → num_slots)
3. **Routing**: Top-1 assignment (each patch to one slot)
4. **Aggregation**: Weighted sum of patches per slot
5. **Normalization**: Divide by patch count per slot

**Output**: [Batch, num_slots, input_dim]
- num_slots: Fixed number of compressed tokens (e.g., 64)

**Auxiliary Loss**:
```
expert_importance = Σ gate_probs per slot
CV² = (std / mean)²
```

### WSI Classifier

**Full Pipeline**:
1. Input: [1, N, 1024]
2. MoE Compressor: [1, 64, 1024]
3. Mean Pool: [1, 1024]
4. Linear: [1, 512] + ReLU + Dropout
5. Linear: [1, 256] + ReLU + Dropout
6. Linear: [1, num_classes]
7. Output: Logits [1, num_classes]

## Hyperparameter Guidelines

### MoE Configuration

| Parameter | Range | Effect |
|-----------|-------|--------|
| num_slots | 32-128 | Compression vs. capacity trade-off |
| temperature | 0.5-2.0 | Routing discreteness (lower = more discrete) |
| aux_loss_weight | 0.001-0.1 | Load balancing strength |

### Training Configuration

| Parameter | Range | Effect |
|-----------|-------|--------|
| lr | 5e-5 to 2e-4 | Learning rate |
| grad_accum_steps | 4-16 | Effective batch size |
| num_epochs | 30-100 | Training duration |
| dropout | 0.2-0.4 | Regularization |

### Recommended Configurations

**Binary Classification (Cancer vs. Normal)**:
- num_slots: 64
- lr: 1e-4
- num_epochs: 50
- aux_loss_weight: 0.01

**Multi-Class Classification**:
- num_slots: 96
- lr: 5e-5
- num_epochs: 75
- aux_loss_weight: 0.01

**Memory-Constrained**:
- num_slots: 32
- use_amp: True
- grad_accum_steps: 16

## Development Workflow

1. **Setup**
   ```bash
   pip install -r requirements.txt
   python test_installation.py
   ```

2. **Prepare Data**
   ```bash
   python scripts/generate_dummy_data.py --num_slides 100
   ```

3. **Train**
   ```bash
   python train.py --train_csv data/train.csv --features_dir data/features
   ```

4. **Evaluate**
   ```bash
   python eval.py --test_csv data/test.csv --checkpoint outputs/best_model.pth
   ```

5. **Inference**
   ```bash
   python inference_example.py
   ```

## Extension Points

### Adding New Features

1. **Custom Loss Functions**: Modify train.py, add to total_loss
2. **New Architectures**: Add to model.py, update build_model()
3. **Data Augmentation**: Modify dataset.py __getitem__()
4. **New Metrics**: Add to utils.py compute_metrics()
5. **Visualization**: Create new script using matplotlib/seaborn

### Common Modifications

**Change Backbone Features**:
- Update feature_dim in dataset.py and model.py
- Example: 768 for ViT-B, 1024 for UNI

**Add Attention Visualization**:
- Modify model.py to return attention weights
- Create visualization script

**Multi-GPU Training**:
- Add DataParallel or DistributedDataParallel to train.py
- Adjust batch_size and grad_accum_steps

## Troubleshooting Guide

### Common Issues

1. **Out of Memory**
   - Enable --use_amp
   - Increase --grad_accum_steps
   - Reduce --num_slots

2. **Poor Performance**
   - Check feature quality
   - Verify label correctness
   - Tune hyperparameters
   - Increase --num_epochs

3. **File Loading Errors**
   - Verify CSV format
   - Check .pt file format
   - Ensure feature_dim matches

4. **Slow Training**
   - Enable AMP
   - Use GPU
   - Increase num_workers

## References

- **Paper**: [Your paper citation]
- **GitHub**: [Repository URL]
- **Documentation**: See README.md and QUICKSTART.md
- **Issues**: [GitHub Issues URL]

## License

MIT License - See LICENSE file for details
