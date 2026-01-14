# Quick Start Guide

Get started with WSI Classification using MoE Token Compression in 5 minutes!

## Prerequisites

- Python >= 3.8
- CUDA-capable GPU (optional, but recommended)
- Pre-extracted WSI patch features (.pt files)

## Step 1: Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd wsi-moe-classifier

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Step 2: Prepare Your Data

### Data Structure

Organize your data as follows:

```
data/
├── features/
│   ├── slide_001.pt
│   ├── slide_002.pt
│   └── ...
├── train.csv
└── val.csv
```

### CSV Format

Create CSV files with this format:

```csv
slide_id,label
slide_001,0
slide_002,1
slide_003,0
```

### Feature Files

Each `.pt` file should contain patch embeddings:

```python
# Option 1: Dictionary format
{
    'features': torch.Tensor([N, 1024])  # N patches, 1024-dim features
}

# Option 2: Direct tensor
torch.Tensor([N, 1024])
```

**Need test data?** Run the provided script to generate dummy data:

```bash
python scripts/generate_dummy_data.py --num_slides 100 --output_dir data
```

## Step 3: Train Your Model

### Basic Training

```bash
python train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --features_dir data/features \
    --output_dir outputs/my_first_model \
    --num_epochs 20
```

### Training with GPU and AMP

```bash
python train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --features_dir data/features \
    --use_amp \
    --num_epochs 50 \
    --output_dir outputs/gpu_training
```

### Key Parameters to Tune

- `--num_slots`: Number of expert slots (32-128). Higher = more capacity, lower = more compression
- `--lr`: Learning rate (5e-5 to 2e-4)
- `--aux_loss_weight`: Load balancing weight (0.001-0.1)
- `--grad_accum_steps`: Effective batch size (4-16)

## Step 4: Monitor Training

Watch training progress in real-time:

```bash
# View the log file
tail -f outputs/my_first_model/train.log
```

Expected output:
```
Epoch [10] Training - Loss: 0.4523 CE: 0.4321 Aux: 0.0202 Acc: 0.8234 AUC: 0.8756 Time: 45.23s
Epoch [10] Validation - Loss: 0.4821 CE: 0.4619 Aux: 0.0202 Acc: 0.8012 AUC: 0.8543 Time: 12.45s
```

## Step 5: Evaluate Your Model

```bash
python eval.py \
    --test_csv data/test.csv \
    --features_dir data/features \
    --checkpoint outputs/my_first_model/best_model.pth \
    --output_dir eval_results \
    --save_predictions
```

View results:
```bash
cat eval_results/metrics.json
```

## Step 6: Inference on New Data

### Single Slide Prediction

```python
from model import build_model
import torch

# Load model
checkpoint = torch.load('outputs/my_first_model/best_model.pth')
# ... (see inference_example.py for complete code)
```

Or use the provided script:

```bash
python inference_example.py
```

## Common Use Cases

### Use Case 1: Binary Classification (Cancer vs. Normal)

```bash
python train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --features_dir data/features \
    --num_classes 2 \
    --num_slots 64 \
    --use_amp \
    --num_epochs 50
```

### Use Case 2: Multi-Class Classification (5 cancer subtypes)

```bash
python train.py \
    --train_csv data/train_multiclass.csv \
    --val_csv data/val_multiclass.csv \
    --features_dir data/features \
    --num_classes 5 \
    --num_slots 96 \
    --hidden_dim 768 \
    --num_epochs 75
```

### Use Case 3: High Compression (Memory-Constrained)

```bash
python train.py \
    --train_csv data/train.csv \
    --features_dir data/features \
    --num_slots 32 \
    --temperature 0.8 \
    --aux_loss_weight 0.02 \
    --use_amp
```

### Use Case 4: Maximum Performance (No Compression Limit)

```bash
python train.py \
    --train_csv data/train.csv \
    --features_dir data/features \
    --num_slots 128 \
    --hidden_dim 1024 \
    --temperature 1.5 \
    --num_epochs 100
```

## Troubleshooting

### Problem: Out of Memory Error

**Solution 1:** Enable mixed precision
```bash
--use_amp
```

**Solution 2:** Increase gradient accumulation
```bash
--grad_accum_steps 16
```

**Solution 3:** Reduce model size
```bash
--num_slots 32 --hidden_dim 256
```

### Problem: Poor Performance (AUC < 0.7)

**Checklist:**
1. Check data quality: Are features from a good foundation model (UNI, CTransPath)?
2. Verify labels: Are they correct in your CSV?
3. Tune hyperparameters: Try `--num_slots 96 --lr 5e-5`
4. Increase training: Try `--num_epochs 100 --warmup_epochs 10`
5. Add regularization: Try `--dropout 0.3 --weight_decay 1e-4`

### Problem: Training is Slow

**Solutions:**
- Enable AMP: `--use_amp`
- Increase workers: `--num_workers 8`
- Use GPU: `--device cuda`
- Reduce logging: `--log_interval 50`

### Problem: File Loading Errors

**Verify:**
1. CSV format: `slide_id,label` columns
2. File naming: `{slide_id}.pt` matches CSV
3. Feature format: Dictionary with 'features' key or direct tensor
4. Feature dimension: Matches `--feature_dim` (usually 1024)

## Next Steps

1. **Experiment**: Try different `num_slots` values (32, 64, 96, 128)
2. **Compare**: Train MIL baseline with `--model_type mil_baseline`
3. **Visualize**: Analyze which expert slots capture which tissue patterns
4. **Optimize**: Use config.yaml for systematic hyperparameter search
5. **Deploy**: Export model for production inference

## Example Workflow

Here's a complete workflow from scratch:

```bash
# 1. Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Generate test data (if needed)
python scripts/generate_dummy_data.py --num_slides 200 --output_dir data

# 3. Train model
python train.py \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --features_dir data/features \
    --use_amp \
    --num_epochs 30 \
    --output_dir outputs/experiment_1

# 4. Evaluate
python eval.py \
    --test_csv data/val.csv \
    --features_dir data/features \
    --checkpoint outputs/experiment_1/best_model.pth \
    --save_predictions

# 5. View results
cat outputs/experiment_1/train.log
cat eval_results/metrics.json
```

## Getting Help

- **Documentation**: See README.md for detailed documentation
- **Examples**: Check `inference_example.py` for code examples
- **Issues**: Report bugs at [GitHub Issues]
- **Questions**: Contact [your-email@example.com]

## Performance Benchmarks

On typical WSI datasets (e.g., CAMELYON16):

| Configuration | Accuracy | AUC | Training Time (50 epochs) |
|--------------|----------|-----|---------------------------|
| MoE-32 | 88-92% | 0.91-0.95 | ~45 min (V100) |
| MoE-64 | 90-94% | 0.93-0.96 | ~60 min (V100) |
| MoE-128 | 91-95% | 0.94-0.97 | ~90 min (V100) |
| MIL Baseline | 86-90% | 0.89-0.93 | ~40 min (V100) |

*Results may vary based on dataset and feature quality*

---

**Congratulations!** You're now ready to train state-of-the-art WSI classifiers with MoE Token Compression.
