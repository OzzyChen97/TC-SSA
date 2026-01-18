# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **WSI-VQA (Whole Slide Image Visual Question Answering)** system that combines a frozen MoE (Mixture of Experts) token compressor with Qwen3-4B-Instruct LLM for pathology image question answering. The system uses a two-stage training strategy to efficiently adapt visual features from pathology slides to natural language responses.

**Key Innovation**: Compresses variable-length WSI patch embeddings (N patches) into a fixed set of 16 semantic tokens using a pretrained MoE compressor, then projects these tokens into the LLM's embedding space for efficient vision-language alignment.

## Architecture

```
WSI Patches [B, N, 1024]
    ↓
MoE_Compressor (Frozen) → [B, 16, 1024]
    ↓
MLP Projector (Trainable) → [B, 16, hidden_size]
    ↓
Qwen3-4B-Instruct + Text
    ↓
Generated Answer
```

### Core Components

**MoE_Compressor (Visual Encoder)**
- Location: Imported from parent `src.models.MoE_Compressor`
- Always frozen during VQA training to preserve pretrained representations
- Compresses variable N patches → fixed 16 tokens via expert routing
- Input: `[B, N, 1024]` patch features (from UNI/ResNet/ConCH)
- Output: `[B, 16, 1024]` compressed semantic tokens

**MLPProjector**
- Location: `src/vqa_model.py`
- 2-layer MLP with GELU activation: `1024 → 2048 → hidden_size`
- Projects visual features to match LLM embedding dimension
- Stage 1: Trained alone (LLM frozen)
- Stage 2: Trained jointly with LLM

**MoE_Qwen_VQA (Main Model)**
- Location: `src/vqa_model.py`
- Combines frozen visual encoder + trainable projector + Qwen LLM
- Replaces `<image>` tokens in text with projected visual embeddings
- Two training modes via `freeze_llm()` / `unfreeze_llm()`

### Two-Stage Training Strategy

**Stage 1: Caption Pretraining**
- Freeze: LLM (Qwen3-4B)
- Train: Projector only
- Data: Caption data (slide descriptions)
- Goal: Learn visual-to-language alignment
- Learning rate: 1e-4
- Typical epochs: 3-5

**Stage 2: VQA Finetuning**
- Freeze: MoE_Compressor (always)
- Train: Projector + LLM (full finetuning)
- Data: Question-answer pairs
- Goal: Adapt to VQA task
- Learning rate: 1e-5 (10x smaller than Stage 1)
- Typical epochs: 5-10

## Common Commands

### Stage 1 Training (Caption Pretraining)

**Single GPU:**
```bash
python vqa/tools/train_vqa.py \
    --stage 1 \
    --moe_checkpoint /path/to/moe_best.pt \
    --llm_path Qwen/Qwen3-4B-Instruct-2507 \
    --data_path data/captions_train.csv \
    --output_dir outputs/stage1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --lr 1e-4
```

**Multi-GPU (DDP - Recommended):**
```bash
torchrun --nproc_per_node=2 vqa/tools/train_vqa.py \
    --stage 1 \
    --moe_checkpoint /path/to/moe_best.pt \
    --llm_path Qwen/Qwen3-4B-Instruct-2507 \
    --data_path data/captions_train.csv \
    --output_dir outputs/stage1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --lr 1e-4
```

### Stage 2 Training (VQA Finetuning)

```bash
torchrun --nproc_per_node=2 vqa/tools/train_vqa.py \
    --stage 2 \
    --moe_checkpoint /path/to/moe_best.pt \
    --llm_path Qwen/Qwen3-4B-Instruct-2507 \
    --load_projector outputs/stage1/final/projector.pt \
    --data_path data/vqa_train.json \
    --output_dir outputs/stage2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 5 \
    --lr 1e-5
```

**Note**: Always load the Stage 1 projector checkpoint when starting Stage 2.

### Training with SlideChat Dataset

The codebase includes specialized support for the SlideChat dataset format:

**Stage 1 (SlideChat):**
```bash
torchrun --nproc_per_node=2 vqa/tools/train_slidechat.py \
    --stage 1 \
    --moe_checkpoint /path/to/moe_best.pt \
    --llm_path Qwen/Qwen3-4B-Instruct-2507 \
    --data_path vqa/data/SlideChat/SlideInstruct_train_stage1_caption.json \
    --features_dir vqa/data/SlideChat/Feat \
    --output_dir outputs/slidechat_stage1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --lr 1e-4
```

**Stage 2 (SlideChat):**
```bash
torchrun --nproc_per_node=2 vqa/tools/train_slidechat.py \
    --stage 2 \
    --moe_checkpoint /path/to/moe_best.pt \
    --llm_path Qwen/Qwen3-4B-Instruct-2507 \
    --load_projector outputs/slidechat_stage1/final/projector.pt \
    --data_path vqa/data/SlideChat/SlideInstruct_train_stage2_vqa.json \
    --features_dir vqa/data/SlideChat/Feat \
    --output_dir outputs/slidechat_stage2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_epochs 5 \
    --lr 1e-5
```

### Benchmark Evaluation

Evaluate trained model on SlideBench (TCGA benchmark):

```bash
python vqa/tools/test_benchmark.py \
    --model_path outputs/stage2/final \
    --moe_checkpoint /path/to/moe_best.pt \
    --llm_path Qwen/Qwen3-4B-Instruct-2507 \
    --benchmark_path data/slidebench_test.json \
    --output_path results/benchmark_results.json \
    --batch_size 8
```

**Outputs:**
- `benchmark_results.json`: Full predictions and metrics
- `benchmark_results_summary.csv`: Summary table
- Console: Per-category accuracy breakdown

### Inference Example

```python
import torch
from vqa.src.vqa_model import MoE_Qwen_VQA

# Load model
model = MoE_Qwen_VQA(
    moe_checkpoint='checkpoints/moe_best.pt',
    llm_path='Qwen/Qwen3-4B-Instruct-2507',
    num_visual_tokens=16
)
model.load_pretrained('outputs/stage2/final')
model = model.cuda().eval()

# Prepare input
question = "What is the primary diagnosis?"
prompt = f"<image> {question}"

inputs = model.tokenizer(prompt, return_tensors='pt').to('cuda')
patch_features = torch.load('path/to/features.pt').unsqueeze(0).cuda()

# Generate answer
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        patch_features=patch_features,
        max_new_tokens=128,
        temperature=0.7
    )

answer = model.tokenizer.decode(
    output_ids[0][inputs['input_ids'].shape[1]:],
    skip_special_tokens=True
)
print(f"Answer: {answer}")
```

## Key Implementation Details

### Data Format Requirements

**Caption Data (Stage 1) - CSV:**
```csv
slide_id,caption,features_path
TCGA-XX-XXXX,"This is a lung adenocarcinoma slide showing...",/path/to/features.pt
```

**VQA Data (Stage 2) - JSON:**
```json
[
  {
    "slide_id": "TCGA-XX-XXXX",
    "question": "What is the primary diagnosis?",
    "answer": "Adenocarcinoma",
    "features_path": "/path/to/features.pt"
  }
]
```

**Benchmark Data (Evaluation) - JSON:**
```json
{
  "samples": [
    {
      "slide_id": "TCGA-XX-XXXX",
      "question": "What is the primary diagnosis?",
      "choices": ["A) Adenocarcinoma", "B) Squamous cell carcinoma", "C) Small cell", "D) Large cell"],
      "answer": "A",
      "category": "Diagnosis",
      "features_path": "/path/to/features.pt"
    }
  ]
}
```

**SlideChat Data Format:**
- JSON with `conversations` structure
- Image paths reference CSV files (e.g., `"./BLCA/TCGA-XXX.csv"`)
- Features can be .pt, .npy, or .csv format
- ConCH features (512-dim) automatically expanded to 1024-dim

### Feature Format

Patch features should be:
- Format: `.pt`, `.pth`, or `.npy` files
- Shape: `[N, 1024]` where N = number of patches
- Dtype: `float32`
- Source: Pre-extracted from WSI using UNI, ResNet, ConCH, etc.

### DDP (Distributed Data Parallel) Support

The training scripts automatically detect DDP environment:
- Single GPU: Run script directly with `python`
- Multi-GPU: Use `torchrun --nproc_per_node=N`
- Rank 0 handles all logging and checkpoint saving
- Gradient accumulation simulates larger batch sizes

**Effective Batch Size** = `batch_size × num_gpus × gradient_accumulation_steps`

Example: `4 × 2 × 4 = 32` effective batch size

### Memory Optimization

**Stage 1** (Lower memory):
- LLM frozen (no gradients)
- Only Projector trained (~8M parameters)
- Can use larger batch sizes

**Stage 2** (Higher memory):
- LLM unfrozen (full gradients for 4B parameters)
- Reduce batch size (typically 2 per GPU)
- Increase gradient accumulation to compensate

**BFloat16**: LLM loaded in `torch.bfloat16` to reduce memory

### Image Token Handling

- Special token `<image>` added to tokenizer vocabulary
- `prepare_inputs_embeds()` replaces `<image>` token with visual embeddings
- Visual embeddings inserted at token position, maintaining sequence structure
- Original sequence length preserved by truncating excess tokens

### Label Masking

Training only computes loss on answer tokens:
- Prompt tokens (including question): masked with `-100`
- Padding tokens: masked with `-100`
- Answer tokens: compute cross-entropy loss

This prevents the model from learning to copy the question.

### Checkpoint Structure

**Saved files:**
- `projector.pt`: MLP projector weights
- `llm/`: Full Qwen model weights (if finetuned)
- `llm/tokenizer_config.json`: Tokenizer with custom tokens

**Loading:**
```python
model.load_pretrained('outputs/stage2/final')
```

## File Structure

```
vqa/
├── src/
│   ├── vqa_model.py           # MoE_Qwen_VQA, MLPProjector
│   ├── vqa_dataset.py         # SlideChatDataset, BenchmarkDataset
│   └── slidechat_dataset.py   # SlideChat format adapter
├── tools/
│   ├── train_vqa.py           # Main training script (DDP)
│   ├── train_slidechat.py     # SlideChat-specific training
│   └── test_benchmark.py      # Benchmark evaluation
├── examples/
│   └── inference_example.py   # Inference code example
├── data/
│   ├── SlideChat/             # SlideChat dataset
│   ├── GTEx-TCGA-Embeddings/  # Precomputed features
│   └── Qwen3-4B-Instruct-2507/# LLM weights
├── README.md                  # Detailed documentation (Chinese)
└── requirements.txt           # Dependencies
```

## Hyperparameter Recommendations

### Stage 1 (Caption Pretraining)

- **Batch Size**: 4-8 per GPU
- **Learning Rate**: `1e-4`
- **Epochs**: 3-5
- **Warmup**: 10% of total steps
- **Gradient Accumulation**: 4-8 steps
- **Goal**: Projector learns basic visual-to-language mapping

### Stage 2 (VQA Finetuning)

- **Batch Size**: 2-4 per GPU (lower due to LLM gradients)
- **Learning Rate**: `1e-5` (10x smaller than Stage 1)
- **Epochs**: 5-10
- **Warmup**: 10% of total steps
- **Gradient Accumulation**: 8-16 steps
- **Goal**: LLM adapts to pathology VQA domain

### Hardware Requirements

- **Minimum**: 1× A6000 (48GB VRAM)
  - Stage 1: Batch=4, feasible
  - Stage 2: Batch=2, feasible with gradient accumulation

- **Recommended**: 2× A6000 (48GB each)
  - DDP training for faster convergence
  - Larger effective batch sizes

## Important Notes

### Always Freeze MoE_Compressor

The visual encoder is pretrained on classification and should remain frozen:
- Prevents VQA training from degrading visual representations
- Reduces memory and computation costs significantly
- Maintains modular architecture

### Never Skip Stage 1

Stage 1 pretraining is critical:
- Projector starts from random initialization
- Without Stage 1, LLM cannot understand visual features
- Skipping leads to slow convergence and poor performance

### Feature Dimension Handling

- UNI, ResNet features: 1024-dim (native)
- ConCH features: 512-dim (automatically expanded to 1024 via concatenation)
- MoE expects 1024-dim input

### Answer Extraction (Benchmarks)

The evaluation script uses regex to extract A/B/C/D answers:
- Looks for standalone letters: `\b([A-D])\b`
- Checks for "The answer is A" patterns
- Falls back to first character if it's A/B/C/D

Ensure generated answers are clear and unambiguous.

### Dataset-Specific Adapters

- `vqa_dataset.py`: Generic VQA data (direct feature paths)
- `slidechat_dataset.py`: SlideChat format (conversation-based, CSV features)

Use the appropriate dataset class for your data format.

## Expected Performance

Based on SlideChat architecture benchmarks:

| Benchmark | Metric | Expected Range |
|-----------|--------|----------------|
| SlideBench Overall | Accuracy | 65-75% |
| Diagnosis Category | Accuracy | 70-80% |
| Clinical Category | Accuracy | 60-70% |
| Microscopy Category | Accuracy | 65-75% |

Performance depends on:
- Quality of MoE pretraining
- Feature extractor (UNI > ResNet)
- Training data size and quality
- Hyperparameter tuning

## Dependencies

Install requirements:
```bash
pip install -r vqa/requirements.txt
```

Key dependencies:
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `accelerate>=0.24.0`
- `pandas`, `numpy`
- `wandb` (optional, for logging)
- `sentencepiece` (for tokenization)

## Troubleshooting

**OOM (Out of Memory)**:
- Reduce `batch_size`
- Increase `gradient_accumulation_steps`
- Ensure LLM is in `bfloat16`

**Slow Convergence**:
- Verify Stage 1 projector is loaded for Stage 2
- Check learning rate (Stage 2 should be 10x smaller)
- Ensure effective batch size is adequate (≥32)

**Feature Loading Errors**:
- Verify feature files exist at specified paths
- Check feature dimension (should be 1024 or 512)
- Ensure features are `float32` tensors

**DDP Issues**:
- Use `torchrun` instead of manual `torch.distributed.launch`
- Ensure `RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars are set
- Check that batch size is consistent across ranks
