# WSI-VQA: Vision-Language Model for Pathology Slides

基于 MoE-Compressor 和 Qwen3-4B 的高效病理图像问答系统。

## 架构设计

```
WSI Patches [B, N, 1024]
    ↓
MoE_Compressor (Frozen)
    ↓
Visual Tokens [B, 16, 1024]
    ↓
Projector (Trainable)
    ↓
Visual Embeddings [B, 16, hidden_size]
    ↓
Qwen3-4B-Instruct + Text Embeddings
    ↓
Generated Response
```

### 核心组件

1. **Visual Encoder**: MoE_Compressor (预训练+冻结)
   - 将可变数量的 patches 压缩成 16 个语义 tokens
   - 全程冻结，保持特征提取能力

2. **Projector**: MLP (可训练)
   - 将视觉特征 (1024维) 对齐到 LLM 空间 (hidden_size维)
   - Stage 1: 独立训练
   - Stage 2: 与 LLM 联合微调

3. **LLM**: Qwen3-4B-Instruct
   - Stage 1: 冻结
   - Stage 2: 全量微调

## 两阶段训练策略

### Stage 1: Caption Pretraining (仅训 Projector)

**目的**: 让 Projector 学会将视觉特征映射到 LLM 的语义空间

```bash
# 单卡训练
python vqa/tools/train_vqa.py \
    --stage 1 \
    --moe_checkpoint /path/to/moe_best.pt \
    --llm_path Qwen/Qwen3-4B-Instruct-2507 \
    --data_path data/captions_train.csv \
    --output_dir outputs/stage1 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3 \
    --lr 1e-4 \
    --wandb_project wsi-vqa

# 多卡 DDP 训练 (推荐)
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

**数据格式 (CSV)**:
```csv
slide_id,caption,features_path
TCGA-XX-XXXX,This is a lung adenocarcinoma slide showing...,/path/to/features.pt
```

### Stage 2: VQA Finetuning (全量微调)

**目的**: 在 VQA 数据上微调整个模型

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

**数据格式 (JSON)**:
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

## 基准测试

在 SlideBench (TCGA) 上评测模型性能:

```bash
python vqa/tools/test_benchmark.py \
    --model_path outputs/stage2/final \
    --moe_checkpoint /path/to/moe_best.pt \
    --llm_path Qwen/Qwen3-4B-Instruct-2507 \
    --benchmark_path data/slidebench_test.json \
    --output_path results/benchmark_results.json \
    --batch_size 8
```

**Benchmark 数据格式**:
```json
{
  "samples": [
    {
      "slide_id": "TCGA-XX-XXXX",
      "question": "What is the primary diagnosis?",
      "choices": [
        "A) Adenocarcinoma",
        "B) Squamous cell carcinoma",
        "C) Small cell carcinoma",
        "D) Large cell carcinoma"
      ],
      "answer": "A",
      "category": "Diagnosis",
      "features_path": "/path/to/features.pt"
    }
  ]
}
```

**输出指标**:
- Overall Accuracy
- Per-Category Accuracy (Diagnosis, Clinical, Microscopy, etc.)
- 详细的预测结果 JSON
- 汇总表格 CSV

## 文件结构

```
vqa/
├── src/
│   ├── __init__.py
│   ├── vqa_model.py          # MoE_Qwen_VQA 模型定义
│   └── vqa_dataset.py        # 数据集和 DataLoader
├── tools/
│   ├── __init__.py
│   ├── train_vqa.py          # DDP 训练脚本
│   └── test_benchmark.py     # 基准测试脚本
└── README.md                 # 本文档
```

## 关键特性

### 1. DDP 支持
- 自动检测多卡环境
- 单卡和多卡无缝切换
- 仅在 Rank 0 打印日志和保存 checkpoint

### 2. 梯度累积
- 模拟大 Batch Size
- 有效 Batch Size = `batch_size × num_gpus × gradient_accumulation_steps`
- 示例: 4 × 2 × 4 = 32

### 3. 内存优化
- Visual Encoder 始终冻结 (`no_grad`)
- Stage 1 冻结 LLM，仅训 Projector
- 使用 `bfloat16` 降低 LLM 显存

### 4. 灵活的 Prompt
- 自动插入 `<image>` token
- 支持 Qwen 的 chat template
- 可自定义对话格式

## 训练建议

### Stage 1 超参数
- Batch Size: 4-8 per GPU
- Learning Rate: 1e-4
- Epochs: 3-5
- Warmup: 10%
- 目标: 让 Projector 学会基本的视觉-语言对齐

### Stage 2 超参数
- Batch Size: 2-4 per GPU (因为要训 LLM)
- Learning Rate: 1e-5 (比 Stage 1 小一个数量级)
- Epochs: 5-10
- Warmup: 10%
- 目标: 微调 LLM 使其适应病理 VQA 任务

### 硬件需求
- **最低**: 1x A6000 (48GB)
  - Stage 1: Batch=4, 可训练
  - Stage 2: Batch=2, 可训练

- **推荐**: 2x A6000 (48GB each)
  - DDP 加速训练
  - 更大的有效 Batch Size

## 推理示例

```python
import torch
from vqa.src.vqa_model import MoE_Qwen_VQA

# 加载模型
model = MoE_Qwen_VQA(
    moe_checkpoint='checkpoints/moe_best.pt',
    llm_path='Qwen/Qwen3-4B-Instruct-2507',
    num_visual_tokens=16
)
model.load_pretrained('outputs/stage2/final')
model = model.cuda().eval()

# 准备输入
question = "What is the primary diagnosis?"
prompt = f"<image> {question}"

inputs = model.tokenizer(prompt, return_tensors='pt').to('cuda')
patch_features = torch.load('path/to/features.pt').unsqueeze(0).cuda()

# 生成回答
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        patch_features=patch_features,
        max_new_tokens=128,
        temperature=0.7
    )

# 解码
answer = model.tokenizer.decode(
    output_ids[0][inputs['input_ids'].shape[1]:],
    skip_special_tokens=True
)
print(f"Answer: {answer}")
```

## FAQ

### Q1: 为什么使用 16 个视觉 tokens?
**A**: 平衡效率和性能。16 tokens 足以表达 WSI 的关键语义信息，同时保持极高的计算效率。

### Q2: 为什么冻结 MoE_Compressor?
**A**: MoE 已经在分类任务上预训练，具备强大的特征提取能力。冻结它可以:
- 防止 VQA 训练破坏已学到的表征
- 大幅降低显存和计算成本
- 保持模块化，便于复用

### Q3: Stage 1 可以跳过吗?
**A**: 不建议。Stage 1 让 Projector 学会视觉-语言对齐，是 Stage 2 的基础。直接 Stage 2 会导致收敛慢、性能差。

### Q4: 如何处理超长 WSI?
**A**: MoE_Compressor 会自动将任意数量的 patches 压缩成固定的 16 tokens，无论输入多长，输出都是固定长度。

### Q5: 可以使用其他 LLM 吗?
**A**: 可以。只需修改 `llm_path` 和 Projector 的 `output_dim` (匹配 LLM 的 hidden_size)。Qwen 系列效果较好。

## 性能预期

基于类似架构 (SlideChat) 的结果，预期性能:

| Benchmark | Metric | Expected |
|-----------|--------|----------|
| SlideBench Overall | Accuracy | 65-75% |
| Diagnosis | Accuracy | 70-80% |
| Clinical | Accuracy | 60-70% |
| Microscopy | Accuracy | 65-75% |

*注: 具体性能取决于数据质量、训练超参数等。*

## Citation

如果使用本代码，请引用:

```bibtex
@article{moe-compressor-vqa,
  title={Efficient WSI-VQA with MoE Token Compression},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
