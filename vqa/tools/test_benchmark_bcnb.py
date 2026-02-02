"""
Benchmark evaluation script for SlideBench-VQA-BCNB.

Usage:
cd /workspace/zhuo/ETC

CUDA_VISIBLE_DEVICES=0 python vqa/tools/test_benchmark_bcnb.py \
    --model_path /workspace/zhuo/ETC/vqa/outputs/slidechat_stage2_lora/epoch2_step1000 \
    --moe_checkpoint /workspace/zhuo/ETC/outputs/moe_tcga_32slots_top2_robust/best_model.pth \
    --stage1_moe_path /workspace/zhuo/ETC/vqa/outputs/slidechat_stage1_32slots_robust/final/moe_compressor.pt \
    --llm_path /workspace/jhsheng/huggingface/models/Qwen/Qwen2.5-7B-Instruct/ \
    --benchmark_path vqa/data/SlideChat/SlideBench-VQA-BCNB.csv \
    --features_dir vqa/data/SlideChat/Feat/BCNB_patch_feat/BCNB_patch_feat \
    --output_path vqa/results/benchmark_bcnb_generalization.json \
    --visual_dim 512 \
    --moe_num_slots 32 \
    --batch_size 24
"""

import argparse
import os
import sys
import json
import re
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqa.src.vqa_model import MoE_Qwen_VQA


class BCNBBenchmarkDataset(Dataset):
    """
    Dataset for BCNB benchmark evaluation.
    
    CSV columns: ID, Slide, Task, Question, A, B, C, D, Answer
    Features are in CSV format (each row is a 512-dim patch feature)
    """
    
    def __init__(self, csv_path, features_dir, tokenizer, visual_dim=512):
        self.features_dir = features_dir
        self.tokenizer = tokenizer
        self.visual_dim = visual_dim
        
        # Read benchmark CSV
        self.samples = []
        self.skipped = 0
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                slide_id = str(row['Slide'])
                feat_path = os.path.join(features_dir, f"{slide_id}.csv")
                
                if os.path.exists(feat_path):
                    self.samples.append({
                        'slide_id': slide_id,
                        'task': row['Task'],
                        'question': row['Question'],
                        'choices': {
                            'A': row['A'],
                            'B': row['B'],
                            'C': row['C'] if row['C'] else None,
                            'D': row['D'] if row['D'] else None
                        },
                        'answer': row['Answer'],
                        'feat_path': feat_path
                    })
                else:
                    self.skipped += 1
        
        print(f"Loaded {len(self.samples)} samples, skipped {self.skipped} (missing features)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Format choices (only include non-empty ones)
        choices = []
        for letter in ['A', 'B', 'C', 'D']:
            if sample['choices'][letter]:
                choices.append(f"{letter}) {sample['choices'][letter]}")
        choices_text = "\n".join(choices)
        
        # Create prompt
        user_prompt = f"<image> {sample['question']}\n{choices_text}\nAnswer with only the letter (A/B/C/D):"
        
        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": user_prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"User: {user_prompt}\nAssistant:"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=False
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Load features from CSV
        patch_features = self._load_features(sample['feat_path'])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'patch_features': patch_features,
            'slide_id': sample['slide_id'],
            'question': sample['question'],
            'choices': choices_text,
            'answer': sample['answer'],
            'task': sample['task'],
            'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        }
    
    def _load_features(self, feat_path):
        """Load patch features from CSV file."""
        features = []
        with open(feat_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header if exists
            for row in reader:
                # Each row is 512 floats + patch_name at the end
                feat_values = [float(x) for x in row[:512]]
                features.append(feat_values)
        
        if len(features) == 0:
            # Return dummy if no features
            return torch.zeros(1, self.visual_dim)
        
        return torch.tensor(features, dtype=torch.float32)


def collate_fn(batch):
    """Collate function for benchmark evaluation."""
    max_patches = max([item['patch_features'].shape[0] for item in batch])
    max_seq_len = max([item['input_ids'].shape[0] for item in batch])
    pad_token_id = batch[0]['pad_token_id']
    
    # Pad patch features
    padded_features = []
    for item in batch:
        features = item['patch_features']
        N = features.shape[0]
        if N < max_patches:
            padding = torch.zeros(max_patches - N, features.shape[1])
            features = torch.cat([features, padding], dim=0)
        padded_features.append(features)
    
    # Pad input sequences (left padding for generation)
    padded_input_ids = []
    padded_attention_mask = []
    
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        L = input_ids.shape[0]
        
        if L < max_seq_len:
            pad_len = max_seq_len - L
            input_ids = torch.cat([
                torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype),
                input_ids
            ])
            attention_mask = torch.cat([
                torch.zeros(pad_len, dtype=torch.long),
                attention_mask
            ])
        
        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'patch_features': torch.stack(padded_features),
        'slide_id': [item['slide_id'] for item in batch],
        'question': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'answer': [item['answer'] for item in batch],
        'task': [item['task'] for item in batch]
    }


def extract_answer(text):
    """Extract A/B/C/D from generated text."""
    text = text.strip()
    
    # Check if starts with a letter
    if text and text[0].upper() in 'ABCD':
        return text[0].upper()
    
    # Look for patterns like "A)", "A.", "A:"
    patterns = [
        r'^([ABCD])\)',
        r'^([ABCD])\.',
        r'^([ABCD]):',
        r'^([ABCD])\s',
        r'answer[:\s]+([ABCD])',
        r'([ABCD])\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    return 'Unknown'


def evaluate_model(model, dataloader, args):
    """Run evaluation on the benchmark."""
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    all_tasks = []
    all_slide_ids = []
    all_questions = []
    all_generated_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            patch_features = batch['patch_features'].cuda()
            
            # Generate responses
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_features=patch_features,
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.9,
                do_sample=False
            )
            
            # Decode generated text
            for i in range(len(generated_ids)):
                generated_text = model.tokenizer.decode(
                    generated_ids[i],
                    skip_special_tokens=True
                ).strip()
                
                # Debug: print first few
                if len(all_predictions) < 3:
                    print(f"  DEBUG: Raw output '{generated_text[:100]}...'")
                
                predicted_answer = extract_answer(generated_text)
                
                all_predictions.append(predicted_answer)
                all_ground_truth.append(batch['answer'][i])
                all_tasks.append(batch['task'][i])
                all_slide_ids.append(batch['slide_id'][i])
                all_questions.append(batch['question'][i])
                all_generated_texts.append(generated_text)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    correct = sum([1 for pred, gt in zip(all_predictions, all_ground_truth) if pred == gt])
    total = len(all_predictions)
    overall_accuracy = correct / total if total > 0 else 0
    
    # Per-task accuracy
    task_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, gt, task in zip(all_predictions, all_ground_truth, all_tasks):
        task_stats[task]['total'] += 1
        if pred == gt:
            task_stats[task]['correct'] += 1
    
    task_accuracies = {}
    for task, stats in task_stats.items():
        stats['accuracy'] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        task_accuracies[task] = stats['accuracy']
    
    results = {
        'overall_accuracy': overall_accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'task_accuracies': task_accuracies,
        'task_stats': dict(task_stats),
        'predictions': [
            {
                'slide_id': slide_id,
                'question': question,
                'predicted_answer': pred,
                'ground_truth': gt,
                'generated_text': gen_text,
                'task': task,
                'correct': pred == gt
            }
            for slide_id, question, pred, gt, gen_text, task in zip(
                all_slide_ids, all_questions, all_predictions,
                all_ground_truth, all_generated_texts, all_tasks
            )
        ]
    }
    
    return results


def print_results(results):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("BCNB Generalization Benchmark Results")
    print("=" * 80)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2%} ({results['correct_predictions']}/{results['total_samples']})")
    
    print("\nPer-Task Accuracy:")
    for task, acc in sorted(results['task_accuracies'].items()):
        stats = results['task_stats'][task]
        print(f"  {task}: {acc:.2%} ({stats['correct']}/{stats['total']})")
    
    print("=" * 80)


def save_results(results, output_path):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate WSI-VQA on BCNB Benchmark')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--moe_checkpoint', type=str, required=True)
    parser.add_argument('--llm_path', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--stage1_moe_path', type=str, default=None)
    
    parser.add_argument('--benchmark_path', type=str, required=True)
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--visual_dim', type=int, default=512)
    parser.add_argument('--moe_num_slots', type=int, default=32)
    parser.add_argument('--skip_lora', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("BCNB Generalization Benchmark Evaluation")
    print("=" * 80)
    print(f"Model Path: {args.model_path}")
    print(f"Benchmark Path: {args.benchmark_path}")
    print(f"Features Dir: {args.features_dir}")
    print("=" * 80)
    
    # Initialize model
    print("\nLoading model...")
    
    model = MoE_Qwen_VQA(
        moe_checkpoint=args.moe_checkpoint,
        llm_path=args.llm_path,
        num_visual_tokens=args.moe_num_slots,
        visual_dim=args.visual_dim,
        moe_num_slots=args.moe_num_slots,
        device='cuda'
    )
    
    # Load checkpoints
    projector_path = os.path.join(args.model_path, "projector.pt")
    if os.path.exists(projector_path):
        model.projector.load_state_dict(torch.load(projector_path, map_location='cpu'))
        print(f"Loaded projector from {projector_path}")
    
    if args.stage1_moe_path and os.path.exists(args.stage1_moe_path):
        model.visual_encoder.load_state_dict(torch.load(args.stage1_moe_path, map_location='cpu'))
        print(f"Loaded Stage 1 MoE from {args.stage1_moe_path}")
    else:
        moe_in_model = os.path.join(args.model_path, "moe_compressor.pt")
        if os.path.exists(moe_in_model):
            model.visual_encoder.load_state_dict(torch.load(moe_in_model, map_location='cpu'))
            print(f"Loaded MoE from {moe_in_model}")
    
    # Load LoRA
    lora_path = os.path.join(args.model_path, "lora_adapter")
    if not args.skip_lora and os.path.exists(lora_path):
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")
    
    model = model.cuda()
    model.eval()
    
    # Create dataset
    print("\nLoading BCNB benchmark dataset...")
    
    dataset = BCNBBenchmarkDataset(
        csv_path=args.benchmark_path,
        features_dir=args.features_dir,
        tokenizer=model.tokenizer,
        visual_dim=args.visual_dim
    )
    
    if len(dataset) == 0:
        print("ERROR: No samples found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # Evaluate
    results = evaluate_model(model, dataloader, args)
    
    # Print and save
    print_results(results)
    save_results(results, args.output_path)
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
