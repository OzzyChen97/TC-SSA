"""
Benchmark evaluation script for SlideBench (TCGA benchmark) with CSV format.

Evaluates VQA model on the benchmark test set and computes accuracy metrics.
Supports loading features from GTEx-TCGA-Embeddings directory structure.

Usage:
cd /workspace/zhuo/ETC

CUDA_VISIBLE_DEVICES=0 python vqa/tools/test_benchmark_csv.py \
    --model_path /workspace/zhuo/ETC/vqa/outputs/slidechat_stage2_lora/epoch2_step1000 \
    --moe_checkpoint /workspace/zhuo/ETC/outputs/moe_tcga_32slots_top2_robust/best_model.pth \
    --stage1_moe_path /workspace/zhuo/ETC/vqa/outputs/slidechat_stage1_32slots_robust/final/moe_compressor.pt \
    --llm_path /workspace/jhsheng/huggingface/models/Qwen/Qwen2.5-7B-Instruct/ \
    --benchmark_path vqa/data/SlideChat/SlideBench-VQA-TCGA.csv \
    --features_dir vqa/data/GTEx-TCGA-Embeddings \
    --output_path vqa/results/benchmark_7B_more_patches.json \
    --visual_dim 512 \
    --feature_suffix 1024 \
    --num_visual_tokens 32 \
    --batch_size 16
"""

import argparse
import os
import sys
import json
import re
from collections import defaultdict
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqa.src.vqa_model import MoE_Qwen_VQA


def find_feature_file(slide_id, features_dir, feature_suffix=1024):
    """
    Find the feature file for a given slide ID.
    
    Supports multiple directory structures:
    1. GTEx-TCGA-Embeddings: features_dir/TCGA-XXX/TCGA-XXX/slide_id.*_N_suffix.npy
    2. SlideChat/Feat: features_dir/TCGA_XXX_feat_conch/slide_id.csv
    
    Args:
        slide_id: Slide ID like 'TCGA-05-4244-01Z-00-DX1'
        features_dir: Base directory for features
        feature_suffix: Preferred suffix (1024 or 512), will fallback to alternative if not found
    
    Returns:
        Path to feature file or None if not found
    """
    subdirs = ['TCGA-LUNG', 'TCGA-GBM', 'TCGA-BR', 'TCGA-BLCA', 'TCGA-COAD', 
               'TCGA-HNSC', 'TCGA-LGG', 'TCGA-SKCM', 'TCGA-Rest']
    
    # Try preferred suffix first, then fallback to alternative
    suffixes_to_try = [feature_suffix]
    if feature_suffix == 1024:
        suffixes_to_try.append(512)  # Fallback to 512
    elif feature_suffix == 512:
        suffixes_to_try.append(1024)  # Fallback to 1024
    
    for suffix in suffixes_to_try:
        # Method 1: GTEx-TCGA-Embeddings format (*.npy) - specific subdirs
        for subdir in subdirs:
            # Pattern: slide_id.*_0_1024.npy or slide_id.*_1_1024.npy
            pattern = os.path.join(features_dir, subdir, subdir, f"{slide_id}.*_*_{suffix}.npy")
            matches = glob.glob(pattern)
            
            if matches:
                # Prefer _0_ version (usually the first/main embedding with more patches)
                for m in matches:
                    if '_0_' in m:
                        return m
                return matches[0]
        
        # Also try direct pattern for GTEx format (any subdir)
        pattern = os.path.join(features_dir, "*", "*", f"{slide_id}.*_*_{suffix}.npy")
        matches = glob.glob(pattern)
        if matches:
            for m in matches:
                if '_0_' in m:
                    return m
            return matches[0]
    
    # Method 2: SlideChat/Feat format (*.csv) - TCGA_XXX_feat_conch/slide_id.csv
    feat_dirs_pattern = os.path.join(features_dir, "TCGA_*_feat_*")
    feat_dirs = glob.glob(feat_dirs_pattern)
    
    for feat_dir in feat_dirs:
        csv_path = os.path.join(feat_dir, f"{slide_id}.csv")
        if os.path.exists(csv_path):
            return csv_path
    
    # Method 3: Direct file in features_dir (any format)
    for ext in ['.csv', '.npy', '.pt', '.pth']:
        direct_path = os.path.join(features_dir, f"{slide_id}{ext}")
        if os.path.exists(direct_path):
            return direct_path
    
    return None


class BenchmarkDatasetCSV(Dataset):
    """
    Dataset for SlideBench evaluation from CSV format.
    
    CSV columns: ID, Slide, Tumor, Broad Category, Narrow Category, Question, A, B, C, D, Answer
    """
    
    def __init__(self, csv_path, features_dir, tokenizer, visual_dim=512, num_visual_tokens=16, feature_suffix=1024):
        """
        Args:
            csv_path: Path to benchmark CSV file
            features_dir: Base directory for features
            tokenizer: HuggingFace tokenizer
            visual_dim: Feature dimension for model (512)
            num_visual_tokens: Number of visual tokens
            feature_suffix: Suffix in filename (1024 for more patches, 512 for fewer)
        """
        self.tokenizer = tokenizer
        self.features_dir = features_dir
        self.visual_dim = visual_dim
        self.num_visual_tokens = num_visual_tokens
        self.feature_suffix = feature_suffix
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} benchmark samples from {csv_path}")
        
        # Build feature path mapping
        self.feature_paths = {}
        missing_count = 0
        
        print(f"Building feature path mapping (using *_{feature_suffix}.npy files)...")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Mapping features"):
            slide_id = row['Slide']
            feat_path = find_feature_file(slide_id, features_dir, feature_suffix)
            if feat_path:
                self.feature_paths[slide_id] = feat_path
            else:
                missing_count += 1
        
        print(f"Found features for {len(self.feature_paths)}/{len(self.df)} slides")
        print(f"Missing features: {missing_count}")
        
        # Filter to only samples with features
        self.df = self.df[self.df['Slide'].isin(self.feature_paths.keys())].reset_index(drop=True)
        print(f"Using {len(self.df)} samples with available features")
        
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get benchmark sample for evaluation.
        """
        row = self.df.iloc[idx]
        
        slide_id = row['Slide']
        question = row['Question']
        choices = [
            f"A) {row['A']}",
            f"B) {row['B']}",
            f"C) {row['C']}",
            f"D) {row['D']}"
        ]
        answer = row['Answer']  # 'A', 'B', 'C', or 'D'
        category = f"{row['Broad Category']}/{row['Narrow Category']}"
        
        # Format multiple choice prompt - SlideChat style
        choices_text = "\n".join(choices)
        user_content = f"<image> {question}\n{choices_text}"
        
        # Use chat template for consistent format with training
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": user_content}
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True  # Add assistant prompt for generation
            )
        else:
            prompt = f"User: {user_content}\nAssistant:"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=False
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Load features - support multiple formats
        feat_path = self.feature_paths[slide_id]
        
        if feat_path.endswith('.csv'):
            # CSV format (SlideChat/Feat) - each row is a patch, columns are features
            df = pd.read_csv(feat_path, header=None)
            # Remove last column if it's a string (patch filename)
            if df.iloc[:, -1].dtype == object:
                df = df.iloc[:, :-1]
            patch_features = torch.from_numpy(df.values.astype(np.float32))
        elif feat_path.endswith('.pt') or feat_path.endswith('.pth'):
            patch_features = torch.load(feat_path, map_location='cpu')
        elif feat_path.endswith('.npy'):
            data = np.load(feat_path, allow_pickle=True)
            
            # Handle dict format (common in GTEx-TCGA-Embeddings)
            if data.dtype == np.object_:
                data = data.item()
                if isinstance(data, dict) and 'feature' in data:
                    patch_features = data['feature']
                    # Convert to tensor if not already
                    if not isinstance(patch_features, torch.Tensor):
                        patch_features = torch.from_numpy(patch_features)
                else:
                    raise ValueError(f"Unexpected data format in {feat_path}")
            else:
                patch_features = torch.from_numpy(data)
        else:
            raise ValueError(f"Unsupported feature format: {feat_path}")
        
        if patch_features.dtype != torch.float32:
            patch_features = patch_features.float()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'patch_features': patch_features,
            'slide_id': slide_id,
            'question': question,
            'choices': choices,
            'answer': answer,
            'category': category,
            'pad_token_id': self.pad_token_id
        }


def benchmark_collate_fn(batch):
    """
    Collate function for benchmark evaluation.
    """
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
        'category': [item['category'] for item in batch]
    }


def extract_answer(text):
    """
    Extract answer choice (A/B/C/D) from generated text.
    Improved logic with multiple patterns.
    """
    # Clean the text - remove repetitions and newlines
    text = text.strip()
    
    # Handle repetitions like "C\nAnswer with only..." - take first line
    first_line = text.split('\n')[0].strip()
    
    # Try to extract from first line first
    text_upper = first_line.upper()
    
    # Pattern 1: Single letter response "A", "B", "C", "D"
    if len(first_line) <= 3:
        for char in first_line.upper():
            if char in ['A', 'B', 'C', 'D']:
                return char
    
    # Pattern 2: Letter with punctuation "A." "A)" "A:"
    match = re.match(r'^([A-D])[\.):\s]', text_upper)
    if match:
        return match.group(1)
    
    # Pattern 3: "The answer is A" or "Answer: A"
    match = re.search(r'(?:ANSWER\s*(?:IS)?[:\s]*)([A-D])', text_upper)
    if match:
        return match.group(1)
    
    # Pattern 4: "Option A" or "Choice A"
    match = re.search(r'(?:OPTION|CHOICE)\s*([A-D])', text_upper)
    if match:
        return match.group(1)
    
    # Pattern 5: First standalone A/B/C/D in text
    match = re.search(r'\b([A-D])\b', text_upper)
    if match:
        return match.group(1)
    
    # Pattern 6: First character if it's A/B/C/D
    if len(text_upper) > 0 and text_upper[0] in ['A', 'B', 'C', 'D']:
        return text_upper[0]
    
    return None


def evaluate_model(model, dataloader, args):
    """
    Run inference and compute accuracy metrics.
    """
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    all_categories = []
    all_slide_ids = []
    all_questions = []
    all_generated_texts = []
    
    print("\nRunning inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to GPU
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            patch_features = batch['patch_features'].cuda()
            
            # Generate responses - fix repetition with proper parameters
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_features=patch_features,
                max_new_tokens=16,  # Reduced - we only need A/B/C/D
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                repetition_penalty=1.2  # Prevent repetition
            )
            
            # Decode generated text
            # Note: When using inputs_embeds, generated_ids only contains new tokens
            for i in range(len(generated_ids)):
                # The model generates only new tokens when using inputs_embeds
                generated_text = model.tokenizer.decode(
                    generated_ids[i],
                    skip_special_tokens=True
                ).strip()
                
                # Debug: print first few to check
                if len(all_predictions) < 3:
                    print(f"  DEBUG: Raw output '{generated_text[:100]}...' (len={len(generated_ids[i])})")
                
                predicted_answer = extract_answer(generated_text)
                
                all_predictions.append(predicted_answer)
                all_ground_truth.append(batch['answer'][i])
                all_categories.append(batch['category'][i])
                all_slide_ids.append(batch['slide_id'][i])
                all_questions.append(batch['question'][i])
                all_generated_texts.append(generated_text)
    
    # Compute metrics
    print("\nComputing metrics...")
    
    correct = sum([
        1 for pred, gt in zip(all_predictions, all_ground_truth)
        if pred == gt
    ])
    total = len(all_predictions)
    overall_accuracy = correct / total if total > 0 else 0
    
    # Per-category accuracy (narrow: Diagnosis/Grading, etc.)
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # Broad category accuracy (Microscopy, Diagnosis, Clinical)
    broad_category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, gt, cat in zip(all_predictions, all_ground_truth, all_categories):
        category_stats[cat]['total'] += 1
        if pred == gt:
            category_stats[cat]['correct'] += 1
        
        # Extract broad category (first part before '/')
        broad_cat = cat.split('/')[0] if '/' in cat else cat
        broad_category_stats[broad_cat]['total'] += 1
        if pred == gt:
            broad_category_stats[broad_cat]['correct'] += 1
    
    category_accuracies = {
        cat: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        for cat, stats in category_stats.items()
    }
    
    broad_category_accuracies = {
        cat: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        for cat, stats in broad_category_stats.items()
    }
    
    results = {
        'overall_accuracy': overall_accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        # Broad category stats (Microscopy, Diagnosis, Clinical)
        'broad_category_accuracies': broad_category_accuracies,
        'broad_category_stats': {
            cat: {
                'accuracy': acc,
                'correct': broad_category_stats[cat]['correct'],
                'total': broad_category_stats[cat]['total']
            }
            for cat, acc in broad_category_accuracies.items()
        },
        # Narrow category stats
        'category_accuracies': category_accuracies,
        'category_stats': {
            cat: {
                'accuracy': acc,
                'correct': category_stats[cat]['correct'],
                'total': category_stats[cat]['total']
            }
            for cat, acc in category_accuracies.items()
        },
        'predictions': [
            {
                'slide_id': slide_id,
                'question': question,
                'predicted_answer': pred,
                'ground_truth': gt,
                'generated_text': gen_text,
                'category': cat,
                'correct': pred == gt
            }
            for slide_id, question, pred, gt, gen_text, cat in zip(
                all_slide_ids, all_questions, all_predictions,
                all_ground_truth, all_generated_texts, all_categories
            )
        ]
    }
    
    return results


def print_results(results):
    """Pretty print evaluation results."""
    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    
    # Broad category stats (Microscopy, Diagnosis, Clinical)
    print("\n" + "-" * 80)
    print("BROAD CATEGORY ACCURACY (Microscopy / Diagnosis / Clinical):")
    print("-" * 80)
    
    print(f"{'Category':<20} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-" * 80)
    
    for cat in ['Microscopy', 'Diagnosis', 'Clinical']:
        if cat in results.get('broad_category_stats', {}):
            stats = results['broad_category_stats'][cat]
            acc = stats['accuracy']
            correct = stats['correct']
            total = stats['total']
            print(f"{cat:<20} {acc:>7.2%}         {correct:>4}/{total:<4}")
    
    print("=" * 80)
    
    # Narrow category stats
    print("\n" + "-" * 80)
    print("Per-Category Accuracy (Narrow):")
    print("-" * 80)
    
    print(f"{'Category':<40} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-" * 80)
    
    for cat, stats in sorted(results['category_stats'].items()):
        acc = stats['accuracy']
        correct = stats['correct']
        total = stats['total']
        print(f"{cat:<40} {acc:>7.2%}         {correct:>4}/{total:<4}")
    
    print("=" * 80)
    
    # Sample predictions
    print("\nSample Predictions (First 5):")
    print("-" * 80)
    
    for i, pred in enumerate(results['predictions'][:5]):
        print(f"\n[{i+1}] Slide: {pred['slide_id']}")
        print(f"    Question: {pred['question'][:80]}...")
        print(f"    Predicted: {pred['predicted_answer']} | Ground Truth: {pred['ground_truth']}")
        print(f"    Generated: {pred['generated_text'][:100]}...")
        print(f"    Category: {pred['category']} | Correct: {pred['correct']}")
    
    print("\n" + "=" * 80)


def save_results(results, output_path):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Also save a summary CSV
    summary_path = output_path.replace('.json', '_summary.csv')
    
    summary_data = {
        'Metric': ['Overall Accuracy', 'Total Samples', 'Correct Predictions'],
        'Value': [
            f"{results['overall_accuracy']:.2%}",
            results['total_samples'],
            results['correct_predictions']
        ]
    }
    
    # Add broad category stats first (Microscopy, Diagnosis, Clinical)
    summary_data['Metric'].append('')  # Empty row separator
    summary_data['Value'].append('')
    summary_data['Metric'].append('=== BROAD CATEGORIES ===')
    summary_data['Value'].append('')
    
    for cat in ['Microscopy', 'Diagnosis', 'Clinical']:
        if cat in results.get('broad_category_stats', {}):
            stats = results['broad_category_stats'][cat]
            summary_data['Metric'].append(f"{cat} Accuracy")
            summary_data['Value'].append(f"{stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # Add narrow category stats
    summary_data['Metric'].append('')  # Empty row separator
    summary_data['Value'].append('')
    summary_data['Metric'].append('=== NARROW CATEGORIES ===')
    summary_data['Value'].append('')
    
    for cat, stats in sorted(results['category_stats'].items()):
        summary_data['Metric'].append(f"{cat} Accuracy")
        summary_data['Value'].append(f"{stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_path, index=False)
    
    print(f"Summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate WSI-VQA on Benchmark (CSV format)')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint directory')
    parser.add_argument('--moe_checkpoint', type=str, required=True,
                       help='Path to MoE_Compressor checkpoint')
    parser.add_argument('--llm_path', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Path to base Qwen LLM')
    parser.add_argument('--stage1_moe_path', type=str, default=None,
                       help='Path to Stage 1 finetuned MoE checkpoint (moe_compressor.pt)')
    
    # Data arguments
    parser.add_argument('--benchmark_path', type=str, required=True,
                       help='Path to benchmark CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Base directory for feature files')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save results JSON')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--num_visual_tokens', type=int, default=16,
                       help='Number of visual tokens')
    parser.add_argument('--visual_dim', type=int, default=512,
                       help='Visual feature dimension for model (default 512)')
    parser.add_argument('--feature_suffix', type=int, default=1024,
                       help='Suffix in feature filename (1024=more patches, 512=fewer patches)')
    parser.add_argument('--skip_lora', action='store_true',
                       help='Skip loading LoRA adapter (use original LLM)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("WSI-VQA Benchmark Evaluation (CSV Format)")
    print("=" * 80)
    print(f"Model Path: {args.model_path}")
    print(f"MoE Checkpoint: {args.moe_checkpoint}")
    print(f"Benchmark Path: {args.benchmark_path}")
    print(f"Features Dir: {args.features_dir}")
    print(f"Output Path: {args.output_path}")
    print(f"Visual Dim: {args.visual_dim}")
    print("=" * 80)
    
    # Initialize model
    print("\nLoading model...")
    
    model = MoE_Qwen_VQA(
        moe_checkpoint=args.moe_checkpoint,
        llm_path=args.llm_path,
        num_visual_tokens=args.num_visual_tokens,
        visual_dim=args.visual_dim,
        device='cuda'
    )
    
    # Load trained weights (LoRA adapter + projector)
    print(f"Loading checkpoint from {args.model_path}")
    
    # Load projector
    projector_path = os.path.join(args.model_path, "projector.pt")
    if os.path.exists(projector_path):
        model.projector.load_state_dict(torch.load(projector_path, map_location='cpu'))
        print(f"Loaded projector from {projector_path}")
    
    # Load Stage 1 finetuned MoE if available (check for moe_compressor.pt in model_path or stage1_dir)
    stage1_moe_path = getattr(args, 'stage1_moe_path', None)
    if stage1_moe_path and os.path.exists(stage1_moe_path):
        moe_state_dict = torch.load(stage1_moe_path, map_location='cpu')
        model.visual_encoder.load_state_dict(moe_state_dict)
        print(f"Loaded Stage 1 finetuned MoE from {stage1_moe_path}")
    else:
        # Try to find moe_compressor.pt in model_path
        moe_in_model_path = os.path.join(args.model_path, "moe_compressor.pt")
        if os.path.exists(moe_in_model_path):
            moe_state_dict = torch.load(moe_in_model_path, map_location='cpu')
            model.visual_encoder.load_state_dict(moe_state_dict)
            print(f"Loaded MoE from {moe_in_model_path}")
    
    # Load LoRA adapter (unless --skip_lora is set)
    skip_lora = getattr(args, 'skip_lora', False)
    lora_path = os.path.join(args.model_path, "lora_adapter")
    if not skip_lora and os.path.exists(lora_path):
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")
    elif skip_lora:
        print("Skipping LoRA adapter loading (using original LLM)")
    
    model = model.cuda()
    model.eval()
    
    # Create dataset
    print("\nLoading benchmark dataset...")
    
    dataset = BenchmarkDatasetCSV(
        csv_path=args.benchmark_path,
        features_dir=args.features_dir,
        tokenizer=model.tokenizer,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        feature_suffix=args.feature_suffix
    )
    
    if len(dataset) == 0:
        print("ERROR: No samples with available features found!")
        return
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=benchmark_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Run evaluation
    results = evaluate_model(model, dataloader, args)
    
    # Print results
    print_results(results)
    
    # Save results
    save_results(results, args.output_path)
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
