"""
WsiVQA* Test script - Only evaluates closed-set (multiple choice) VQA.

This matches the WSI-VQA* benchmark in the SlideChat paper, which
"exclusively consists of closed-set VQA pairs from the public WSI-VQA dataset."

Usage:
cd /workspace/zhuo/ETC

CUDA_VISIBLE_DEVICES=0 python vqa/tools/test_wsivqa_closed.py \
    --model_path /workspace/zhuo/ETC/vqa/outputs/slidechat_stage2_lora/epoch2_step1000 \
    --moe_checkpoint /workspace/zhuo/ETC/outputs/moe_tcga_32slots_top2_robust/best_model.pth \
    --stage1_moe_path /workspace/zhuo/ETC/vqa/outputs/slidechat_stage1_32slots_robust/final/moe_compressor.pt \
    --llm_path /workspace/jhsheng/huggingface/models/Qwen/Qwen2.5-7B-Instruct/ \
    --test_path vqa/data/WsiVQA_test_cleaned.json \
    --features_dir vqa/data/GTEx-TCGA-Embeddings \
    --output_path vqa/results/wsivqa_closed_results.json \
    --visual_dim 512 \
    --feature_suffix 1024 \
    --num_visual_tokens 32 \
    --batch_size 32
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
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqa.src.vqa_model import MoE_Qwen_VQA


def find_feature_file(case_id, features_dir, feature_suffix=1024):
    """
    Find the feature file for a given case ID in the GTEx-TCGA-Embeddings directory.
    """
    subdirs = ['TCGA-LUNG', 'TCGA-GBM', 'TCGA-BR', 'TCGA-BLCA', 'TCGA-COAD', 
               'TCGA-HNSC', 'TCGA-LGG', 'TCGA-SKCM', 'TCGA-Rest']
    
    for subdir in subdirs:
        pattern = os.path.join(features_dir, subdir, subdir, f"{case_id}-*_*_{feature_suffix}.npy")
        matches = glob.glob(pattern)
        
        if matches:
            for m in matches:
                if '_0_' in m:
                    return m
            return matches[0]
    
    pattern = os.path.join(features_dir, "*", "*", f"{case_id}-*_*_{feature_suffix}.npy")
    matches = glob.glob(pattern)
    if matches:
        for m in matches:
            if '_0_' in m:
                return m
        return matches[0]
    
    return None


class WsiVQAClosedDataset(Dataset):
    """
    Dataset for WSI-VQA* evaluation - ONLY closed-set (multiple choice) questions.
    
    This matches the WSI-VQA* benchmark definition in the SlideChat paper.
    """
    
    def __init__(self, json_path, features_dir, tokenizer, visual_dim=512, num_visual_tokens=16, feature_suffix=1024):
        self.tokenizer = tokenizer
        self.features_dir = features_dir
        self.visual_dim = visual_dim
        self.num_visual_tokens = num_visual_tokens
        self.feature_suffix = feature_suffix
        
        # Load JSON
        with open(json_path, 'r') as f:
            all_data = json.load(f)
        print(f"Loaded {len(all_data)} total samples from {json_path}")
        
        # Filter to ONLY closed-set (multiple choice) questions
        self.data = [item for item in all_data if 'Choice' in item and item['Choice']]
        print(f"Filtered to {len(self.data)} closed-set (multiple choice) samples")
        
        # Build feature path mapping
        self.feature_paths = {}
        missing_case_ids = set()
        
        print(f"Building feature path mapping...")
        unique_case_ids = set(item['Id'] for item in self.data)
        
        for case_id in tqdm(unique_case_ids, desc="Mapping features"):
            feat_path = find_feature_file(case_id, features_dir, feature_suffix)
            if feat_path:
                self.feature_paths[case_id] = feat_path
            else:
                missing_case_ids.add(case_id)
        
        print(f"Found features for {len(self.feature_paths)}/{len(unique_case_ids)} case IDs")
        
        # Filter to samples with features
        self.data = [item for item in self.data if item['Id'] in self.feature_paths]
        print(f"Final: {len(self.data)} closed-set samples with available features")
        
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        case_id = item['Id']
        question = item['Question']
        answer = item['Answer']
        choices = item['Choice']
        
        # Map choices to A, B, C, D
        choice_labels = ['A', 'B', 'C', 'D']
        choices_text = "\n".join([f"{label}) {choice}" for label, choice in zip(choice_labels, choices)])
        prompt = f"<image> {question}\n{choices_text}\nAnswer with only the letter (A/B/C/D):"
        
        # Find correct answer label
        try:
            correct_idx = choices.index(answer)
            answer_label = choice_labels[correct_idx]
        except ValueError:
            # Try case-insensitive match
            answer_label = None
            for i, choice in enumerate(choices):
                if choice.lower().strip() == answer.lower().strip():
                    answer_label = choice_labels[i]
                    break
            if answer_label is None:
                answer_label = 'UNKNOWN'
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=False
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Load features
        feat_path = self.feature_paths[case_id]
        data = np.load(feat_path, allow_pickle=True)
        
        if data.dtype == np.object_:
            data = data.item()
            if isinstance(data, dict) and 'feature' in data:
                patch_features = data['feature']
                if not isinstance(patch_features, torch.Tensor):
                    patch_features = torch.from_numpy(patch_features)
            else:
                raise ValueError(f"Unexpected data format in {feat_path}")
        else:
            patch_features = torch.from_numpy(data)
        
        if patch_features.dtype != torch.float32:
            patch_features = patch_features.float()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'patch_features': patch_features,
            'case_id': case_id,
            'question': question,
            'choices': choices,
            'answer': answer,
            'answer_label': answer_label,
            'pad_token_id': self.pad_token_id
        }


def collate_fn(batch):
    max_patches = max([item['patch_features'].shape[0] for item in batch])
    max_seq_len = max([item['input_ids'].shape[0] for item in batch])
    pad_token_id = batch[0]['pad_token_id']
    
    padded_features = []
    for item in batch:
        features = item['patch_features']
        N = features.shape[0]
        if N < max_patches:
            padding = torch.zeros(max_patches - N, features.shape[1])
            features = torch.cat([features, padding], dim=0)
        padded_features.append(features)
    
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
        'case_id': [item['case_id'] for item in batch],
        'question': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'answer': [item['answer'] for item in batch],
        'answer_label': [item['answer_label'] for item in batch]
    }


def extract_answer(text):
    """Extract answer choice (A/B/C/D) from generated text."""
    text = text.strip().upper()
    
    # Pattern 1: Direct answer "A" or "A)" or "A."
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    
    # Pattern 2: "The answer is A"
    match = re.search(r'ANSWER\s+IS\s+([A-D])', text)
    if match:
        return match.group(1)
    
    # Pattern 3: First letter if it's A/B/C/D
    if len(text) > 0 and text[0] in ['A', 'B', 'C', 'D']:
        return text[0]
    
    return None


def evaluate_model(model, dataloader, args):
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    all_case_ids = []
    all_questions = []
    all_generated_texts = []
    all_choices = []
    
    print("\nRunning inference...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            patch_features = batch['patch_features'].cuda()
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_features=patch_features,
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.9,
                do_sample=False
            )
            
            for i in range(len(generated_ids)):
                generated_text = model.tokenizer.decode(
                    generated_ids[i],
                    skip_special_tokens=True
                ).strip()
                
                predicted_answer = extract_answer(generated_text)
                
                if len(all_predictions) < 3:
                    print(f"  DEBUG: Pred='{predicted_answer}' GT='{batch['answer_label'][i]}' | Raw='{generated_text[:60]}...'")
                
                all_predictions.append(predicted_answer)
                all_ground_truth.append(batch['answer_label'][i])
                all_case_ids.append(batch['case_id'][i])
                all_questions.append(batch['question'][i])
                all_generated_texts.append(generated_text)
                all_choices.append(batch['choices'][i])
    
    # Compute metrics
    print("\nComputing metrics...")
    
    correct = sum([1 for pred, gt in zip(all_predictions, all_ground_truth) if pred == gt])
    total = len(all_predictions)
    accuracy = correct / total if total > 0 else 0
    
    # Per case_id accuracy
    case_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for pred, gt, case_id in zip(all_predictions, all_ground_truth, all_case_ids):
        case_stats[case_id]['total'] += 1
        if pred == gt:
            case_stats[case_id]['correct'] += 1
    
    case_accuracies = {
        case_id: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        for case_id, stats in case_stats.items()
    }
    
    results = {
        'accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'case_accuracies': case_accuracies,
        'case_stats': {
            case_id: {
                'accuracy': acc,
                'correct': case_stats[case_id]['correct'],
                'total': case_stats[case_id]['total']
            }
            for case_id, acc in case_accuracies.items()
        },
        'predictions': [
            {
                'case_id': case_id,
                'question': question,
                'choices': choices,
                'predicted_answer': pred,
                'ground_truth': gt,
                'generated_text': gen_text,
                'correct': pred == gt
            }
            for case_id, question, choices, pred, gt, gen_text in zip(
                all_case_ids, all_questions, all_choices,
                all_predictions, all_ground_truth, all_generated_texts
            )
        ]
    }
    
    return results


def print_results(results):
    print("\n" + "=" * 80)
    print("WSI-VQA* EVALUATION RESULTS (Closed-Set Only)")
    print("=" * 80)
    
    print(f"\n*** Accuracy: {results['accuracy']:.2%} ***")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    
    print("\n" + "-" * 80)
    print("Comparison with SlideChat paper benchmarks:")
    print("-" * 80)
    print(f"  SlideChat:  60.18%")
    print(f"  MedDr:      54.36%")
    print(f"  GPT-4o:     14.03%")
    print(f"  Ours:       {results['accuracy']:.2%}")
    
    print("\n" + "-" * 80)
    print("Per Case Accuracy:")
    print("-" * 80)
    
    print(f"{'Case ID':<20} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-" * 55)
    
    for case_id, stats in sorted(results['case_stats'].items()):
        acc = stats['accuracy']
        correct = stats['correct']
        total = stats['total']
        print(f"{case_id:<20} {acc:>7.2%}         {correct:>4}/{total:<4}")
    
    print("=" * 80)


def save_results(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {output_path}")
    
    import pandas as pd
    summary_path = output_path.replace('.json', '_summary.csv')
    
    summary_data = {
        'Metric': ['WSI-VQA* Accuracy', 'Total Samples', 'Correct Predictions'],
        'Value': [f"{results['accuracy']:.2%}", results['total_samples'], results['correct_predictions']]
    }
    
    for case_id, stats in sorted(results['case_stats'].items()):
        summary_data['Metric'].append(f"{case_id}")
        summary_data['Value'].append(f"{stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate WSI-VQA* (Closed-Set Only)')
    
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--moe_checkpoint', type=str, required=True)
    parser.add_argument('--llm_path', type=str, default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--stage1_moe_path', type=str, default=None)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_visual_tokens', type=int, default=16)
    parser.add_argument('--visual_dim', type=int, default=512)
    parser.add_argument('--feature_suffix', type=int, default=1024)
    parser.add_argument('--skip_lora', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("WSI-VQA* Evaluation (Closed-Set VQA Only)")
    print("=" * 80)
    print(f"Model Path: {args.model_path}")
    print(f"Test Path: {args.test_path}")
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
    
    print(f"Loading checkpoint from {args.model_path}")
    
    projector_path = os.path.join(args.model_path, "projector.pt")
    if os.path.exists(projector_path):
        model.projector.load_state_dict(torch.load(projector_path, map_location='cpu'))
        print(f"Loaded projector from {projector_path}")
    
    stage1_moe_path = getattr(args, 'stage1_moe_path', None)
    if stage1_moe_path and os.path.exists(stage1_moe_path):
        moe_state_dict = torch.load(stage1_moe_path, map_location='cpu')
        model.visual_encoder.load_state_dict(moe_state_dict)
        print(f"Loaded Stage 1 finetuned MoE from {stage1_moe_path}")
    else:
        moe_in_model_path = os.path.join(args.model_path, "moe_compressor.pt")
        if os.path.exists(moe_in_model_path):
            moe_state_dict = torch.load(moe_in_model_path, map_location='cpu')
            model.visual_encoder.load_state_dict(moe_state_dict)
            print(f"Loaded MoE from {moe_in_model_path}")
    
    skip_lora = getattr(args, 'skip_lora', False)
    lora_path = os.path.join(args.model_path, "lora_adapter")
    if not skip_lora and os.path.exists(lora_path):
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, lora_path)
        print(f"Loaded LoRA adapter from {lora_path}")
    
    model = model.cuda()
    model.eval()
    
    # Create dataset
    print("\nLoading WSI-VQA* dataset (closed-set only)...")
    
    dataset = WsiVQAClosedDataset(
        json_path=args.test_path,
        features_dir=args.features_dir,
        tokenizer=model.tokenizer,
        visual_dim=args.visual_dim,
        num_visual_tokens=args.num_visual_tokens,
        feature_suffix=args.feature_suffix
    )
    
    if len(dataset) == 0:
        print("ERROR: No samples with available features found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    results = evaluate_model(model, dataloader, args)
    print_results(results)
    save_results(results, args.output_path)
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()
