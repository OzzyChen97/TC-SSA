"""
Benchmark evaluation for SlideChat SlideBench.

Usage:
    python vqa/tools/test_slidechat.py \
        --model_path outputs/slidechat_stage2/final \
        --moe_checkpoint /path/to/moe_best.pt \
        --llm_path Qwen/Qwen3-4B-Instruct-2507 \
        --benchmark_path vqa/data/SlideChat/SlideBench-VQA-TCGA-plus.csv \
        --features_dir vqa/data/SlideChat/Feat \
        --output_path results/slidechat_results.json \
        --batch_size 8
"""

import argparse
import os
import sys
import json
import re
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqa.src.vqa_model import MoE_Qwen_VQA
from vqa.src.slidechat_dataset import SlideChatBenchmarkDataset, benchmark_collate_fn


def extract_answer(text):
    """Extract answer (A/B/C/D) from generated text."""
    text = text.strip().upper()

    # Pattern 1: Direct "A" or "A)" or "A."
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)

    # Pattern 2: "The answer is A"
    match = re.search(r'ANSWER\s+IS\s+([A-D])', text)
    if match:
        return match.group(1)

    # Pattern 3: First letter
    if len(text) > 0 and text[0] in ['A', 'B', 'C', 'D']:
        return text[0]

    return None


def evaluate_model(model, dataloader, args):
    """Run inference and compute metrics."""
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
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            patch_features = batch['patch_features'].cuda()

            # Generate
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                patch_features=patch_features,
                max_new_tokens=64,
                temperature=0.1,
                top_p=0.9,
                do_sample=False
            )

            # Decode
            for i in range(len(generated_ids)):
                input_length = input_ids[i].shape[0]
                generated_tokens = generated_ids[i][input_length:]

                generated_text = model.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()

                predicted_answer = extract_answer(generated_text)

                all_predictions.append(predicted_answer)
                all_ground_truth.append(batch['answer'][i])
                all_categories.append(batch['category'][i])
                all_slide_ids.append(batch['slide_id'][i])
                all_questions.append(batch['question'][i])
                all_generated_texts.append(generated_text)

    # Compute metrics
    print("\nComputing metrics...")

    correct = sum([1 for pred, gt in zip(all_predictions, all_ground_truth) if pred == gt])
    total = len(all_predictions)
    overall_accuracy = correct / total if total > 0 else 0

    # Per-category accuracy
    category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for pred, gt, cat in zip(all_predictions, all_ground_truth, all_categories):
        category_stats[cat]['total'] += 1
        if pred == gt:
            category_stats[cat]['correct'] += 1

    category_accuracies = {
        cat: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        for cat, stats in category_stats.items()
    }

    results = {
        'overall_accuracy': overall_accuracy,
        'total_samples': total,
        'correct_predictions': correct,
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
    """Print results."""
    print("\n" + "=" * 80)
    print("SLIDECHAT BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Correct: {results['correct_predictions']}")

    print("\n" + "-" * 80)
    print("Per-Category Accuracy:")
    print("-" * 80)
    print(f"{'Category':<30} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-" * 80)

    for cat, stats in sorted(results['category_stats'].items()):
        acc = stats['accuracy']
        correct = stats['correct']
        total = stats['total']
        print(f"{cat:<30} {acc:>7.2%}         {correct:>4}/{total:<4}")

    print("=" * 80)


def save_results(results, output_path):
    """Save results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")

    # Summary CSV
    summary_path = output_path.replace('.json', '_summary.csv')
    summary_data = {
        'Metric': ['Overall Accuracy', 'Total Samples', 'Correct'],
        'Value': [
            f"{results['overall_accuracy']:.2%}",
            results['total_samples'],
            results['correct_predictions']
        ]
    }

    for cat, stats in sorted(results['category_stats'].items()):
        summary_data['Metric'].append(f"{cat} Accuracy")
        summary_data['Value'].append(f"{stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

    df = pd.DataFrame(summary_data)
    df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate on SlideBench')

    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--moe_checkpoint', type=str, required=True)
    parser.add_argument('--llm_path', type=str, default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--benchmark_path', type=str, required=True)
    parser.add_argument('--features_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_visual_tokens', type=int, default=16)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("SlideBench Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Benchmark: {args.benchmark_path}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = MoE_Qwen_VQA(
        moe_checkpoint=args.moe_checkpoint,
        llm_path=args.llm_path,
        num_visual_tokens=args.num_visual_tokens,
        device='cuda'
    )

    model.load_pretrained(args.model_path)
    model = model.cuda()
    model.eval()

    # Create dataset
    print("\nLoading dataset...")
    dataset = SlideChatBenchmarkDataset(
        csv_path=args.benchmark_path,
        features_dir=args.features_dir,
        tokenizer=model.tokenizer,
        num_visual_tokens=args.num_visual_tokens
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=benchmark_collate_fn,
        num_workers=args.num_workers,
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
