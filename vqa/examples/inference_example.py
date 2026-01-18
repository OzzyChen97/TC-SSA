"""
Example: Quick inference with trained WSI-VQA model.

This script demonstrates how to:
1. Load a trained VQA model
2. Run inference on a single WSI
3. Generate answers to questions

Usage:
    python vqa/examples/inference_example.py \
        --model_path outputs/stage2/final \
        --moe_checkpoint checkpoints/moe_best.pt \
        --features_path data/example_features.pt \
        --question "What is the primary diagnosis?"
"""

import argparse
import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqa.src.vqa_model import MoE_Qwen_VQA


def run_inference(args):
    """
    Run single-sample inference.
    """
    print("=" * 80)
    print("WSI-VQA Inference Example")
    print("=" * 80)

    # Load model
    print(f"\n[1/4] Loading model from {args.model_path}...")
    model = MoE_Qwen_VQA(
        moe_checkpoint=args.moe_checkpoint,
        llm_path=args.llm_path,
        num_visual_tokens=args.num_visual_tokens,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Load trained weights
    model.load_pretrained(args.model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    print(f"Model loaded successfully on {device}")

    # Load features
    print(f"\n[2/4] Loading patch features from {args.features_path}...")
    patch_features = torch.load(args.features_path, map_location=device)

    # Handle different feature formats
    if patch_features.dim() == 2:
        # [N, 1024] -> [1, N, 1024]
        patch_features = patch_features.unsqueeze(0)

    print(f"Features shape: {patch_features.shape}")

    # Prepare prompt
    print(f"\n[3/4] Preparing prompt...")
    prompt = f"<image> {args.question}"
    print(f"Prompt: {prompt}")

    # Tokenize
    inputs = model.tokenizer(
        prompt,
        return_tensors='pt',
        padding=False,
        truncation=False
    ).to(device)

    print(f"Input token IDs shape: {inputs['input_ids'].shape}")

    # Generate
    print(f"\n[4/4] Generating answer...")
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            patch_features=patch_features,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample
        )

    # Decode
    # Extract only newly generated tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = output_ids[0][input_length:]

    answer = model.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Question: {args.question}")
    print(f"Answer: {answer}")
    print("=" * 80)

    return answer


def parse_args():
    parser = argparse.ArgumentParser(description='WSI-VQA Inference Example')

    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--moe_checkpoint', type=str, required=True,
                       help='Path to MoE_Compressor checkpoint')
    parser.add_argument('--llm_path', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Path to base Qwen LLM')
    parser.add_argument('--num_visual_tokens', type=int, default=16,
                       help='Number of visual tokens')

    # Input
    parser.add_argument('--features_path', type=str, required=True,
                       help='Path to patch features (.pt or .npy)')
    parser.add_argument('--question', type=str, required=True,
                       help='Question to ask about the slide')

    # Generation
    parser.add_argument('--max_new_tokens', type=int, default=128,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Nucleus sampling top-p')
    parser.add_argument('--do_sample', action='store_true',
                       help='Use sampling instead of greedy decoding')

    return parser.parse_args()


def main():
    args = parse_args()
    run_inference(args)


if __name__ == '__main__':
    main()
