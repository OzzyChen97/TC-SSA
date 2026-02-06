"""
Test the compression ratio of the MoE token compressor.

This script evaluates:
1. Compression ratio: N (original patches) -> num_slots (compressed tokens)
2. Token reduction statistics across multiple WSI samples
3. Memory savings estimation

Usage:
    cd /workspace/zhuo/ETC
    CUDA_VISIBLE_DEVICES=0 python tools/test_compression_ratio.py \
        --checkpoint_path /workspace/zhuo/ETC/outputs/moe_tcga_32slots_top2_robust/best_model.pth \
        --csv_path /workspace/zhuo/ETC/vqa/data/TCGA_priority/val.csv \
        --num_samples 500 \
        --feature_dim 512 \
        --num_slots 32 \
        --num_classes 9
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict

from src.models import build_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test MoE compression ratio on WSI data'
    )
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with data')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples to test (default: 500)')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='Feature dimension (default: 512)')
    parser.add_argument('--num_slots', type=int, default=32,
                        help='Number of MoE slots (default: 32)')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of classes (default: 9)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Optional: save detailed results to CSV')
    
    return parser.parse_args()


def load_model(checkpoint_path, args, device):
    """Load the trained MoE model."""
    model = build_model(
        model_type='moe',
        input_dim=args.feature_dim,
        num_slots=args.num_slots,
        num_classes=args.num_classes
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def load_features(file_path, feature_dim):
    """Load features from .npy file."""
    try:
        d = np.load(file_path, allow_pickle=True).item()
        features = d['feature']
        
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        
        features = features.float()
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
            
        return features
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_compression_stats(original_patches, num_slots):
    """Compute compression statistics for a single sample."""
    compression_ratio = original_patches / num_slots
    
    # Each patch has 'feature_dim' floats (4 bytes per float32)
    # Original memory: N * D * 4 bytes
    # Compressed: num_slots * D * 4 bytes
    # Memory reduction is the same as compression ratio
    
    return {
        'original_patches': original_patches,
        'compressed_tokens': num_slots,
        'compression_ratio': compression_ratio,
        'token_reduction': original_patches - num_slots,
        'token_reduction_percentage': (1 - num_slots / original_patches) * 100
    }


def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"MoE Token Compression Ratio Analysis")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Data: {args.csv_path}")
    print(f"Samples to test: {args.num_samples}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Number of slots: {args.num_slots}")
    print(f"{'='*60}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint_path, args, device)
    print(f"Model loaded successfully!")
    
    # Load data metadata
    print(f"Loading data from {args.csv_path}...")
    metadata = pd.read_csv(args.csv_path)
    
    # Limit to num_samples
    if len(metadata) > args.num_samples:
        metadata = metadata.sample(n=args.num_samples, random_state=42)
    
    print(f"Testing on {len(metadata)} samples\n")
    
    # Collect statistics
    all_stats = []
    patch_counts = []
    compression_ratios = []
    routing_data = defaultdict(list)
    
    # Process each sample
    with torch.no_grad():
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing"):
            file_path = row['file_path']
            slide_id = row['slide_id']
            
            # Load features
            features = load_features(file_path, args.feature_dim)
            if features is None:
                continue
            
            num_patches = features.shape[0]
            patch_counts.append(num_patches)
            
            # Move to device
            features = features.unsqueeze(0).to(device)  # [1, N, D]
            
            # Forward pass with routing stats
            try:
                logits, aux_loss, routing_stats = model(features, return_routing_stats=True)
                
                # Compute compression stats
                stats = compute_compression_stats(num_patches, args.num_slots)
                stats['slide_id'] = slide_id
                all_stats.append(stats)
                compression_ratios.append(stats['compression_ratio'])
                
                # Store routing info
                routing_data['slot_counts'].append(routing_stats['slot_counts'].numpy())
                routing_data['importance'].append(routing_stats['importance'].numpy())
                
            except Exception as e:
                print(f"Error processing {slide_id}: {e}")
                continue
    
    # Compute summary statistics
    patch_counts = np.array(patch_counts)
    compression_ratios = np.array(compression_ratios)
    
    print(f"\n{'='*60}")
    print("COMPRESSION RATIO SUMMARY")
    print(f"{'='*60}")
    
    # 计算总体压缩率
    total_original_patches = patch_counts.sum()
    total_compressed_tokens = args.num_slots * len(patch_counts)
    overall_compression_ratio = total_original_patches / total_compressed_tokens
    
    print(f"\n🎯 ===== 总体压缩率 (Overall Compression Ratio) =====")
    print(f"   * 原始 Patch 总数:        {total_original_patches:,}")
    print(f"   * 压缩后 Token 总数:      {total_compressed_tokens:,}")
    print(f"   * 【总压缩率】:           {overall_compression_ratio:.2f}x")
    print(f"   * 【总缩减比例】:         {100 * (1 - 1/overall_compression_ratio):.1f}%")
    print(f"   ================================================")
    
    print(f"\n📊 Patch Count Statistics:")
    print(f"   * Total samples analyzed: {len(patch_counts)}")
    print(f"   * Min patches per WSI:    {patch_counts.min():,}")
    print(f"   * Max patches per WSI:    {patch_counts.max():,}")
    print(f"   * Mean patches per WSI:   {patch_counts.mean():,.1f}")
    print(f"   * Median patches per WSI: {np.median(patch_counts):,.1f}")
    print(f"   * Std patches per WSI:    {patch_counts.std():,.1f}")
    
    print(f"\n🗜️ Compression Ratio Statistics (per sample):")
    print(f"   * Fixed output tokens:    {args.num_slots}")
    print(f"   * Min compression ratio:  {compression_ratios.min():.2f}x")
    print(f"   * Max compression ratio:  {compression_ratios.max():.2f}x")
    print(f"   * Mean compression ratio: {compression_ratios.mean():.2f}x")
    print(f"   * Median compression ratio: {np.median(compression_ratios):.2f}x")
    
    # Token reduction
    token_reductions = patch_counts - args.num_slots
    reduction_percentages = (1 - args.num_slots / patch_counts) * 100
    
    print(f"\n🎯 Token Reduction Statistics:")
    print(f"   * Mean tokens reduced:    {token_reductions.mean():,.1f}")
    print(f"   * Mean reduction %:       {reduction_percentages.mean():.1f}%")
    print(f"   * Best case reduction:    {token_reductions.max():,} tokens ({reduction_percentages.max():.1f}%)")
    print(f"   * Worst case reduction:   {token_reductions.min():,} tokens ({reduction_percentages.min():.1f}%)")
    
    # Memory savings estimation (assuming float32)
    bytes_per_token = args.feature_dim * 4  # float32
    original_memory = patch_counts * bytes_per_token
    compressed_memory = args.num_slots * bytes_per_token
    memory_savings = original_memory - compressed_memory
    
    print(f"\n💾 Estimated Memory Savings (per sample):")
    print(f"   * Bytes per token:        {bytes_per_token:,} bytes")
    print(f"   * Mean original memory:   {original_memory.mean() / 1024:.1f} KB")
    print(f"   * Fixed compressed size:  {compressed_memory / 1024:.1f} KB")
    print(f"   * Mean memory saved:      {memory_savings.mean() / 1024:.1f} KB ({100 * memory_savings.mean() / original_memory.mean():.1f}%)")
    
    # Expert utilization
    if routing_data['slot_counts']:
        all_slot_counts = np.stack(routing_data['slot_counts'])
        mean_slot_utilization = all_slot_counts.mean(axis=0)
        
        print(f"\n🔀 Expert Routing Statistics:")
        print(f"   * Mean patches per slot:  {mean_slot_utilization.mean():.1f}")
        print(f"   * Slot utilization std:   {mean_slot_utilization.std():.1f}")
        print(f"   * Min slot utilization:   {mean_slot_utilization.min():.1f}")
        print(f"   * Max slot utilization:   {mean_slot_utilization.max():.1f}")
        
        # Check for underutilized slots
        idle_slots = (mean_slot_utilization < 1).sum()
        print(f"   * Underutilized slots:    {idle_slots}/{args.num_slots}")
    
    print(f"\n{'='*60}")
    print("DISTRIBUTION BY COMPRESSION RATIO")
    print(f"{'='*60}")
    
    # Binned analysis
    bins = [0, 10, 50, 100, 200, 500, float('inf')]
    bin_labels = ['<10x', '10-50x', '50-100x', '100-200x', '200-500x', '>500x']
    
    for i in range(len(bins) - 1):
        mask = (compression_ratios >= bins[i]) & (compression_ratios < bins[i+1])
        count = mask.sum()
        percentage = 100 * count / len(compression_ratios)
        if count > 0:
            avg_patches = patch_counts[mask].mean()
            print(f"   {bin_labels[i]:10s}: {count:4d} samples ({percentage:5.1f}%), avg {avg_patches:,.0f} patches")
    
    print(f"{'='*60}\n")
    
    # Save detailed results if requested
    if args.output_file:
        df_results = pd.DataFrame(all_stats)
        df_results.to_csv(args.output_file, index=False)
        print(f"Detailed results saved to: {args.output_file}")


if __name__ == '__main__':
    main()
