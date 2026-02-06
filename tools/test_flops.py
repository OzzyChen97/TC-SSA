"""
Test FLOPs (Floating Point Operations) of the MoE Token Compressor.

This script measures:
1. FLOPs of the MoE compressor
2. FLOPs comparison with different input sizes
3. Memory usage estimation

Usage:
    cd /workspace/zhuo/ETC
    python tools/test_flops.py \
        --feature_dim 512 \
        --num_slots 32 \
        --num_classes 9
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import torch
import numpy as np
from src.models import build_model, MoE_Compressor

# Try different profiling libraries
try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

try:
    from thop import profile, clever_format
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def manual_flops_moe_compressor(num_patches, feature_dim, num_slots, top_k=2):
    """
    Manually calculate FLOPs for MoE Compressor.
    
    Components:
    1. Gate network: Linear(D, num_slots) -> N * D * num_slots MACs
    2. Softmax: N * num_slots
    3. TopK selection: N * num_slots * log(num_slots)
    4. Weighted aggregation (bmm): num_slots * N * D MACs
    5. Expert processing (per slot): 
       - Linear(D, hidden_dim) + Linear(hidden_dim, D) + LayerNorm
    """
    hidden_dim = 512  # Expert hidden dim
    
    # 1. Gate network: [N, D] @ [D, num_slots] = [N, num_slots]
    gate_flops = num_patches * feature_dim * num_slots * 2  # MACs -> FLOPs (*2)
    
    # 2. Softmax: exp + sum + div = 3 * N * num_slots
    softmax_flops = 3 * num_patches * num_slots
    
    # 3. TopK: roughly O(N * num_slots)
    topk_flops = num_patches * num_slots
    
    # 4. Weighted aggregation: [num_slots, N] @ [N, D] = [num_slots, D]
    aggregation_flops = num_slots * num_patches * feature_dim * 2
    
    # 5. Count normalization: num_slots * N additions
    norm_flops = num_slots * num_patches
    
    # 6. Expert processing (for each of num_slots experts):
    #    - Linear(D, hidden_dim): D * hidden_dim * 2
    #    - GELU: ~10 ops per element
    #    - Linear(hidden_dim, D): hidden_dim * D * 2
    #    - LayerNorm: D * 4
    #    - Residual add: D
    expert_flops_per_slot = (
        feature_dim * hidden_dim * 2 +  # First linear
        hidden_dim * 10 +  # GELU approximation
        hidden_dim +  # Dropout (just assignments)
        hidden_dim * feature_dim * 2 +  # Second linear
        feature_dim * 4 +  # LayerNorm
        feature_dim  # Residual
    )
    total_expert_flops = num_slots * expert_flops_per_slot
    
    total_flops = (
        gate_flops + 
        softmax_flops + 
        topk_flops + 
        aggregation_flops + 
        norm_flops + 
        total_expert_flops
    )
    
    return {
        'gate': gate_flops,
        'softmax': softmax_flops,
        'topk': topk_flops,
        'aggregation': aggregation_flops,
        'normalization': norm_flops,
        'experts': total_expert_flops,
        'total': total_flops
    }


def manual_flops_classifier(feature_dim, num_classes):
    """
    FLOPs for the classifier head.
    - LayerNorm(D)
    - Linear(D, 512)
    - GELU
    - Dropout
    - Linear(512, num_classes)
    """
    hidden = 512
    
    layernorm_flops = feature_dim * 4
    linear1_flops = feature_dim * hidden * 2
    gelu_flops = hidden * 10
    linear2_flops = hidden * num_classes * 2
    
    return layernorm_flops + linear1_flops + gelu_flops + linear2_flops


def format_flops(flops):
    """Format FLOPs to human readable format."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


def benchmark_inference_time(model, input_tensor, num_runs=100, warmup=10):
    """Benchmark inference time."""
    model.eval()
    device = next(model.parameters()).device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times)
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Test FLOPs of MoE Compressor')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--num_slots', type=int, default=32)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--benchmark', action='store_true', help='Run inference time benchmark')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print(f"\n{'='*70}")
    print(f"MoE Token Compressor FLOPs Analysis")
    print(f"{'='*70}")
    print(f"Feature dimension: {args.feature_dim}")
    print(f"Number of slots:   {args.num_slots}")
    print(f"Number of classes: {args.num_classes}")
    print(f"{'='*70}\n")
    
    # Build model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = build_model(
        model_type='moe',
        input_dim=args.feature_dim,
        num_slots=args.num_slots,
        num_classes=args.num_classes
    )
    model = model.to(device)
    model.eval()
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"📊 Model Parameters:")
    print(f"   Total:     {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Size:      {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test different input sizes
    test_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
    
    print(f"\n{'='*70}")
    print("FLOPs Analysis by Input Size (Number of Patches)")
    print(f"{'='*70}")
    print(f"{'Patches':>10} | {'Compressor':>15} | {'Classifier':>12} | {'Total':>15} | {'Ratio':>8}")
    print(f"{'-'*10}-+-{'-'*15}-+-{'-'*12}-+-{'-'*15}-+-{'-'*8}")
    
    results = []
    
    for num_patches in test_sizes:
        # Manual FLOPs calculation
        compressor_flops = manual_flops_moe_compressor(
            num_patches, args.feature_dim, args.num_slots
        )
        classifier_flops = manual_flops_classifier(args.feature_dim, args.num_classes)
        
        total_flops = compressor_flops['total'] + classifier_flops
        
        # Calculate compression ratio impact
        ratio = num_patches / args.num_slots
        
        results.append({
            'patches': num_patches,
            'compressor': compressor_flops['total'],
            'classifier': classifier_flops,
            'total': total_flops
        })
        
        print(f"{num_patches:>10,} | {format_flops(compressor_flops['total']):>15} | "
              f"{format_flops(classifier_flops):>12} | {format_flops(total_flops):>15} | {ratio:>7.1f}x")
    
    # Detailed breakdown for typical case
    typical_patches = 1842  # Based on previous analysis
    print(f"\n{'='*70}")
    print(f"Detailed FLOPs Breakdown (Typical: {typical_patches} patches)")
    print(f"{'='*70}")
    
    breakdown = manual_flops_moe_compressor(typical_patches, args.feature_dim, args.num_slots)
    classifier_flops = manual_flops_classifier(args.feature_dim, args.num_classes)
    
    total = breakdown['total'] + classifier_flops
    
    print(f"\n🔧 MoE Compressor Components:")
    for name, flops in breakdown.items():
        if name != 'total':
            pct = 100 * flops / breakdown['total']
            print(f"   {name:15s}: {format_flops(flops):>15} ({pct:5.1f}%)")
    print(f"   {'─'*40}")
    print(f"   {'Compressor Total':15s}: {format_flops(breakdown['total']):>15}")
    
    print(f"\n🎯 Classifier Head:")
    print(f"   {'Classifier':15s}: {format_flops(classifier_flops):>15}")
    
    print(f"\n📈 Summary:")
    print(f"   {'Total FLOPs':15s}: {format_flops(total):>15}")
    print(f"   {'FLOPs per patch':15s}: {format_flops(total / typical_patches):>15}")
    
    # Use fvcore or thop if available for verification
    if HAS_FVCORE:
        print(f"\n{'='*70}")
        print("FLOPs Verification using fvcore")
        print(f"{'='*70}")
        
        for num_patches in [500, 1000, 2000]:
            dummy_input = torch.randn(1, num_patches, args.feature_dim).to(device)
            
            try:
                flops = FlopCountAnalysis(model, dummy_input)
                measured_flops = flops.total()
                print(f"   {num_patches} patches: {format_flops(measured_flops)}")
            except Exception as e:
                print(f"   Error: {e}")
    
    elif HAS_THOP:
        print(f"\n{'='*70}")
        print("FLOPs Verification using thop")
        print(f"{'='*70}")
        
        for num_patches in [500, 1000, 2000]:
            dummy_input = torch.randn(1, num_patches, args.feature_dim).to(device)
            
            try:
                macs, params = profile(model, inputs=(dummy_input,), verbose=False)
                flops = macs * 2  # MACs to FLOPs
                print(f"   {num_patches} patches: {format_flops(flops)}")
            except Exception as e:
                print(f"   Error: {e}")
    
    # Benchmark inference time if requested
    if args.benchmark:
        print(f"\n{'='*70}")
        print("Inference Time Benchmark")
        print(f"{'='*70}")
        print(f"{'Patches':>10} | {'Mean (ms)':>12} | {'Std (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10}")
        print(f"{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
        
        for num_patches in [100, 500, 1000, 2000, 5000]:
            dummy_input = torch.randn(1, num_patches, args.feature_dim).to(device)
            timing = benchmark_inference_time(model, dummy_input)
            print(f"{num_patches:>10,} | {timing['mean_ms']:>12.3f} | {timing['std_ms']:>10.3f} | "
                  f"{timing['min_ms']:>10.3f} | {timing['max_ms']:>10.3f}")
    
    print(f"\n{'='*70}")
    print("Analysis Complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
