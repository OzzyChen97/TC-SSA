"""
Visualize MoE Expert Routing on WSI Images.

This script:
1. Downloads WSI slides from GDC (optional, if slides_dir not provided)
2. Creates thumbnail images from WSI
3. Overlays expert routing information on the thumbnail

Usage:
    python tools/visualize_expert_on_wsi.py \
        --checkpoint /workspace/ETC/outputs/nsclc_r50_moe_experiment/best_model.pth \
        --features_dir /workspace/ETC/CPathPatchFeature/nsclc/r50/pt_files \
        --test_csv /workspace/ETC/data/nsclc-r50/test.csv \
        --output_dir /workspace/ETC/outputs/nsclc_r50_moe_experiment/eval/expert_wsi \
        --num_slides 5
"""

import argparse
import os
import sys
import subprocess
import json
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgba
from PIL import Image, ImageDraw, ImageFont

from src.models import build_model

# Try to import openslide for WSI reading
try:
    import openslide
    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False
    print("Warning: openslide not installed. Will use placeholder images.")


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize MoE Expert Routing on WSI')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing .pt feature files')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--slides_dir', type=str, default=None,
                        help='Directory containing WSI slides (will download if not provided)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--num_slides', type=int, default=5,
                        help='Number of slides to visualize')
    parser.add_argument('--thumbnail_size', type=int, default=1024,
                        help='Thumbnail size (longest edge)')
    parser.add_argument('--patch_size', type=int, default=256,
                        help='Patch size used during feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--download', action='store_true',
                        help='Download slides from GDC')
    parser.add_argument('--slide_ids', type=str, nargs='+', default=None,
                        help='Specific slide IDs to visualize')
    return parser.parse_args()


def get_gdc_file_uuid(slide_id):
    """
    Query GDC API to get file UUID from slide ID (filename).
    """
    # Extract case ID from slide ID (e.g., TCGA-05-4402 from TCGA-05-4402-01Z-00-DX1.xxx)
    case_id = '-'.join(slide_id.split('-')[:3])
    
    # GDC API endpoint
    files_endpt = "https://api.gdc.cancer.gov/files"
    
    # Query for the slide file
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "file_name",
                    "value": f"{slide_id}.svs"
                }
            }
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size",
        "format": "JSON",
        "size": "1"
    }
    
    try:
        response = requests.get(files_endpt, params=params, timeout=30)
        data = response.json()
        
        if data.get('data', {}).get('hits'):
            file_info = data['data']['hits'][0]
            return file_info['file_id'], file_info.get('file_size', 0)
    except Exception as e:
        print(f"  Error querying GDC API: {e}")
    
    return None, 0


def download_slide_from_gdc(file_uuid, output_path):
    """
    Download slide from GDC using file UUID.
    """
    data_endpt = f"https://api.gdc.cancer.gov/data/{file_uuid}"
    
    try:
        print(f"  Downloading from GDC (UUID: {file_uuid})...")
        response = requests.get(data_endpt, stream=True, timeout=300)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\r  Progress: {pct:.1f}%", end='', flush=True)
            print()
            return True
    except Exception as e:
        print(f"  Error downloading: {e}")
    
    return False


def create_thumbnail_from_wsi(wsi_path, thumbnail_size=1024):
    """
    Create a thumbnail from WSI file.
    """
    if not HAS_OPENSLIDE:
        return None, None, None
    
    try:
        slide = openslide.OpenSlide(wsi_path)
        
        # Get dimensions
        w, h = slide.dimensions
        
        # Calculate thumbnail dimensions
        if w > h:
            thumb_w = thumbnail_size
            thumb_h = int(h * thumbnail_size / w)
        else:
            thumb_h = thumbnail_size
            thumb_w = int(w * thumbnail_size / h)
        
        # Get thumbnail
        thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
        
        # Calculate scale factor
        scale_x = thumb_w / w
        scale_y = thumb_h / h
        
        slide.close()
        
        return thumbnail, (scale_x, scale_y), (w, h)
    except Exception as e:
        print(f"  Error creating thumbnail: {e}")
        return None, None, None


def create_placeholder_thumbnail(coords, thumbnail_size=1024, patch_size=256):
    """
    Create a placeholder thumbnail based on patch coordinates.
    """
    if coords is None or len(coords) == 0:
        return None, None, None
    
    # Get coordinate bounds
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max() + patch_size
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max() + patch_size
    
    w = x_max - x_min
    h = y_max - y_min
    
    # Calculate thumbnail dimensions
    if w > h:
        thumb_w = thumbnail_size
        thumb_h = int(h * thumbnail_size / w)
    else:
        thumb_h = thumbnail_size
        thumb_w = int(w * thumbnail_size / h)
    
    # Create white placeholder
    thumbnail = Image.new('RGB', (thumb_w, thumb_h), color=(240, 240, 240))
    
    # Scale factors
    scale_x = thumb_w / w
    scale_y = thumb_h / h
    
    return thumbnail, (scale_x, scale_y), (w, h)


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        num_classes = config.get('num_classes', 2)
        input_dim = config.get('input_dim', 1024)
        num_slots = config.get('num_slots', 64)
        top_k = config.get('top_k', 1)
    else:
        # Infer from state dict
        state_dict = checkpoint['model_state_dict']
        # Infer num_slots from gate weight shape
        if 'compressor.gate.weight' in state_dict:
            num_slots = state_dict['compressor.gate.weight'].shape[0]
        else:
            num_slots = 16  # Default
        
        # Infer input_dim from gate weight
        if 'compressor.gate.weight' in state_dict:
            input_dim = state_dict['compressor.gate.weight'].shape[1]
        else:
            input_dim = 1024
        
        # Infer num_classes from classifier
        if 'classifier.weight' in state_dict:
            num_classes = state_dict['classifier.weight'].shape[0]
        else:
            num_classes = 2
        
        top_k = 1  # Default
    
    print(f"  Inferred: num_slots={num_slots}, input_dim={input_dim}, num_classes={num_classes}")
    
    model = build_model(
        num_classes=num_classes,
        input_dim=input_dim,
        num_slots=num_slots,
        top_k=top_k
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, num_slots, top_k


def load_features(features_dir, slide_id):
    """Load features for a slide."""
    feature_path = os.path.join(features_dir, f"{slide_id}.pt")
    if not os.path.exists(feature_path):
        return None, None
    
    data = torch.load(feature_path, map_location='cpu')
    
    coords = None
    features = None
    
    if isinstance(data, dict):
        # Try to get features
        features = data.get('features', data.get('feats', None))
        if features is None:
            for key in data.keys():
                if isinstance(data[key], torch.Tensor) and data[key].dim() == 2:
                    features = data[key]
                    break
        
        # Try to get coordinates
        coords = data.get('coords', data.get('coordinates', None))
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.numpy()
    else:
        features = data
    
    return features, coords


@torch.no_grad()
def get_expert_routing(model, features, device, top_k=1):
    """
    Get expert routing indices for each patch.
    """
    features = features.unsqueeze(0).to(device)  # [1, N, D]
    
    # Get gate logits from the MoE compressor
    compressor = model.compressor
    logits = compressor.gate(features)  # [1, N, num_slots]
    probs = F.softmax(logits, dim=-1)  # [1, N, num_slots]
    
    # Get top-k experts
    topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)  # [1, N, top_k]
    
    return topk_indices[0].cpu().numpy(), topk_probs[0].cpu().numpy()


def get_expert_colors(num_slots):
    """
    Generate distinct colors for experts.
    """
    # Use a combination of colormaps for better distinction
    if num_slots <= 20:
        cmap = plt.cm.get_cmap('tab20', num_slots)
        colors = [cmap(i) for i in range(num_slots)]
    else:
        # Combine multiple colormaps
        colors = []
        cmaps = ['tab20', 'tab20b', 'tab20c', 'Set3']
        for i in range(num_slots):
            cmap = plt.cm.get_cmap(cmaps[i % len(cmaps)], 20)
            colors.append(cmap(i % 20))
    
    return colors


def overlay_expert_routing(thumbnail, coords, expert_indices, scale, patch_size,
                          num_slots, alpha=0.5, top_n_experts=8, show_all=False):
    """
    Overlay expert routing information on the thumbnail.

    Args:
        thumbnail: PIL Image
        coords: [N, 2] patch coordinates
        expert_indices: [N, top_k] expert indices
        scale: (scale_x, scale_y) from original to thumbnail
        patch_size: original patch size
        num_slots: total number of experts
        alpha: transparency for overlay
        top_n_experts: only color patches from top N most used experts (if show_all=False)
        show_all: if True, show all experts; if False, only show top N
    """
    if thumbnail is None or coords is None:
        return None

    # Create a copy of the thumbnail
    result = thumbnail.copy().convert('RGBA')
    overlay = Image.new('RGBA', result.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Get primary expert for each patch
    primary_expert = expert_indices[:, 0]

    # Count expert usage
    expert_counts = np.bincount(primary_expert, minlength=num_slots)
    top_experts = np.argsort(expert_counts)[::-1][:top_n_experts]

    # Get colors
    colors = get_expert_colors(num_slots)

    scale_x, scale_y = scale
    scaled_patch_w = int(patch_size * scale_x)
    scaled_patch_h = int(patch_size * scale_y)

    # Offset for coordinate origin
    x_offset = coords[:, 0].min()
    y_offset = coords[:, 1].min()

    # Draw patches
    for i, (x, y) in enumerate(coords):
        expert_idx = primary_expert[i]

        # Only draw patches from top experts (unless show_all is True)
        if not show_all and expert_idx not in top_experts:
            continue

        # Scale coordinates
        sx = int((x - x_offset) * scale_x)
        sy = int((y - y_offset) * scale_y)

        # Get color with alpha
        color = colors[expert_idx % len(colors)]
        rgba = tuple(int(c * 255) for c in color[:3]) + (int(alpha * 255),)

        # Draw rectangle
        draw.rectangle([sx, sy, sx + scaled_patch_w, sy + scaled_patch_h], fill=rgba)

    # Composite
    result = Image.alpha_composite(result, overlay)

    return result, top_experts, expert_counts


def create_expert_legend(top_experts, expert_counts, num_slots, output_path):
    """
    Create a legend image showing top experts and their colors.
    """
    colors = get_expert_colors(num_slots)
    
    fig, ax = plt.subplots(figsize=(6, len(top_experts) * 0.4 + 1))
    
    for i, expert_idx in enumerate(top_experts):
        color = colors[expert_idx % len(colors)]
        count = expert_counts[expert_idx]
        
        # Draw color box
        rect = mpatches.Rectangle((0, len(top_experts) - i - 1), 0.3, 0.8, 
                                   facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        
        # Add text
        ax.text(0.4, len(top_experts) - i - 0.6, 
                f"Expert {expert_idx}: {count} patches ({count/expert_counts.sum()*100:.1f}%)",
                fontsize=10, va='center')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(top_experts))
    ax.axis('off')
    ax.set_title('Top Experts by Usage', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def create_simple_overlay_with_legend(overlay_img, expert_counts, top_experts,
                                     slide_id, label, pred, num_slots, output_path):
    """
    Create a simple visualization: overlay image with legend on the side.
    """
    colors = get_expert_colors(num_slots)

    # Create figure with image + legend side by side
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.15)

    # 1. Overlay image
    ax1 = fig.add_subplot(gs[0, 0])
    if overlay_img is not None:
        ax1.imshow(overlay_img)
    ax1.set_title(f'{slide_id}\nLabel: {label} | Predicted: {pred}',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.axis('off')

    # 2. Legend with all active experts
    ax2 = fig.add_subplot(gs[0, 1])

    # Show all active experts (expert_counts > 0)
    active_experts = [(i, expert_counts[i]) for i in range(num_slots) if expert_counts[i] > 0]
    active_experts.sort(key=lambda x: x[1], reverse=True)  # Sort by count descending

    n_active = len(active_experts)
    total_patches = expert_counts.sum()

    for i, (expert_idx, count) in enumerate(active_experts[:15]):  # Show top 15
        color = colors[expert_idx % len(colors)]
        pct = count / total_patches * 100

        y_pos = n_active - i - 1 if n_active <= 15 else 14 - i

        # Color box
        rect = mpatches.Rectangle((0, y_pos), 0.15, 0.7,
                                   facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)

        # Text label
        ax2.text(0.2, y_pos + 0.35,
                f"Expert {expert_idx:2d}: {count:4d} patches ({pct:5.1f}%)",
                fontsize=10, va='center', fontweight='bold' if i < 3 else 'normal')

    # Set axis limits
    max_y = max(15, n_active)
    ax2.set_xlim(-0.1, 3)
    ax2.set_ylim(-0.5, max_y)
    ax2.axis('off')
    ax2.set_title(f'Expert Distribution\n({n_active} active / {num_slots} total)',
                  fontsize=12, fontweight='bold', pad=10)

    plt.suptitle('MoE Expert Routing on WSI', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved simple overlay: {output_path}")


def create_visualization_figure(thumbnail, overlay_img, coords, expert_indices,
                                expert_counts, top_experts, slide_id, label, pred,
                                num_slots, output_path):
    """
    Create a comprehensive visualization figure.
    """
    colors = get_expert_colors(num_slots)

    fig = plt.figure(figsize=(20, 12))

    # Grid spec for layout
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.5], height_ratios=[1, 0.4],
                          hspace=0.3, wspace=0.2)

    # 1. Original thumbnail (or placeholder)
    ax1 = fig.add_subplot(gs[0, 0])
    if thumbnail is not None:
        ax1.imshow(thumbnail)
    ax1.set_title(f'Slide Thumbnail\n{slide_id[:50]}...', fontsize=11)
    ax1.axis('off')

    # 2. Expert overlay
    ax2 = fig.add_subplot(gs[0, 1])
    if overlay_img is not None:
        ax2.imshow(overlay_img)
    ax2.set_title(f'Expert Routing Overlay\nTrue Label: {label}, Predicted: {pred}', fontsize=11)
    ax2.axis('off')

    # 3. Legend
    ax3 = fig.add_subplot(gs[0, 2])
    for i, expert_idx in enumerate(top_experts[:10]):
        color = colors[expert_idx % len(colors)]
        count = expert_counts[expert_idx]
        pct = count / expert_counts.sum() * 100

        rect = mpatches.Rectangle((0, 9 - i), 0.15, 0.8,
                                   facecolor=color, edgecolor='black', linewidth=0.5)
        ax3.add_patch(rect)
        ax3.text(0.2, 9.4 - i, f"E{expert_idx}: {count} ({pct:.1f}%)",
                fontsize=9, va='center')

    ax3.set_xlim(0, 2)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title('Top 10 Experts', fontsize=11, fontweight='bold')

    # 4. Expert distribution histogram
    ax4 = fig.add_subplot(gs[1, :2])
    bar_colors = [colors[i % len(colors)] for i in range(num_slots)]
    ax4.bar(range(num_slots), expert_counts, color=bar_colors, edgecolor='black', linewidth=0.3)
    ax4.set_xlabel('Expert Index', fontsize=10)
    ax4.set_ylabel('Number of Patches', fontsize=10)
    ax4.set_title(f'Expert Utilization Distribution (Total: {expert_counts.sum()} patches)', fontsize=11)
    ax4.set_xticks(range(0, num_slots, 4))

    # Highlight top experts
    for expert_idx in top_experts[:3]:
        ax4.axvline(x=expert_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)

    # 5. Statistics text
    ax5 = fig.add_subplot(gs[1, 2])
    stats_text = f"""Statistics:

Total Patches: {len(coords)}
Total Experts: {num_slots}
Active Experts: {(expert_counts > 0).sum()}

Top 3 Experts:
  E{top_experts[0]}: {expert_counts[top_experts[0]]} ({expert_counts[top_experts[0]]/expert_counts.sum()*100:.1f}%)
  E{top_experts[1]}: {expert_counts[top_experts[1]]} ({expert_counts[top_experts[1]]/expert_counts.sum()*100:.1f}%)
  E{top_experts[2]}: {expert_counts[top_experts[2]]} ({expert_counts[top_experts[2]]/expert_counts.sum()*100:.1f}%)

Gini Index: {compute_gini(expert_counts):.3f}
Entropy: {compute_entropy(expert_counts):.3f}
"""
    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.axis('off')

    plt.suptitle(f'MoE Expert Routing Visualization', fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {output_path}")


def compute_gini(counts):
    """Compute Gini coefficient of expert utilization."""
    counts = np.array(counts, dtype=float)
    if counts.sum() == 0:
        return 0
    counts = np.sort(counts)
    n = len(counts)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * counts) - (n + 1) * np.sum(counts)) / (n * np.sum(counts))


def compute_entropy(counts):
    """Compute entropy of expert utilization."""
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    slides_dir = args.slides_dir or os.path.join(args.output_dir, 'slides')
    os.makedirs(slides_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, num_slots, top_k = load_model(args.checkpoint, device)
    print(f"Model loaded: num_slots={num_slots}, top_k={top_k}")
    
    # Load test data
    test_df = pd.read_csv(args.test_csv)
    
    # Try to load predictions
    predictions_path = os.path.join(os.path.dirname(args.output_dir), 'predictions.csv')
    if os.path.exists(predictions_path):
        pred_df = pd.read_csv(predictions_path)
        pred_dict = dict(zip(pred_df['slide_id'], pred_df['pred_label']))
    else:
        pred_dict = {}
    
    # Select slides to visualize
    if args.slide_ids:
        slide_ids = args.slide_ids
    else:
        # Select a mix of labels
        label_0 = test_df[test_df['label'] == 0]['slide_id'].tolist()[:args.num_slides // 2 + 1]
        label_1 = test_df[test_df['label'] == 1]['slide_id'].tolist()[:args.num_slides // 2 + 1]
        slide_ids = (label_0 + label_1)[:args.num_slides]
    
    print(f"\nProcessing {len(slide_ids)} slides...")
    
    for slide_id in slide_ids:
        print(f"\n{'='*60}")
        print(f"Processing: {slide_id}")
        
        # Load features and coordinates
        features, coords = load_features(args.features_dir, slide_id)
        if features is None:
            print(f"  Features not found, skipping...")
            continue
        
        print(f"  Features shape: {features.shape}")
        
        # Try to infer coordinates from feature file or generate grid
        if coords is None:
            # Generate approximate grid coordinates
            n_patches = features.shape[0]
            grid_size = int(np.ceil(np.sqrt(n_patches)))
            coords = np.array([[i % grid_size * args.patch_size, 
                               i // grid_size * args.patch_size] 
                              for i in range(n_patches)])
            print(f"  Generated grid coordinates: {coords.shape}")
        else:
            print(f"  Coordinates shape: {coords.shape}")
        
        # Get label and prediction
        label_row = test_df[test_df['slide_id'] == slide_id]
        label = label_row['label'].values[0] if len(label_row) > 0 else 'N/A'
        pred = pred_dict.get(slide_id, 'N/A')
        
        # Get expert routing
        expert_indices, expert_probs = get_expert_routing(model, features, device, top_k)
        print(f"  Expert indices shape: {expert_indices.shape}")
        
        # Try to load/download WSI and create thumbnail
        thumbnail = None
        scale = None
        wsi_dims = None
        
        wsi_path = os.path.join(slides_dir, f"{slide_id}.svs")
        
        if os.path.exists(wsi_path) and HAS_OPENSLIDE:
            print(f"  Loading existing WSI...")
            thumbnail, scale, wsi_dims = create_thumbnail_from_wsi(wsi_path, args.thumbnail_size)
        elif args.download:
            print(f"  Attempting to download from GDC...")
            file_uuid, file_size = get_gdc_file_uuid(slide_id)
            if file_uuid:
                print(f"  Found file UUID: {file_uuid} (size: {file_size/1e9:.2f} GB)")
                if download_slide_from_gdc(file_uuid, wsi_path):
                    if HAS_OPENSLIDE:
                        thumbnail, scale, wsi_dims = create_thumbnail_from_wsi(wsi_path, args.thumbnail_size)
            else:
                print(f"  Could not find file on GDC")
        
        # If no thumbnail, create placeholder
        if thumbnail is None:
            print(f"  Creating placeholder thumbnail from coordinates...")
            thumbnail, scale, wsi_dims = create_placeholder_thumbnail(
                coords, args.thumbnail_size, args.patch_size
            )
        
        if thumbnail is None or scale is None:
            print(f"  Could not create thumbnail, skipping visualization...")
            continue
        
        print(f"  Thumbnail size: {thumbnail.size}")
        print(f"  Scale: {scale}")
        
        # Create overlay - show all active experts
        overlay_img, top_experts, expert_counts = overlay_expert_routing(
            thumbnail, coords, expert_indices, scale, args.patch_size,
            num_slots, alpha=0.5, top_n_experts=15, show_all=False
        )

        # Save individual thumbnail with overlay (simple PNG)
        overlay_path = os.path.join(args.output_dir, f"{slide_id}_expert_overlay.png")
        if overlay_img is not None:
            overlay_img.convert('RGB').save(overlay_path)
            print(f"  Saved overlay: {overlay_path}")

        # Create simple visualization with legend
        simple_viz_path = os.path.join(args.output_dir, f"{slide_id}_simple.png")
        create_simple_overlay_with_legend(
            overlay_img, expert_counts, top_experts,
            slide_id, label, pred, num_slots, simple_viz_path
        )

        # Create comprehensive visualization
        viz_path = os.path.join(args.output_dir, f"{slide_id}_detailed.png")
        create_visualization_figure(
            thumbnail, overlay_img, coords, expert_indices,
            expert_counts, top_experts, slide_id, label, pred,
            num_slots, viz_path
        )
    
    print(f"\n{'='*60}")
    print(f"Visualization complete! Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
