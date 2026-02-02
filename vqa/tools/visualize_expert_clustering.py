"""
Expert Semantic Clustering Visualization for MoE Token Compressor.

This script visualizes how the MoE (Mixture of Experts) model clusters WSI patches
into semantic groups. It demonstrates the effect of expert routing by:
1. Loading patch features from multiple WSI slides
2. Computing expert routing assignments for each patch
3. Applying t-SNE/UMAP for dimensionality reduction
4. Creating beautiful visualization showing semantic clustering by expert

Usage:
    python vqa/tools/visualize_expert_clustering.py \
        --moe_checkpoint vqa/outputs/slidechat_stage1_32slots_robust/final/moe_compressor.pt \
        --features_dir vqa/data/GTEx-TCGA-Embeddings/TCGA-BR/TCGA-BR \
        --output_dir vqa/results/expert_clustering \
        --num_slides 10 \
        --max_patches 5000
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_rgba
from sklearn.manifold import TSNE
import seaborn as sns

# Try to import UMAP as an alternative
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: UMAP not installed. Using t-SNE only. Install with: pip install umap-learn")

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models import MoE_Compressor


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Expert Semantic Clustering')
    parser.add_argument('--moe_checkpoint', type=str, required=True,
                        help='Path to MoE_Compressor checkpoint (.pt file)')
    parser.add_argument('--features_dir', type=str, nargs='+', required=True,
                        help='Directory(ies) containing .npy feature files (can specify multiple)')
    parser.add_argument('--output_dir', type=str, default='vqa/results/expert_clustering',
                        help='Output directory for visualizations')
    parser.add_argument('--num_slides', type=int, default=10,
                        help='Number of slides to sample (default: 10)')
    parser.add_argument('--max_patches', type=int, default=5000,
                        help='Maximum number of patches to visualize (default: 5000)')
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'],
                        help='Dimensionality reduction method (default: tsne)')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE perplexity (default: 30)')
    parser.add_argument('--n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors (default: 15)')
    parser.add_argument('--top_k_experts', type=int, default=16,
                        help='Show top K most active experts (default: 16)')
    parser.add_argument('--num_slots', type=int, default=32,
                        help='Number of MoE slots (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    return parser.parse_args()


def load_moe_compressor(checkpoint_path: str, num_slots: int, device: str) -> Tuple[MoE_Compressor, int]:
    """Load MoE Compressor from checkpoint."""
    print(f"Loading MoE Compressor from {checkpoint_path}")
    
    # Load weights first to detect dimensions
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    # Remove 'compressor.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('compressor.'):
            new_state_dict[k[len('compressor.'):]] = v
        else:
            new_state_dict[k] = v
    
    # Auto-detect input_dim from gate weight shape
    gate_weight = new_state_dict.get('gate.weight')
    if gate_weight is not None:
        detected_num_slots = gate_weight.shape[0]
        input_dim = gate_weight.shape[1]
        print(f"  Auto-detected: num_slots={detected_num_slots}, input_dim={input_dim}")
        num_slots = detected_num_slots
    else:
        input_dim = 1024  # Default
    
    # Create model with correct dimensions
    model = MoE_Compressor(num_slots=num_slots, top_k=1, input_dim=input_dim)
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"  Loaded with {num_slots} expert slots, input_dim={input_dim}")
    return model, input_dim


def load_features_from_dirs(features_dirs: List[str], num_slides: int, max_patches: int,
                            seed: int = 42, target_dim: int = None) -> Tuple[torch.Tensor, np.ndarray]:
    """Load and sample features from multiple slides across multiple directories."""
    print(f"Loading features from {len(features_dirs)} directories...")
    
    # Find all feature files from all directories
    all_feature_files = []
    for features_dir in features_dirs:
        # Check for nested directory structure
        feature_dir_path = Path(features_dir)
        
        # Try direct path first
        feature_files = list(feature_dir_path.glob("*_0_1024.npy"))
        
        # If no files found, try nested path (e.g., TCGA-BR/TCGA-BR/)
        if len(feature_files) == 0:
            for subdir in feature_dir_path.iterdir():
                if subdir.is_dir():
                    feature_files.extend(list(subdir.glob("*_0_1024.npy")))
        
        tissue_name = feature_dir_path.name
        print(f"  {tissue_name}: Found {len(feature_files)} feature files")
        all_feature_files.extend(feature_files)
    
    print(f"  Total: {len(all_feature_files)} feature files")
    
    if len(all_feature_files) == 0:
        raise ValueError(f"No feature files found in any directory")
    
    # Sample slides
    np.random.seed(seed)
    if len(all_feature_files) > num_slides:
        all_feature_files = list(np.random.choice(all_feature_files, num_slides, replace=False))
    
    all_features = []
    all_slide_ids = []
    
    for file_path in all_feature_files:
        try:
            data = np.load(str(file_path), allow_pickle=True)
            
            # Handle dictionary-wrapped features (e.g., {'index': ..., 'feature': ...})
            if isinstance(data, np.ndarray) and data.dtype == object:
                if data.ndim == 0:
                    # Scalar object array containing a dict
                    data_dict = data.item()
                    if isinstance(data_dict, dict) and 'feature' in data_dict:
                        features = np.array(data_dict['feature'], dtype=np.float32)
                    else:
                        raise ValueError(f"Unknown data format: {type(data_dict)}")
                else:
                    features = np.array(data.tolist(), dtype=np.float32)
            else:
                features = data.astype(np.float32)
            
            slide_id = file_path.stem.split('_')[0][:20]  # Truncate for display
            
            # Sample patches if too many
            n_patches = features.shape[0]
            if n_patches > max_patches // num_slides:
                indices = np.random.choice(n_patches, max_patches // num_slides, replace=False)
                features = features[indices]
            
            all_features.append(features)
            all_slide_ids.extend([slide_id] * features.shape[0])
            
        except Exception as e:
            print(f"  Warning: Could not load {file_path}: {e}")
            continue
    
    if len(all_features) == 0:
        raise ValueError("No features could be loaded")
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    
    # Final sampling if still too many
    if all_features.shape[0] > max_patches:
        np.random.seed(seed)
        indices = np.random.choice(all_features.shape[0], max_patches, replace=False)
        all_features = all_features[indices]
        all_slide_ids = np.array(all_slide_ids)[indices]
    
    # Truncate to target dimension if needed (1024 -> 512 etc.)
    if target_dim is not None and all_features.shape[1] > target_dim:
        print(f"  Truncating features from {all_features.shape[1]} to {target_dim} dimensions")
        all_features = all_features[:, :target_dim]
    
    print(f"  Loaded {all_features.shape[0]} patches from {len(all_feature_files)} slides")
    print(f"  Feature dimension: {all_features.shape[1]}")
    
    features_tensor = torch.tensor(all_features, dtype=torch.float32)
    return features_tensor, np.array(all_slide_ids)


@torch.no_grad()
def get_expert_assignments(model: MoE_Compressor, features: torch.Tensor,
                           device: str, batch_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Get expert assignments and probabilities for all patches."""
    print("Computing expert routing assignments...")
    
    features = features.to(device)
    n_patches = features.shape[0]
    
    all_assignments = []
    all_probs = []
    
    for i in range(0, n_patches, batch_size):
        batch = features[i:i+batch_size].unsqueeze(0)  # [1, B, D]
        
        # Reshape to [B, 1, D] for processing
        batch = batch.squeeze(0).unsqueeze(1)  # [B, 1, D]
        
        # Process each patch
        for j in range(batch.shape[0]):
            patch = batch[j:j+1]  # [1, 1, D]
            logits = model.gate(patch)  # [1, 1, num_slots]
            probs = F.softmax(logits, dim=-1)  # [1, 1, num_slots]
            
            # Top-1 assignment
            assignment = torch.argmax(probs, dim=-1)
            
            all_assignments.append(assignment.cpu().numpy().flatten()[0])
            all_probs.append(probs.cpu().numpy().flatten())
    
    all_assignments = np.array(all_assignments)
    all_probs = np.array(all_probs)
    
    print(f"  Computed assignments for {n_patches} patches")
    return all_assignments, all_probs


def create_expert_colormap(num_experts: int) -> Dict[int, Tuple[float, float, float, float]]:
    """Create a colormap for experts with distinct, beautiful colors."""
    # Use a combination of qualitative colormaps for distinct colors
    if num_experts <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_experts))
    elif num_experts <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, num_experts))
    else:
        # Combine multiple colormaps for more colors
        cmap1 = plt.cm.tab20(np.linspace(0, 1, 20))
        cmap2 = plt.cm.Set3(np.linspace(0, 1, 12))
        colors = np.vstack([cmap1, cmap2])[:num_experts]
    
    return {i: tuple(colors[i]) for i in range(num_experts)}


def reduce_dimensions(features: np.ndarray, method: str = 'tsne',
                      perplexity: int = 30, n_neighbors: int = 15,
                      seed: int = 42) -> np.ndarray:
    """Reduce feature dimensions to 2D using t-SNE or UMAP."""
    print(f"Reducing dimensions using {method.upper()}...")
    
    if method == 'tsne':
        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, features.shape[0] - 1),
            learning_rate='auto',
            init='pca',
            random_state=seed,
            n_jobs=-1
        )
    elif method == 'umap' and HAS_UMAP:
        reducer = UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, features.shape[0] - 1),
            min_dist=0.1,
            metric='cosine',
            random_state=seed
        )
    else:
        print("  UMAP not available, falling back to t-SNE")
        reducer = TSNE(
            n_components=2,
            perplexity=min(perplexity, features.shape[0] - 1),
            learning_rate='auto',
            init='pca',
            random_state=seed,
            n_jobs=-1
        )
    
    embeddings = reducer.fit_transform(features)
    print(f"  Reduced to 2D: {embeddings.shape}")
    return embeddings


def plot_expert_clustering(
    embeddings: np.ndarray,
    expert_assignments: np.ndarray,
    expert_probs: np.ndarray,
    num_slots: int,
    top_k_experts: int,
    output_path: str,
    method: str = 'tsne',
    title: str = "MoE Expert Semantic Clustering"
):
    """Create a clean visualization of expert clustering with only a few representative experts."""
    print("Creating visualization...")
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Count expert usage
    expert_counts = np.bincount(expert_assignments, minlength=num_slots)
    
    # Select only 6 representative experts (top ones with good counts)
    sorted_experts = np.argsort(expert_counts)[::-1]
    selected_experts = sorted_experts[:min(6, top_k_experts)]
    
    # Define distinct, vibrant colors for selected experts
    expert_colors = {
        selected_experts[0]: '#E74C3C',  # Red
        selected_experts[1]: '#3498DB',  # Blue  
        selected_experts[2]: '#27AE60',  # Green
        selected_experts[3]: '#F39C12',  # Orange
        selected_experts[4]: '#8E44AD',  # Purple
        selected_experts[5]: '#16A085',  # Teal
    }
    
    # Create simple figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # First plot "other" slots in light gray (background)
    other_mask = ~np.isin(expert_assignments, selected_experts)
    if other_mask.sum() > 0:
        ax.scatter(
            embeddings[other_mask, 0],
            embeddings[other_mask, 1],
            c='#DDDDDD',
            alpha=0.4,
            s=12,
            edgecolors='none',
            label='Other slots',
            zorder=1
        )
    
    # Plot selected experts with bold colors
    for i, expert_idx in enumerate(selected_experts):
        mask = expert_assignments == expert_idx
        if mask.sum() > 0:
            color = expert_colors.get(expert_idx, '#888888')
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=color,
                label=f'Slot {expert_idx} (n={mask.sum():,})',
                alpha=0.8,
                s=30,
                edgecolors='white',
                linewidth=0.5,
                zorder=2+i
            )
    
    # Axis labels
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=14, fontweight='bold')
    ax.tick_params(labelsize=12)
    
    # Title
    ax.set_title('MoE Slot Semantic Clustering',
                 fontsize=16, fontweight='bold', pad=15)
    
    # Legend - positioned inside the plot
    legend = ax.legend(
        loc='upper right',
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=False,
        title='Slot Assignments',
        title_fontsize=12,
        markerscale=1.5
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('#CCCCCC')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved visualization to {output_path}")


def plot_slot_similarity_heatmap(
    features: np.ndarray,
    expert_assignments: np.ndarray,
    num_slots: int,
    output_path: str
):
    """
    Plot cosine similarity heatmap between slot aggregated feature vectors.
    
    For each slot, aggregate features of all patches assigned to that slot,
    then compute pairwise cosine similarity between all slot vectors.
    """
    print("Creating slot similarity heatmap...")
    
    # Aggregate features by slot (use mean of assigned patches)
    slot_vectors = []
    active_slots = []
    
    for slot_idx in range(num_slots):
        mask = expert_assignments == slot_idx
        if mask.sum() > 0:
            # Mean of all patches assigned to this slot
            slot_vector = features[mask].mean(axis=0)
            slot_vectors.append(slot_vector)
            active_slots.append(slot_idx)
    
    if len(slot_vectors) < 2:
        print("  Warning: Not enough active slots for similarity calculation")
        return
    
    slot_vectors = np.array(slot_vectors)
    
    # Normalize vectors for cosine similarity
    norms = np.linalg.norm(slot_vectors, axis=1, keepdims=True)
    slot_vectors_normalized = slot_vectors / (norms + 1e-8)
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(slot_vectors_normalized, slot_vectors_normalized.T)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a diverging colormap centered at 0
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Create heatmap with slot labels
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine Similarity')
    cbar.ax.tick_params(labelsize=11)
    
    # Set axis labels
    ax.set_xticks(range(len(active_slots)))
    ax.set_yticks(range(len(active_slots)))
    ax.set_xticklabels([f'S{s}' for s in active_slots], fontsize=9)
    ax.set_yticklabels([f'S{s}' for s in active_slots], fontsize=9)
    
    ax.set_xlabel('Slot Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Slot Index', fontsize=13, fontweight='bold')
    ax.set_title('Slot Feature Similarity Matrix\n(Cosine similarity between slot-aggregated features)',
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add text annotations for similarity values (only for small matrices)
    if len(active_slots) <= 20:
        for i in range(len(active_slots)):
            for j in range(len(active_slots)):
                text_color = 'white' if abs(similarity_matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                       ha='center', va='center', fontsize=7, color=text_color)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Print statistics
    off_diag = similarity_matrix[~np.eye(len(active_slots), dtype=bool)]
    print(f"  Active slots: {len(active_slots)}/{num_slots}")
    print(f"  Avg off-diagonal similarity: {off_diag.mean():.3f}")
    print(f"  Max off-diagonal similarity: {off_diag.max():.3f}")
    print(f"  Min off-diagonal similarity: {off_diag.min():.3f}")
    print(f"  Saved heatmap to {output_path}")

def plot_expert_probability_heatmap(
    expert_probs: np.ndarray,
    expert_assignments: np.ndarray,
    num_slots: int,
    output_path: str
):
    """Create a heatmap showing routing probability distribution."""
    print("Creating routing probability heatmap...")
    
    # Calculate average probability for each expert
    avg_probs = expert_probs.mean(axis=0)
    
    # Sort experts by average probability
    sorted_indices = np.argsort(avg_probs)[::-1]
    sorted_probs = avg_probs[sorted_indices]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of average probabilities
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_slots))
    bars = ax1.bar(range(num_slots), sorted_probs, color=colors[sorted_indices])
    ax1.set_xlabel('Expert (sorted by probability)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Routing Probability', fontsize=12, fontweight='bold')
    ax1.set_title('Expert Routing Probability Distribution', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(0, num_slots, max(1, num_slots // 8)))
    ax1.set_xticklabels([f'{sorted_indices[i]}' for i in range(0, num_slots, max(1, num_slots // 8))])
    
    # Add value labels on top bars
    for i, bar in enumerate(bars[:5]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{sorted_probs[i]:.2%}', ha='center', va='bottom', fontsize=9)
    
    # Heatmap of probabilities (sample)
    sample_size = min(100, len(expert_probs))
    sample_indices = np.random.choice(len(expert_probs), sample_size, replace=False)
    sample_probs = expert_probs[sample_indices]
    
    im = ax2.imshow(sample_probs[:, sorted_indices].T, aspect='auto', cmap='viridis')
    ax2.set_xlabel('Sample Patches', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Expert Index (sorted)', fontsize=12, fontweight='bold')
    ax2.set_title('Routing Probability Heatmap (sampled)', fontsize=14, fontweight='bold')
    
    # Color bar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Routing Probability', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved heatmap to {output_path}")

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load MoE Compressor
    model, input_dim = load_moe_compressor(args.moe_checkpoint, args.num_slots, device)
    num_slots = model.num_slots  # Use actual num_slots from model
    
    # Load features (truncated to match model input_dim)
    features, slide_ids = load_features_from_dirs(
        args.features_dir,
        args.num_slides,
        args.max_patches,
        args.seed,
        target_dim=input_dim
    )
    
    # Get expert assignments
    expert_assignments, expert_probs = get_expert_assignments(model, features, device)
    
    # Reduce dimensions
    embeddings = reduce_dimensions(
        features.numpy(),
        method=args.method,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        seed=args.seed
    )
    
    # Create main clustering visualization
    main_output = os.path.join(args.output_dir, f'expert_clustering_{args.method}.png')
    plot_expert_clustering(
        embeddings,
        expert_assignments,
        expert_probs,
        num_slots,
        args.top_k_experts,
        main_output,
        method=args.method
    )
    
    # Create probability heatmap
    heatmap_output = os.path.join(args.output_dir, 'routing_probability_heatmap.png')
    plot_expert_probability_heatmap(
        expert_probs,
        expert_assignments,
        num_slots,
        heatmap_output
    )
    
    # Create slot similarity heatmap
    similarity_output = os.path.join(args.output_dir, 'slot_similarity_heatmap.png')
    plot_slot_similarity_heatmap(
        features.numpy(),  # Convert to numpy
        expert_assignments,
        num_slots,
        similarity_output
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Visualization Complete!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  Main clustering: {main_output}")
    print(f"  Probability heatmap: {heatmap_output}")
    print(f"  Slot similarity: {similarity_output}")
    
    # Print slot statistics
    expert_counts = np.bincount(expert_assignments, minlength=num_slots)
    top_experts = np.argsort(expert_counts)[::-1][:5]
    
    print(f"\nTop 5 Slots:")
    for i, expert_idx in enumerate(top_experts):
        count = expert_counts[expert_idx]
        pct = count / len(expert_assignments) * 100
        print(f"  {i+1}. Slot {expert_idx}: {count:,} patches ({pct:.1f}%)")
    
    print(f"\nTotal patches analyzed: {len(expert_assignments):,}")
    print(f"Active slots: {(expert_counts > 0).sum()}/{num_slots}")


if __name__ == '__main__':
    main()
