"""
MoE-based Token Compressor for WSI Classification.

This module implements the core innovation: Mixture of Experts Token Compression.
Based on ETC.py implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    Individual Expert module with residual connection.
    Each expert processes features assigned to its slot.
    """
    def __init__(self, dim=1024, hidden_dim=512, act_layer=nn.GELU, drop=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.LayerNorm(dim)  # Layer norm for residual stability
        )

    def forward(self, x):
        """
        Forward pass with residual connection.

        Args:
            x: Input features [Batch, D]

        Returns:
            Processed features with residual [Batch, D]
        """
        return x + self.net(x)  # Residual connection


class MoE_Compressor(nn.Module):
    """
    Mixture of Experts Token Compressor.

    Core innovation: Compresses N variable-length patches into K fixed semantic tokens
    using expert routing and weighted aggregation.

    Architecture:
        1. Gate Network: Routes each patch to expert slots
        2. Top-K Routing: Selects top-k experts per patch
        3. Weighted Aggregation: Combines patches per slot
        4. Expert Processing: Each slot processes its aggregated features
        5. Load Balancing Loss: Ensures uniform expert utilization
    """

    def __init__(self, num_slots=64, top_k=2, input_dim=1024):
        """
        Initialize MoE Compressor.

        Args:
            num_slots: Number of compressed tokens (K)
            top_k: Number of experts each patch is routed to (usually 1)
            input_dim: Feature dimension (e.g., 1024 for UNI)
        """
        super().__init__()
        self.num_slots = num_slots
        self.top_k = top_k
        self.input_dim = input_dim

        # Gate network: Determines which slot each patch belongs to
        self.gate = nn.Linear(input_dim, num_slots)

        # Experts: Each slot has its own dedicated processor
        # Independent experts are crucial for semantic clustering
        self.experts = nn.ModuleList([
            Expert(dim=input_dim) for _ in range(num_slots)
        ])

    def forward(self, x):
        """
        Forward pass: Compress N patches to num_slots tokens.

        Args:
            x: Input features [B, N, D] (typically B=1 for WSI)

        Returns:
            final_tokens: Compressed features [B, num_slots, D]
            aux_loss: Load balancing loss (scalar)
        """
        B, N, D = x.shape

        # 1. Compute routing scores
        logits = self.gate(x)  # [B, N, num_slots]
        probs = F.softmax(logits, dim=-1)

        # 2. Select Top-K experts per patch
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # 3. Compute auxiliary load balancing loss
        # Critical for MoE: prevents all patches from routing to the same expert
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            # Importance: Total probability assigned to each expert
            importance = probs.sum(dim=1).mean(dim=0)  # [num_slots]
            # Coefficient of Variation squared for load balancing
            aux_loss = (torch.std(importance) / (importance.mean() + 1e-10)) ** 2

        # 4. Aggregation / Compression
        # Create mask for top-k routing
        mask = torch.zeros_like(probs).scatter_(-1, topk_indices, 1.0)

        # Weighted routing probabilities: keep only top-k
        weighted_probs = probs * mask  # [B, N, num_slots]

        # Core compression step:
        # Aggregate patches into slots via weighted sum
        # [B, num_slots, N] @ [B, N, D] -> [B, num_slots, D]
        compressed_tokens = torch.bmm(weighted_probs.transpose(1, 2), x)

        # Normalize: Convert to weighted average
        # Prevents numerical explosion when many patches route to one expert
        count = weighted_probs.sum(dim=1).unsqueeze(-1) + 1e-9  # [B, num_slots, 1]
        compressed_tokens = compressed_tokens / count

        # 5. Expert processing: Each expert refines its aggregated features
        final_tokens = []
        for i in range(self.num_slots):
            feat = compressed_tokens[:, i, :]  # [B, D]
            processed_feat = self.experts[i](feat)
            final_tokens.append(processed_feat.unsqueeze(1))

        final_tokens = torch.cat(final_tokens, dim=1)  # [B, num_slots, D]

        return final_tokens, aux_loss
