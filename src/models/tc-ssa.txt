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

    def forward(self, x, return_routing_stats=False):
        """
        Forward pass: Compress N patches to num_slots tokens.

        Args:
            x: Input features [B, N, D] (typically B=1 for WSI)
            return_routing_stats: If True, return routing statistics for monitoring

        Returns:
            final_tokens: Compressed features [B, num_slots, D]
            aux_loss: Load balancing loss (scalar)
            routing_stats (optional): Dict with slot assignment statistics
        """
        B, N, D = x.shape

        # 1. Compute routing scores
        logits = self.gate(x)  # [B, N, num_slots]

        # Add noise during training to encourage exploration (prevent collapse)
        if self.training:
            # Stronger noise to encourage exploration
            noise_scale = 0.1  # Increased from 1/num_slots
            noise = torch.randn_like(logits) * noise_scale
            logits = logits + noise

        probs = F.softmax(logits, dim=-1)

        # 2. Select Top-K experts per patch
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)

        # 3. Compute auxiliary load balancing loss
        aux_loss = torch.tensor(0.0, device=x.device)
        if self.training:
            # P_i: Fraction of probability assigned to expert i
            # f_i: Fraction of patches routed to expert i
            
            # 1. P_i (average probability per expert)
            prob_per_expert = probs.mean(dim=(0, 1))
            
            # 2. f_i (fraction of samples routed to expert i)
            zeros = torch.zeros_like(probs)
            mask_hard = zeros.scatter(-1, topk_indices, 1.0)
            frac_per_expert = mask_hard.mean(dim=(0, 1))
            
            # 3. Switch Transformer Loss: N * sum(P_i * f_i)
            switch_loss = self.num_slots * torch.sum(prob_per_expert * frac_per_expert)
            
            # 4. Entropy Loss: Encourage uniform probability distribution
            # Higher entropy = more uniform = better load balancing
            entropy = -torch.sum(prob_per_expert * torch.log(prob_per_expert + 1e-8))
            max_entropy = torch.log(torch.tensor(self.num_slots, dtype=torch.float32, device=x.device))
            entropy_loss = 1.0 - (entropy / max_entropy)  # 0 when uniform, 1 when collapsed
            
            # 5. Z-Loss: Penalize large logits to prevent one expert dominating
            z_loss = torch.logsumexp(logits, dim=-1).mean() ** 2 * 1e-4
            
            # Combined loss
            aux_loss = switch_loss + entropy_loss * 0.5 + z_loss

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

        if return_routing_stats:
            # Compute routing statistics
            # slot_counts: How many patches are assigned to each slot (based on top-k)
            slot_counts = mask.sum(dim=1).mean(dim=0)  # [num_slots], averaged over batch
            routing_stats = {
                'slot_counts': slot_counts.detach().cpu(),  # [num_slots]
                'importance': probs.sum(dim=1).mean(dim=0).detach().cpu(),  # [num_slots]
                'aux_loss': aux_loss.item() if aux_loss.numel() == 1 else aux_loss.detach().cpu(),
                'num_patches': N,
            }
            return final_tokens, aux_loss, routing_stats

        return final_tokens, aux_loss
