"""
WSI Classifier using MoE Token Compression.

Complete model for Whole Slide Image classification.
"""

import torch
import torch.nn as nn
from .moe_compressor import MoE_Compressor


class WSI_Classifier(nn.Module):
    """
    Complete WSI Classification Model with MoE Token Compression.

    Architecture:
        Input [1, N, 1024] (variable N)
          ↓
        MoE Compressor [1, num_slots, 1024] (fixed size)
          ↓
        Mean Pooling [1, 1024]
          ↓
        MLP Classifier [1, num_classes]
          ↓
        Output: Logits
    """

    def __init__(self, num_classes=2, input_dim=1024, num_slots=64):
        """
        Initialize WSI Classifier.

        Args:
            num_classes: Number of output classes
            input_dim: Feature dimension (e.g., 1024 for UNI, 768 for CTransPath)
            num_slots: Number of compressed tokens
        """
        super().__init__()

        # A. Compression Layer (Core Innovation)
        self.compressor = MoE_Compressor(num_slots=num_slots, input_dim=input_dim)

        # B. Classification Head
        # Since input is compressed to [num_slots, D], we can use simple Pooling + MLP
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features [1, N, 1024] where N varies per slide

        Returns:
            logits: Classification logits [1, num_classes]
            aux_loss: Auxiliary load balancing loss (scalar)
        """
        # 1. Compression: [1, N, 1024] -> [1, num_slots, 1024]
        compressed_x, aux_loss = self.compressor(x)

        # 2. Aggregation: Pool across all expert slots
        # Mean pooling combines opinions from all experts
        slide_repr = compressed_x.mean(dim=1)  # [1, 1024]

        # 3. Classification
        logits = self.classifier(slide_repr)  # [1, num_classes]

        return logits, aux_loss


class SimpleMILBaseline(nn.Module):
    """
    Simple Multiple Instance Learning baseline using attention pooling.
    Provided for comparison with the MoE approach.
    """

    def __init__(self, num_classes=2, input_dim=1024, hidden_dim=512, dropout=0.25):
        super().__init__()

        # Attention mechanism for pooling
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features [1, N, input_dim]

        Returns:
            logits: Classification logits [1, num_classes]
            aux_loss: Dummy loss (0.0) for compatibility
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # [1, N, 1]
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Attention-weighted pooling
        pooled_features = (x * attention_weights).sum(dim=1)  # [1, input_dim]

        # Classification
        logits = self.classifier(pooled_features)

        # Return dummy aux_loss for compatibility with training loop
        aux_loss = torch.tensor(0.0, device=x.device)

        return logits, aux_loss


def build_model(model_type='moe', num_classes=2, input_dim=1024, num_slots=64, **kwargs):
    """
    Factory function to build models.

    Args:
        model_type: Type of model ('moe' or 'mil_baseline')
        num_classes: Number of output classes
        input_dim: Feature dimension
        num_slots: Number of expert slots (only for MoE)
        **kwargs: Additional arguments

    Returns:
        Initialized model

    Example:
        >>> model = build_model('moe', num_classes=2, num_slots=64)
        >>> model = build_model('mil_baseline', num_classes=5)
    """
    if model_type.lower() == 'moe':
        return WSI_Classifier(
            num_classes=num_classes,
            input_dim=input_dim,
            num_slots=num_slots
        )
    elif model_type.lower() == 'mil_baseline':
        return SimpleMILBaseline(
            num_classes=num_classes,
            input_dim=input_dim
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Choose 'moe' or 'mil_baseline'"
        )
