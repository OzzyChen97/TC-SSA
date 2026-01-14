"""
Dataset module for loading pre-extracted WSI patch features.
Handles .pt files containing patch embeddings for Whole Slide Images.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import warnings


class WSIFeatureDataset(Dataset):
    """
    PyTorch Dataset for loading pre-extracted WSI patch features.

    The dataset expects:
    - A CSV file with columns: slide_id, label
    - A directory containing .pt files named as {slide_id}.pt
    - .pt files can be either:
        * Dictionaries with 'features' key: {'features': tensor}
        * Direct tensors: tensor

    Each feature tensor should have shape [N, feature_dim] where N varies per slide.
    """

    def __init__(
        self,
        csv_path: str,
        features_dir: str,
        feature_dim: int = 1024,
        label_mapping: Optional[dict] = None
    ):
        """
        Initialize the WSI Feature Dataset.

        Args:
            csv_path: Path to CSV file with columns [slide_id, label]
            features_dir: Directory containing .pt feature files
            feature_dim: Expected feature dimension (default: 1024 for UNI/CTransPath)
            label_mapping: Optional dictionary to map string labels to integers
                          If None, assumes labels are already numeric
        """
        self.csv_path = csv_path
        self.features_dir = features_dir
        self.feature_dim = feature_dim
        self.label_mapping = label_mapping

        # Load CSV metadata
        self.metadata = pd.read_csv(csv_path)

        # Validate CSV columns
        required_cols = ['slide_id', 'label']
        for col in required_cols:
            if col not in self.metadata.columns:
                raise ValueError(
                    f"CSV file must contain '{col}' column. "
                    f"Found columns: {list(self.metadata.columns)}"
                )

        # Process labels
        if label_mapping is not None:
            self.metadata['label'] = self.metadata['label'].map(label_mapping)
            # Check for unmapped labels
            if self.metadata['label'].isna().any():
                raise ValueError("Some labels could not be mapped using provided label_mapping")

        # Ensure labels are integers
        self.metadata['label'] = self.metadata['label'].astype(int)

        # Validate that feature files exist
        self._validate_files()

        print(f"Loaded dataset with {len(self)} samples")
        print(f"Label distribution:\n{self.metadata['label'].value_counts().sort_index()}")

    def _validate_files(self):
        """Validate that all .pt files exist and log missing ones."""
        missing_files = []
        for idx, row in self.metadata.iterrows():
            slide_id = row['slide_id']
            feature_path = os.path.join(self.features_dir, f"{slide_id}.pt")
            if not os.path.exists(feature_path):
                missing_files.append(slide_id)

        if missing_files:
            warnings.warn(
                f"Missing {len(missing_files)} feature files out of {len(self.metadata)}. "
                f"First few missing: {missing_files[:5]}"
            )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (features, label, slide_id)
            - features: Tensor of shape [N, feature_dim]
            - label: Integer label
            - slide_id: String identifier for the slide
        """
        row = self.metadata.iloc[idx]
        slide_id = row['slide_id']
        label = row['label']

        # Load feature file
        feature_path = os.path.join(self.features_dir, f"{slide_id}.pt")

        try:
            data = torch.load(feature_path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Feature file not found: {feature_path} for slide_id: {slide_id}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error loading {feature_path} for slide_id {slide_id}: {str(e)}"
            )

        # Handle both dictionary and direct tensor formats
        if isinstance(data, dict):
            # Dictionary format: {'features': tensor, ...}
            if 'features' in data:
                features = data['features']
            elif 'feat' in data:
                features = data['feat']
            else:
                # Try to find any tensor in the dictionary
                tensor_keys = [k for k, v in data.items() if isinstance(v, torch.Tensor)]
                if tensor_keys:
                    features = data[tensor_keys[0]]
                    warnings.warn(
                        f"Using key '{tensor_keys[0]}' for slide {slide_id}. "
                        f"Expected 'features' key."
                    )
                else:
                    raise ValueError(
                        f"No tensor found in dictionary for {slide_id}. Keys: {list(data.keys())}"
                    )
        elif isinstance(data, torch.Tensor):
            # Direct tensor format
            features = data
        else:
            raise TypeError(
                f"Unexpected data type in {feature_path}: {type(data)}. "
                f"Expected dict or torch.Tensor"
            )

        # Ensure features are 2D: [N, feature_dim]
        if features.dim() == 1:
            features = features.unsqueeze(0)  # [feature_dim] -> [1, feature_dim]
        elif features.dim() == 3:
            # Handle case where features might be [1, N, feature_dim]
            if features.size(0) == 1:
                features = features.squeeze(0)  # [1, N, feature_dim] -> [N, feature_dim]

        # Validate shape
        if features.dim() != 2:
            raise ValueError(
                f"Features for {slide_id} have unexpected shape: {features.shape}. "
                f"Expected [N, {self.feature_dim}]"
            )

        if features.size(1) != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch for {slide_id}. "
                f"Expected {self.feature_dim}, got {features.size(1)}"
            )

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return features, label, slide_id

    def get_num_classes(self) -> int:
        """Return the number of unique classes in the dataset."""
        return self.metadata['label'].nunique()

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets.
        Uses inverse frequency weighting.

        Returns:
            Tensor of class weights
        """
        class_counts = self.metadata['label'].value_counts().sort_index().values
        total = len(self.metadata)
        weights = total / (len(class_counts) * class_counts)
        return torch.tensor(weights, dtype=torch.float32)


def collate_fn_variable_length(batch):
    """
    Custom collate function to handle variable-length sequences.

    Since WSI slides have different numbers of patches (N varies),
    we cannot batch them directly. This collate function handles
    batching by returning lists for variable-length tensors.

    Args:
        batch: List of tuples (features, label, slide_id)

    Returns:
        Tuple of (features_list, labels_tensor, slide_ids_list)
    """
    features_list = []
    labels_list = []
    slide_ids_list = []

    for features, label, slide_id in batch:
        features_list.append(features)
        labels_list.append(label)
        slide_ids_list.append(slide_id)

    # Stack labels into a tensor
    labels_tensor = torch.stack(labels_list)

    return features_list, labels_tensor, slide_ids_list
