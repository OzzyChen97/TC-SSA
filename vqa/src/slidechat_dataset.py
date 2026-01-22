"""
SlideChat Dataset Adapter for WSI-VQA.

Adapts the SlideChat data format to work with our VQA framework.

SlideChat Data Format:
    - JSON with conversations format
    - Image paths point to CSV files (e.g., "./BLCA/TCGA-XXX.csv")
    - Features stored separately in tar.gz archives

Our Format:
    - Direct path to features (.pt or .npy)
    - Standard question-answer pairs
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
import numpy as np


class SlideChatDataset(Dataset):
    """
    Dataset adapter for SlideChat format.

    Handles both Stage 1 (caption) and Stage 2 (VQA) data.
    """

    def __init__(
        self,
        data_path,
        features_dir,
        tokenizer,
        mode='caption',
        max_length=512,
        num_visual_tokens=16,
        visual_dim=512,
        ignore_index=-100,
        feature_suffix=1024
    ):
        """
        Args:
            data_path: Path to SlideInstruct JSON file
            features_dir: Directory containing extracted features
            tokenizer: HuggingFace tokenizer
            mode: 'caption' or 'vqa'
            max_length: Maximum sequence length
            num_visual_tokens: Number of visual tokens from MoE
            visual_dim: Visual feature dimension (512 for CONCH)
            ignore_index: Label index to ignore in loss
            feature_suffix: Suffix in filename (1024 for more patches via _0_1024.npy)
        """
        self.features_dir = features_dir
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.num_visual_tokens = num_visual_tokens
        self.visual_dim = visual_dim
        self.ignore_index = ignore_index
        self.feature_suffix = feature_suffix

        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Features directory: {features_dir}")
        print(f"Using feature files with suffix: *_{feature_suffix}.npy (prefer _0_)")

        # Ensure <image> token exists
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    def _get_features_path(self, image_path):
        """
        Convert SlideChat image path to features path.

        Args:
            image_path: Original path like "./BLCA/TCGA-XXX.csv"

        Returns:
            Full path to features file
        """
        # Parse the image path: "./BLCA/TCGA-GV-A40G-01Z-00-DX1.csv"
        rel_path = image_path.lstrip('./')
        parts = rel_path.split('/')

        if len(parts) >= 2:
            cancer_type = parts[0]  # e.g., "BLCA"
            csv_filename = parts[1]  # e.g., "TCGA-GV-A40G-01Z-00-DX1.csv"
            slide_id = csv_filename.replace('.csv', '')  # e.g., "TCGA-GV-A40G-01Z-00-DX1"
        else:
            # Fallback
            slide_id = os.path.basename(rel_path).replace('.csv', '')
            cancer_type = None

        # Map cancer types to TCGA directory names
        tcga_mapping = {
            'BLCA': 'TCGA-BLCA',
            'BRCA': 'TCGA-BR',
            'COAD': 'TCGA-COAD',
            'READ': 'TCGA-COAD',  # READ is often grouped with COAD
            'GBM': 'TCGA-GBM',
            'HNSC': 'TCGA-HNSC',
            'LGG': 'TCGA-LGG',
            'LUAD': 'TCGA-LUNG',
            'LUSC': 'TCGA-LUNG',
            'SKCM': 'TCGA-SKCM',
        }

        # Build possible directory paths
        possible_dirs = []
        if cancer_type:
            tcga_dir = tcga_mapping.get(cancer_type, f'TCGA-{cancer_type}')
            # Check both single and double nested directories
            possible_dirs.extend([
                os.path.join(self.features_dir, tcga_dir, tcga_dir),  # Double nested
                os.path.join(self.features_dir, tcga_dir, 'feat'),    # Standard structure
                os.path.join(self.features_dir, tcga_dir),             # Direct
            ])

        # Also check TCGA-Rest for other cancer types
        possible_dirs.extend([
            os.path.join(self.features_dir, 'TCGA-Rest', 'TCGA-Rest'),
            os.path.join(self.features_dir, 'TCGA-Rest', 'feat'),
            os.path.join(self.features_dir, 'TCGA-Rest'),
        ])

        # Search for feature files matching the slide_id
        # Files may have format: TCGA-XX-XXXX.UUID_N_DIM.npy
        for search_dir in possible_dirs:
            if not os.path.exists(search_dir):
                continue

            # List all files in directory that start with slide_id
            try:
                files = os.listdir(search_dir)
                matching_files = [f for f in files if f.startswith(slide_id)]

                if matching_files:
                    # Priority order: prefer _0_ (more patches) and prefer feature_suffix
                    # Full fallback chain: _0_{suffix} > _1_{suffix} > _0_other > _1_other > any
                    priority_patterns = [
                        f'_0_{self.feature_suffix}.npy',  # Best: _0_ with preferred suffix
                        f'_1_{self.feature_suffix}.npy',  # Second: _1_ with preferred suffix
                        '_0_1024.npy',  # Fallback: _0_ with 1024
                        '_0_512.npy',   # Fallback: _0_ with 512
                        '_1_1024.npy',  # Fallback: _1_ with 1024
                        '_1_512.npy',   # Fallback: _1_ with 512
                    ]
                    
                    for pattern in priority_patterns:
                        for fname in matching_files:
                            if pattern in fname:
                                return os.path.join(search_dir, fname)
                    
                    # Last resort: return first matching .npy file
                    for fname in matching_files:
                        if fname.endswith('.npy'):
                            return os.path.join(search_dir, fname)
                    
                    # Absolute last resort
                    return os.path.join(search_dir, matching_files[0])
            except (OSError, PermissionError):
                continue

        # If still not found, try simple direct paths as fallback
        for ext in ['.pt', '.pth', '.npy', '.csv']:
            feat_path = os.path.join(self.features_dir, f"{slide_id}{ext}")
            if os.path.exists(feat_path):
                return feat_path

        # Return a default path (will raise error in _load_patch_features)
        return os.path.join(self.features_dir, f"{slide_id}.npy")

    def _load_patch_features(self, features_path):
        """
        Load precomputed patch features.

        Args:
            features_path: Path to features file

        Returns:
            features: [N, D] tensor (D=512 for ConCH, 1024 for UNI)
        """
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features not found: {features_path}")

        # Try different formats
        if features_path.endswith('.pt') or features_path.endswith('.pth'):
            features = torch.load(features_path, map_location='cpu')
        elif features_path.endswith('.npy'):
            data = np.load(features_path, allow_pickle=True)
            # data could be a 0-d array wrapping a dict
            if data.ndim == 0 and data.dtype == object:
                try:
                    data_dict = data.item()
                    if 'feature' in data_dict:
                        feat_obj = data_dict['feature']
                        if isinstance(feat_obj, torch.Tensor):
                            features = feat_obj
                        elif isinstance(feat_obj, np.ndarray):
                            features = torch.from_numpy(feat_obj)
                        else:
                            # Try converting whatever it is
                            features = torch.tensor(feat_obj)
                    else:
                        features = torch.from_numpy(data) # Fallback
                except Exception as e:
                    # If extraction fails, try using data directly if compatible
                    # But if data is object array, this fails. 
                    raise ValueError(f"Failed to extract features from {features_path}: {e}")
            else:
                features = torch.from_numpy(data)
        elif features_path.endswith('.csv'):
            # SlideChat stores features as CSV
            df = pd.read_csv(features_path, header=None)
            features = torch.from_numpy(df.values)
        else:
            raise ValueError(f"Unsupported format: {features_path}")

        # Ensure float32
        if features.dtype != torch.float32:
            features = features.float()

        # Handle different feature dimensions
        # Handle different feature dimensions
        feat_dim = features.shape[-1]
        
        if feat_dim == self.visual_dim:
            # Correct dimension, no changes needed
            pass
        elif feat_dim == 512 and self.visual_dim == 1024:
            # ConCH features (512) -> UNI dimension (1024)
            # Repeat features to match expected dimension
            features = torch.cat([features, features], dim=-1)
        else:
            raise ValueError(f"Unexpected feature dimension: {feat_dim}, expected {self.visual_dim} (or 512->1024 expansion)")

        return features

    def _create_sample(self, item):
        """
        Create training sample from SlideChat format.

        Args:
            item: Dict with 'image', 'conversations', 'id'

        Returns:
            Dict with input_ids, attention_mask, labels, patch_features
        """
        # Extract conversation
        conversations = item['conversations']
        image_path = item['image'][0]  # First image

        # Find question and answer
        question = None
        answer = None

        for conv in conversations:
            if conv['from'] == 'human':
                question = conv['value']
            elif conv['from'] == 'gpt':
                answer = conv['value']

        if question is None or answer is None:
            raise ValueError(f"Invalid conversation format in {item['id']}")

        # Remove <image> from question if exists, we'll add it back
        question = question.replace('<image>', '').strip()

        # Construct prompt with <image> at the beginning
        prompt = f"<image> {question}"

        # Tokenize conversation
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer}
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback format
            full_text = f"User: {prompt}\nAssistant: {answer}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Create labels: mask prompt, only train on response
        labels = input_ids.clone()

        # Tokenize prompt to find where to mask
        prompt_encoding = self.tokenizer(
            prompt if not hasattr(self.tokenizer, 'apply_chat_template') else prompt,
            add_special_tokens=False
        )
        prompt_len = len(prompt_encoding['input_ids'])

        # Mask prompt tokens
        labels[:prompt_len] = self.ignore_index

        # Mask padding tokens
        labels[attention_mask == 0] = self.ignore_index

        # Load features
        features_path = self._get_features_path(image_path)
        patch_features = self._load_patch_features(features_path)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'patch_features': patch_features,
            'slide_id': item['id']
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single training sample."""
        return self._create_sample(self.data[idx])

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for training.
        """
        max_patches = max([item['patch_features'].shape[0] for item in batch])

        # Pad patch features
        padded_features = []
        for item in batch:
            features = item['patch_features']
            N = features.shape[0]

            if N < max_patches:
                padding = torch.zeros(max_patches - N, features.shape[1])
                features = torch.cat([features, padding], dim=0)

            padded_features.append(features)

        # Stack batch
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': torch.stack([item['labels'] for item in batch]),
            'patch_features': torch.stack(padded_features),
            'slide_id': [item['slide_id'] for item in batch]
        }


def benchmark_collate_fn(batch):
    """
    Collate function for benchmark evaluation.
    """
    max_patches = max([item['patch_features'].shape[0] for item in batch])
    max_seq_len = max([item['input_ids'].shape[0] for item in batch])

    # Pad patch features
    padded_features = []
    for item in batch:
        features = item['patch_features']
        N = features.shape[0]
        if N < max_patches:
            padding = torch.zeros(max_patches - N, features.shape[1])
            features = torch.cat([features, padding], dim=0)
        padded_features.append(features)

    # Pad sequences
    padded_input_ids = []
    padded_attention_mask = []
    pad_token_id = batch[0]['input_ids'][0].item() if len(batch) > 0 else 0

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        L = input_ids.shape[0]

        if L < max_seq_len:
            pad_len = max_seq_len - L
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)

    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'patch_features': torch.stack(padded_features),
        'slide_id': [item['slide_id'] for item in batch],
        'question': [item['question'] for item in batch],
        'choices': [item['choices'] for item in batch],
        'answer': [item['answer'] for item in batch],
        'category': [item['category'] for item in batch]
    }
