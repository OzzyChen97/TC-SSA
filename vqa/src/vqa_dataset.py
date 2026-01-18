"""
VQA Dataset for WSI Question Answering.

Supports two modes:
    - 'caption': WSI captioning for Stage 1 pretraining
    - 'vqa': Visual Question Answering for Stage 2 finetuning
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import os
import numpy as np


class SlideChatDataset(Dataset):
    """
    Dataset for WSI-VQA tasks.

    Data Format:
        - Caption mode: CSV with columns [slide_id, caption, features_path]
        - VQA mode: JSON with [{"slide_id", "question", "answer", "features_path"}]
    """

    def __init__(
        self,
        data_path,
        tokenizer,
        mode='caption',
        max_length=512,
        num_visual_tokens=16,
        ignore_index=-100
    ):
        """
        Args:
            data_path: Path to data file (CSV for caption, JSON for VQA)
            tokenizer: HuggingFace tokenizer
            mode: 'caption' or 'vqa'
            max_length: Maximum sequence length
            num_visual_tokens: Number of visual tokens from MoE
            ignore_index: Label index to ignore in loss computation
        """
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.num_visual_tokens = num_visual_tokens
        self.ignore_index = ignore_index

        # Load data
        if mode == 'caption':
            self.data = pd.read_csv(data_path)
            print(f"Loaded {len(self.data)} caption samples from {data_path}")
        elif mode == 'vqa':
            with open(data_path, 'r') as f:
                self.data = json.load(f)
            print(f"Loaded {len(self.data)} VQA samples from {data_path}")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Ensure <image> token exists
        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    def __len__(self):
        return len(self.data)

    def _load_patch_features(self, features_path):
        """
        Load precomputed patch features.

        Args:
            features_path: Path to .pt or .npy file with features

        Returns:
            features: [N, 1024] tensor
        """
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features not found: {features_path}")

        if features_path.endswith('.pt'):
            features = torch.load(features_path, map_location='cpu')
        elif features_path.endswith('.npy'):
            features = torch.from_numpy(np.load(features_path))
        else:
            raise ValueError(f"Unsupported format: {features_path}")

        # Ensure float32
        if features.dtype != torch.float32:
            features = features.float()

        return features

    def _create_caption_sample(self, item):
        """
        Create training sample for caption mode.

        Prompt format: "<image> Describe this slide."
        Response: "{caption}"
        """
        slide_id = item['slide_id']
        caption = item['caption']
        features_path = item['features_path']

        # Construct conversation
        prompt = "<image> Describe this slide."
        response = caption

        # Tokenize
        # We'll use the chat template if available, otherwise manual format
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # Fallback manual format
            full_text = f"User: {prompt}\nAssistant: {response}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)  # [L]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [L]

        # Create labels: mask out prompt tokens, only train on response
        labels = input_ids.clone()

        # Find where response starts (after "Assistant:" or equivalent)
        # Simple heuristic: mask everything before the last occurrence of assistant's first token
        # For robustness, we'll mask everything except the caption part
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        prompt_len = len(prompt_tokens)

        # Mask prompt tokens
        labels[:prompt_len] = self.ignore_index

        # Mask padding tokens
        labels[attention_mask == 0] = self.ignore_index

        # Load features
        patch_features = self._load_patch_features(features_path)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'patch_features': patch_features,
            'slide_id': slide_id
        }

    def _create_vqa_sample(self, item):
        """
        Create training sample for VQA mode.

        Prompt format: "<image> {question}"
        Response: "{answer}"
        """
        slide_id = item['slide_id']
        question = item['question']
        answer = item['answer']
        features_path = item['features_path']

        # Construct conversation
        prompt = f"<image> {question}"
        response = answer

        # Tokenize
        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            full_text = f"User: {prompt}\nAssistant: {response}"

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

        # Create labels
        labels = input_ids.clone()

        # Mask prompt
        prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)['input_ids']
        prompt_len = len(prompt_tokens)
        labels[:prompt_len] = self.ignore_index

        # Mask padding
        labels[attention_mask == 0] = self.ignore_index

        # Load features
        patch_features = self._load_patch_features(features_path)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'patch_features': patch_features,
            'slide_id': slide_id,
            'question': question,
            'answer': answer
        }

    def __getitem__(self, idx):
        """
        Get a single training sample.

        Returns:
            dict with keys: input_ids, attention_mask, labels, patch_features
        """
        if self.mode == 'caption':
            return self._create_caption_sample(self.data.iloc[idx])
        elif self.mode == 'vqa':
            return self._create_vqa_sample(self.data[idx])


def collate_fn(batch):
    """
    Custom collate function to handle variable-length patch features.

    Args:
        batch: List of samples from __getitem__

    Returns:
        Batched tensors with padding
    """
    # Get max number of patches in this batch
    max_patches = max([item['patch_features'].shape[0] for item in batch])

    # Pad patch features
    padded_features = []
    for item in batch:
        features = item['patch_features']  # [N, 1024]
        N = features.shape[0]

        if N < max_patches:
            # Pad with zeros
            padding = torch.zeros(max_patches - N, features.shape[1])
            features = torch.cat([features, padding], dim=0)

        padded_features.append(features)

    # Stack batch
    batch_dict = {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'patch_features': torch.stack(padded_features),
        'slide_id': [item['slide_id'] for item in batch]
    }

    # Add VQA-specific fields if present
    if 'question' in batch[0]:
        batch_dict['question'] = [item['question'] for item in batch]
        batch_dict['answer'] = [item['answer'] for item in batch]

    return batch_dict


class BenchmarkDataset(Dataset):
    """
    Dataset for SlideBench evaluation (TCGA benchmark).

    Data format (JSON):
    {
        "samples": [
            {
                "slide_id": "TCGA-XX-XXXX",
                "question": "What is the primary diagnosis?",
                "choices": ["A) Adenocarcinoma", "B) Squamous cell carcinoma", ...],
                "answer": "A",
                "category": "Diagnosis",
                "features_path": "/path/to/features.pt"
            },
            ...
        ]
    }
    """

    def __init__(self, json_path, tokenizer, num_visual_tokens=16):
        """
        Args:
            json_path: Path to benchmark JSON file
            tokenizer: HuggingFace tokenizer
            num_visual_tokens: Number of visual tokens
        """
        self.tokenizer = tokenizer
        self.num_visual_tokens = num_visual_tokens

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.samples = data['samples'] if 'samples' in data else data
        print(f"Loaded {len(self.samples)} benchmark samples from {json_path}")

        self.image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get benchmark sample for evaluation.

        Returns:
            dict with all necessary fields for inference
        """
        sample = self.samples[idx]

        slide_id = sample['slide_id']
        question = sample['question']
        choices = sample['choices']
        answer = sample['answer']
        category = sample.get('category', 'Unknown')
        features_path = sample['features_path']

        # Format multiple choice prompt
        choices_text = "\n".join(choices)
        prompt = f"<image> {question}\n{choices_text}\nAnswer with only the letter (A/B/C/D):"

        # Tokenize
        encoding = self.tokenizer(
            prompt,
            return_tensors='pt',
            padding=False,
            truncation=False
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Load features
        if features_path.endswith('.pt'):
            patch_features = torch.load(features_path, map_location='cpu')
        elif features_path.endswith('.npy'):
            patch_features = torch.from_numpy(np.load(features_path))
        else:
            raise ValueError(f"Unsupported format: {features_path}")

        if patch_features.dtype != torch.float32:
            patch_features = patch_features.float()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'patch_features': patch_features,
            'slide_id': slide_id,
            'question': question,
            'choices': choices,
            'answer': answer,
            'category': category
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

    # Pad input sequences
    padded_input_ids = []
    padded_attention_mask = []
    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        L = input_ids.shape[0]

        if L < max_seq_len:
            # Pad
            pad_len = max_seq_len - L
            input_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), batch[0]['input_ids'][0].item())  # pad_token_id
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(pad_len, dtype=torch.long)
            ])

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
