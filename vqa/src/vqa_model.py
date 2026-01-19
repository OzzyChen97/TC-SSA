"""
MoE-Qwen VQA Model for WSI Question Answering.

Architecture:
    - Visual Encoder: MoE_Compressor (Frozen) -> [B, 16, 1024]
    - Projector: MLP (Trainable) -> [B, 16, hidden_size]
    - LLM: Qwen2.5-3B-Instruct (Stage 1: Frozen, Stage 2: Trainable)
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
from peft import get_peft_model, LoraConfig, TaskType

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models import MoE_Compressor


class MLPProjector(nn.Module):
    """
    MLP Projector to align visual features to LLM embedding space.
    """
    def __init__(self, input_dim=1024, output_dim=2560, hidden_dim=2048):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: [B, num_tokens, input_dim]
        Returns:
            [B, num_tokens, output_dim]
        """
        return self.proj(x)


class MoE_Qwen_VQA(nn.Module):
    """
    Vision-Language Model combining MoE_Compressor and Qwen2.5-3B.

    Training Strategy:
        - Stage 1: Freeze LLM, train Projector only (caption pretraining)
        - Stage 2: Unfreeze LLM, full finetuning (VQA finetuning)
    """

    def __init__(
        self,
        moe_checkpoint,
        llm_path="/workspace/ETC/vqa/data/Qwen3-4B-Instruct-2507",
        num_visual_tokens=16,
        moe_num_slots=32,
        visual_dim=1024,
        device='cuda'
    ):
        """
        Args:
            moe_checkpoint: Path to pretrained MoE_Compressor weights
            llm_path: HuggingFace model path for Qwen
            num_visual_tokens: Number of compressed visual tokens (default 16)
            moe_num_slots: Number of slots in MoE_Compressor
            visual_dim: Dimension of visual features (1024 for UNI)
            device: Device to load models on
        """
        super().__init__()

        self.num_visual_tokens = num_visual_tokens
        self.device = device
        self.llm_path = llm_path # Save for debugging

        # 1. Load Visual Encoder (Frozen)
        print(f"Loading MoE_Compressor from {moe_checkpoint}...")
        self.visual_encoder = MoE_Compressor(
            num_slots=moe_num_slots,
            top_k=2,
            input_dim=visual_dim
        )

        # Load pretrained weights
        checkpoint = torch.load(moe_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Extract only MoE_Compressor weights (filter out classifier head)
            moe_state = {k.replace('compressor.', ''): v
                        for k, v in state_dict.items()
                        if k.startswith('compressor')}
            self.visual_encoder.load_state_dict(moe_state)
        else:
            self.visual_encoder.load_state_dict(checkpoint)

        # Freeze visual encoder
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        self.visual_encoder.eval()

        # 2. Load LLM
        print(f"Loading LLM from {llm_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            trust_remote_code=True,
            use_fast=False
        )
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None  # We'll handle device placement manually for DDP
        )

        # Add special image token if not exists
        if "<image>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})
            self.llm.resize_token_embeddings(len(self.tokenizer))

        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")

        # 3. Initialize Projector
        llm_hidden_size = self.llm.config.hidden_size
        print(f"Initializing Projector: {visual_dim} -> {llm_hidden_size}")
        self.projector = MLPProjector(
            input_dim=visual_dim,
            output_dim=llm_hidden_size,
            hidden_dim=2048
        )

        # Enable gradient checkpointing to save memory
        self.llm.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for LLM.")

        # 4. Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            # Fix Qwen padding side if needed (usually left for generation, right for training)
            self.tokenizer.padding_side = 'right'

    def freeze_llm(self):
        """Freeze LLM parameters (Stage 1)."""
        for param in self.llm.parameters():
            param.requires_grad = False
        print("LLM frozen.")

    def enable_lora(self, r=16, lora_alpha=32, lora_dropout=0.05):
        """Enable LoRA for Stage 2."""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Target all Linear layers in Qwen
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        print("LoRA enabled.")

    def unfreeze_llm(self):
        """Unfreeze LLM parameters (Stage 2)."""
        for param in self.llm.parameters():
            param.requires_grad = True
        print("LLM unfrozen.")

    def encode_images(self, patch_features):
        """
        Encode WSI patches into visual tokens.

        Args:
            patch_features: [B, N, 1024] - Patch features from UNI/ResNet

        Returns:
            visual_embeds: [B, num_visual_tokens, llm_hidden_size]
        """
        with torch.no_grad():
            # Get MoE compressed tokens
            visual_tokens, _ = self.visual_encoder(patch_features)  # [B, 16, 1024]

        # Project to LLM space
        visual_embeds = self.projector(visual_tokens)  # [B, 16, llm_hidden_size]
        return visual_embeds

    def prepare_model_inputs(self, input_ids, patch_features, attention_mask, labels=None):
        """
        Prepare inputs for the model by inserting visual embeddings and adjusting masks/labels.
        
        Args:
            input_ids: [B, L]
            patch_features: [B, N, 1024]
            attention_mask: [B, L]
            labels: [B, L] or None
            
        Returns:
            inputs_embeds: [B, new_L, hidden_size]
            new_attention_mask: [B, new_L]
            new_labels: [B, new_L] or None
        """
        # Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, hidden_size]
        
        # Get visual embeddings
        visual_embeds = self.encode_images(patch_features)  # [B, num_visual_tokens, hidden_size]
        
        B, L, D = text_embeds.shape
        num_visual_tokens = visual_embeds.shape[1]
        
        # Calculate new length: L - 1 (remove <image>) + num_visual_tokens
        # We assume exactly one <image> token per sequence for now as per SlideChatDataset
        new_L = L + num_visual_tokens - 1
        
        new_inputs_embeds_list = []
        new_attention_mask_list = []
        new_labels_list = []
        
        # Find <image> token positions
        image_positions = (input_ids == self.image_token_id)
        
        for b in range(B):
            img_pos = torch.where(image_positions[b])[0]
            
            if len(img_pos) > 0:
                insert_pos = img_pos[0].item()
                
                # 1. Construct Embeddings
                # [Pre-Image Text] + [Visual Tokens] + [Post-Image Text]
                embeds_b = torch.cat([
                    text_embeds[b, :insert_pos],
                    visual_embeds[b],
                    text_embeds[b, insert_pos+1:]
                ], dim=0)
                
                # 2. Construct Attention Mask
                # [Pre-Mask] + [1s for Visual] + [Post-Mask]
                mask_b = torch.cat([
                    attention_mask[b, :insert_pos],
                    torch.ones(num_visual_tokens, device=self.device, dtype=attention_mask.dtype),
                    attention_mask[b, insert_pos+1:]
                ], dim=0)
                
                # 3. Construct Labels (if provided)
                if labels is not None:
                    # [Pre-Labels] + [Ignore_Index for Visual] + [Post-Labels]
                    # We ignore loss on visual tokens
                    ignore_tensor = torch.full((num_visual_tokens,), -100, device=self.device, dtype=labels.dtype)
                    
                    lbl_b = torch.cat([
                        labels[b, :insert_pos],
                        ignore_tensor,
                        labels[b, insert_pos+1:]
                    ], dim=0)
                    new_labels_list.append(lbl_b)
                    
            else:
                # No image token found (fallback)
                embeds_b = text_embeds[b]
                mask_b = attention_mask[b]
                if labels is not None:
                    new_labels_list.append(labels[b])
            
            new_inputs_embeds_list.append(embeds_b)
            new_attention_mask_list.append(mask_b)
            
        # Stack
        new_inputs_embeds = torch.stack(new_inputs_embeds_list)
        new_attention_mask = torch.stack(new_attention_mask_list)
        
        new_labels = None
        if labels is not None:
            new_labels = torch.stack(new_labels_list)
            
        return new_inputs_embeds, new_attention_mask, new_labels

    def forward(self, input_ids, attention_mask, labels, patch_features):
        """
        Forward pass for training.

        Args:
            input_ids: [B, L] - Tokenized text with <image> placeholder
            attention_mask: [B, L] - Attention mask
            labels: [B, L] - Labels for language modeling (-100 for ignored tokens)
            patch_features: [B, N, 1024] - WSI patch features

        Returns:
            loss: Scalar loss
        """
        # Prepare inputs with visual token insertion and alignment
        inputs_embeds, new_attention_mask, new_labels = self.prepare_model_inputs(
            input_ids, patch_features, attention_mask, labels
        )

        # Convert to LLM dtype (bfloat16)
        inputs_embeds = inputs_embeds.to(dtype=self.llm.dtype)

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            return_dict=True
        )

        return outputs.loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        patch_features,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    ):
        """
        Generate text response for inference.

        Args:
            input_ids: [B, L] - Tokenized prompt with <image>
            attention_mask: [B, L] - Attention mask
            patch_features: [B, N, 1024] - WSI patch features
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            generated_ids: [B, L+new_tokens] - Generated token IDs
        """
        self.eval()

        # Get combined embeddings and expanded mask
        # Note: labels is None for generation
        inputs_embeds, new_attention_mask, _ = self.prepare_model_inputs(
            input_ids, patch_features, attention_mask, labels=None
        )

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return outputs

    def save_pretrained(self, save_dir):
        """Save model checkpoints."""
        os.makedirs(save_dir, exist_ok=True)

        # Save projector
        torch.save(self.projector.state_dict(),
                  os.path.join(save_dir, "projector.pt"))

        # Save LLM (only if finetuned)
        self.llm.save_pretrained(os.path.join(save_dir, "llm"))
        self.tokenizer.save_pretrained(os.path.join(save_dir, "llm"))

        print(f"Model saved to {save_dir}")

    def load_pretrained(self, save_dir):
        """Load model checkpoints."""
        # Load projector
        self.projector.load_state_dict(
            torch.load(os.path.join(save_dir, "projector.pt"))
        )

        # Load LLM
        self.llm = AutoModelForCausalLM.from_pretrained(
            os.path.join(save_dir, "llm"),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None
        )

        print(f"Model loaded from {save_dir}")
