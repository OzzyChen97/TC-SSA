"""
cd /workspace/zhuo/ETC

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 vqa/tools/train_slidechat_stage2.py \
    --moe_checkpoint /workspace/zhuo/ETC/outputs/moe_tcga_9class_experiment/best_model.pth \
    --llm_path /workspace/jhsheng/huggingface/models/Qwen/Qwen2.5-7B-Instruct/ \
    --projector_checkpoint /workspace/zhuo/ETC/vqa/outputs/slidechat_stage1_7B_moe_finetune1/final/projector.pt \
    --data_path vqa/data/SlideChat/SlideInstruct_train_stage2_vqa_filtered.json \
    --features_dir vqa/data/GTEx-TCGA-Embeddings \
    --output_dir vqa/outputs/slidechat_stage2_7B_lora \
    --batch_size 23 \
    --gradient_accumulation_steps 8 \
    --num_epochs 10 \
    --lr 2e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --visual_dim 512 \
    --moe_num_slots 32
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import swanlab
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqa.src.vqa_model import MoE_Qwen_VQA
from vqa.src.slidechat_dataset import SlideChatDataset


def setup_ddp():
    """Initialize DDP environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_ddp():
    """Clean up DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description='Train SlideChat Stage 2 (LoRA Finetuning)')

    # Model arguments
    parser.add_argument('--moe_checkpoint', type=str, required=True,
                       help='Path to pretrained MoE_Compressor checkpoint')
    parser.add_argument('--llm_path', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Path to Qwen LLM')
    parser.add_argument('--projector_checkpoint', type=str, required=True,
                       help='Path to Stage 1 trained projector checkpoint (projector.pt)')
    parser.add_argument('--visual_dim', type=int, default=1024,
                       help='Dimension of visual features (1024 for UNI, 512 for ConCH)')
    parser.add_argument('--moe_num_slots', type=int, default=32,
                       help='Number of MoE slots')
    
    # LoRA arguments
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA r')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')

    # Training arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data JSON')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing feature files')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate (suggest 2e-4 for LoRA)')
    parser.add_argument('--warmup_ratio', type=float, default=0.03,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save every N steps')
    parser.add_argument('--log_steps', type=int, default=1,
                       help='Log every N steps')

    # Dataset arguments
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')

    # Swanlab
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable swanlab logging')

    return parser.parse_args()


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    epoch,
    args,
    rank,
    world_size
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader

    optimizer.zero_grad()

    for step, batch in enumerate(pbar):
        # Move to GPU
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        labels = batch['labels'].cuda()
        patch_features = batch['patch_features'].cuda()

        # Forward
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            patch_features=patch_features
        )

        # Scale loss
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        total_loss += loss.item() * args.gradient_accumulation_steps
        num_batches += 1

        # Update weights
        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if rank == 0 and ((step + 1) // args.gradient_accumulation_steps) % args.log_steps == 0:
                avg_loss = total_loss / num_batches
                lr = scheduler.get_last_lr()[0]
                
                if rank == 0:
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})

                if not args.no_wandb:
                    swanlab.log({
                        'train/loss': avg_loss,
                        'train/lr': lr,
                        'epoch': epoch,
                        'step': (step + 1) // args.gradient_accumulation_steps
                    }, step=(step + 1) // args.gradient_accumulation_steps + epoch * (len(dataloader) // args.gradient_accumulation_steps))

        # Save checkpoint
        global_step = (step + 1) // args.gradient_accumulation_steps
        if rank == 0 and global_step > 0 and global_step % args.save_steps == 0 and (step + 1) % args.gradient_accumulation_steps == 0:
            save_path = os.path.join(args.output_dir, f'epoch{epoch}_step{global_step}')
            print(f"\nSaving checkpoint to {save_path}")
            # Save LoRA adapter and Projector
            model_to_save = model.module if world_size > 1 else model
            model_to_save.llm.save_pretrained(os.path.join(save_path, 'lora_adapter'))
            torch.save(model_to_save.projector.state_dict(), os.path.join(save_path, 'projector.pt'))

    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        if not args.no_wandb:
            swanlab.init(
                project="Slidechat-Stage2",
                name=f"stage2_lora_exp",
                config=vars(args)
            )

    if rank == 0:
        print("="*80)
        print("SlideChat Training - Stage 2 (LoRA)")
        print("="*80)

    # Initialize model
    model = MoE_Qwen_VQA(
        moe_checkpoint=args.moe_checkpoint,
        llm_path=args.llm_path,
        num_visual_tokens=16,
        visual_dim=args.visual_dim,
        moe_num_slots=args.moe_num_slots,
        device=f'cuda:{local_rank}'
    )

    # Load Stage 1 Projector
    if rank == 0:
        print(f"Loading Stage 1 Projector from {args.projector_checkpoint}...")
    
    projector_state_dict = torch.load(args.projector_checkpoint, map_location='cpu')
    model.projector.load_state_dict(projector_state_dict)

    # Load Stage 1 finetuned MoE if available (from same directory as projector)
    stage1_dir = os.path.dirname(args.projector_checkpoint)
    moe_checkpoint_path = os.path.join(stage1_dir, 'moe_compressor.pt')
    if os.path.exists(moe_checkpoint_path):
        if rank == 0:
            print(f"Loading Stage 1 finetuned MoE from {moe_checkpoint_path}...")
        moe_state_dict = torch.load(moe_checkpoint_path, map_location='cpu')
        model.visual_encoder.load_state_dict(moe_state_dict)

    # Enable LoRA
    if rank == 0:
        print(f"Enabling LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    model.enable_lora(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)

    # Move to GPU
    model = model.to(f'cuda:{local_rank}')

    # Wrap with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False 
        )

    # Dataset & Dataloader
    dataset = SlideChatDataset(
        data_path=args.data_path,
        features_dir=args.features_dir,
        tokenizer=model.module.tokenizer if world_size > 1 else model.tokenizer,
        mode='vqa', # Stage 2 usually implies VQA task format
        max_length=args.max_length,
        visual_dim=args.visual_dim
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=SlideChatDataset.collate_fn,
        num_workers=0, # Avoid shm issues
        pin_memory=True
    )

    # Optimizer (Trainable params: LoRA + Projector)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if rank == 0:
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(dataloader)/args.gradient_accumulation_steps * args.num_epochs * args.warmup_ratio),
        num_training_steps=len(dataloader)//args.gradient_accumulation_steps * args.num_epochs
    )

    # Training loop
    for epoch in range(args.num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, epoch, args, rank, world_size)
        
        if rank == 0:
            print(f"Epoch {epoch} finished. Loss: {avg_loss:.4f}")
            
            # Save epoch checkpoint
            save_path = os.path.join(args.output_dir, f'epoch{epoch}')
            os.makedirs(save_path, exist_ok=True)
            model_to_save = model.module if world_size > 1 else model
            model_to_save.llm.save_pretrained(os.path.join(save_path, 'lora_adapter'))
            torch.save(model_to_save.projector.state_dict(), os.path.join(save_path, 'projector.pt'))

    cleanup_ddp()

if __name__ == '__main__':
    main()
