"""
Training script for WSI-VQA with DDP support.

Two-stage training strategy:
    Stage 1: Freeze LLM, train Projector only (caption pretraining)
    Stage 2: Unfreeze LLM, full finetuning (VQA finetuning)

Usage:
    # Stage 1 (Single GPU)
    python tools/train_vqa.py \
        --stage 1 \
        --moe_checkpoint checkpoints/moe_best.pt \
        --llm_path Qwen/Qwen3-4B-Instruct-2507 \
        --data_path data/captions_train.csv \
        --output_dir outputs/stage1 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --num_epochs 3

    # Stage 1 (Multi-GPU DDP)
    torchrun --nproc_per_node=2 tools/train_vqa.py \
        --stage 1 \
        --moe_checkpoint checkpoints/moe_best.pt \
        --llm_path Qwen/Qwen3-4B-Instruct-2507 \
        --data_path data/captions_train.csv \
        --output_dir outputs/stage1 \
        --batch_size 4 \
        --gradient_accumulation_steps 4 \
        --num_epochs 3

    # Stage 2 (Full Finetuning)
    torchrun --nproc_per_node=2 tools/train_vqa.py \
        --stage 2 \
        --moe_checkpoint checkpoints/moe_best.pt \
        --llm_path Qwen/Qwen3-4B-Instruct-2507 \
        --load_projector outputs/stage1/final/projector.pt \
        --data_path data/vqa_train.json \
        --output_dir outputs/stage2 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --num_epochs 5 \
        --lr 1e-5
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
import wandb

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vqa.src.vqa_model import MoE_Qwen_VQA
from vqa.src.vqa_dataset import SlideChatDataset, collate_fn


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
    parser = argparse.ArgumentParser(description='Train WSI-VQA Model')

    # Model arguments
    parser.add_argument('--moe_checkpoint', type=str, required=True,
                       help='Path to pretrained MoE_Compressor checkpoint')
    parser.add_argument('--llm_path', type=str, default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Path to Qwen LLM')
    parser.add_argument('--load_projector', type=str, default=None,
                       help='Path to pretrained projector (for Stage 2)')
    parser.add_argument('--num_visual_tokens', type=int, default=16,
                       help='Number of visual tokens from MoE')

    # Training arguments
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2],
                       help='Training stage: 1 (freeze LLM) or 2 (finetune LLM)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to training data (CSV for stage 1, JSON for stage 2)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (1e-4 for stage 1, 1e-5 for stage 2)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup ratio')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Save checkpoint every N steps')
    parser.add_argument('--log_steps', type=int, default=10,
                       help='Log metrics every N steps')

    # Dataset arguments
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')

    # Wandb
    parser.add_argument('--wandb_project', type=str, default='wsi-vqa',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable wandb logging')

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
    """
    Train for one epoch.

    Args:
        model: DDP-wrapped model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch number
        args: Training arguments
        rank: DDP rank
        world_size: Number of GPUs
    """
    model.train()

    total_loss = 0
    num_batches = 0

    # Progress bar only on rank 0
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

        # Forward pass
        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            patch_features=patch_features
        )

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps

        # Backward
        loss.backward()

        # Accumulate loss
        total_loss += loss.item() * args.gradient_accumulation_steps
        num_batches += 1

        # Update weights
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm
            )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Logging
            if rank == 0 and (step + 1) % args.log_steps == 0:
                avg_loss = total_loss / num_batches
                lr = scheduler.get_last_lr()[0]

                if rank == 0:
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{lr:.2e}'
                    })

                if not args.no_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/lr': lr,
                        'epoch': epoch,
                        'step': step
                    })

        # Save checkpoint
        if rank == 0 and (step + 1) % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f'epoch{epoch}_step{step+1}')
            print(f"\nSaving checkpoint to {save_path}")
            model.module.save_pretrained(save_path)

    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()

    # Setup DDP
    rank, world_size, local_rank = setup_ddp()

    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize wandb (rank 0 only)
    if rank == 0 and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f'stage{args.stage}_vqa',
            config=vars(args)
        )

    # Print configuration
    if rank == 0:
        print("=" * 80)
        print(f"Training Configuration - Stage {args.stage}")
        print("=" * 80)
        print(f"MoE Checkpoint: {args.moe_checkpoint}")
        print(f"LLM Path: {args.llm_path}")
        print(f"Data Path: {args.data_path}")
        print(f"Output Dir: {args.output_dir}")
        print(f"Batch Size: {args.batch_size} x {world_size} GPUs x {args.gradient_accumulation_steps} accum")
        print(f"Effective Batch Size: {args.batch_size * world_size * args.gradient_accumulation_steps}")
        print(f"Learning Rate: {args.lr}")
        print(f"Num Epochs: {args.num_epochs}")
        print("=" * 80)

    # Initialize model
    if rank == 0:
        print("\nInitializing model...")

    model = MoE_Qwen_VQA(
        moe_checkpoint=args.moe_checkpoint,
        llm_path=args.llm_path,
        num_visual_tokens=args.num_visual_tokens,
        device=f'cuda:{local_rank}'
    )

    # Load pretrained projector if specified
    if args.load_projector is not None:
        if rank == 0:
            print(f"Loading pretrained projector from {args.load_projector}")
        model.projector.load_state_dict(torch.load(args.load_projector))

    # Set training mode based on stage
    if args.stage == 1:
        # Stage 1: Freeze LLM, train Projector only
        model.freeze_llm()
        if rank == 0:
            print("Stage 1: Training Projector only (LLM frozen)")
    else:
        # Stage 2: Full finetuning
        model.unfreeze_llm()
        if rank == 0:
            print("Stage 2: Full finetuning (LLM + Projector)")

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

    # Create dataset
    mode = 'caption' if args.stage == 1 else 'vqa'

    dataset = SlideChatDataset(
        data_path=args.data_path,
        tokenizer=model.module.tokenizer if world_size > 1 else model.tokenizer,
        mode=mode,
        max_length=args.max_length,
        num_visual_tokens=args.num_visual_tokens
    )

    # Create dataloader with DDP sampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Setup optimizer
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Setup scheduler
    num_training_steps = len(dataloader) // args.gradient_accumulation_steps * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    if rank == 0:
        print(f"\nTraining steps: {num_training_steps}")
        print(f"Warmup steps: {num_warmup_steps}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        print("\nStarting training...\n")

    # Training loop
    for epoch in range(args.num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)

        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            rank=rank,
            world_size=world_size
        )

        if rank == 0:
            print(f"\nEpoch {epoch} - Average Loss: {avg_loss:.4f}")

            if not args.no_wandb:
                wandb.log({
                    'epoch/loss': avg_loss,
                    'epoch': epoch
                })

    # Save final checkpoint
    if rank == 0:
        final_path = os.path.join(args.output_dir, 'final')
        print(f"\nSaving final checkpoint to {final_path}")
        if world_size > 1:
            model.module.save_pretrained(final_path)
        else:
            model.save_pretrained(final_path)

    # Cleanup
    if not args.no_wandb and rank == 0:
        wandb.finish()

    cleanup_ddp()

    if rank == 0:
        print("\nTraining completed!")


if __name__ == '__main__':
    main()
