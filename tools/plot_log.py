
import re
import matplotlib.pyplot as plt
import argparse
import os

def parse_log(log_path):
    train_epoch_stats = {'epoch': [], 'loss': [], 'ce': [], 'aux': [], 'acc': [], 'auc': []}
    val_epoch_stats = {'epoch': [], 'loss': [], 'ce': [], 'aux': [], 'acc': [], 'auc': []}
    train_batch_stats = {'epoch': [], 'batch': [], 'total_steps': [], 'loss': [], 'ce': [], 'aux': []}

    total_steps = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse Training Epoch Stats
            # Epoch [1] Training - Loss: 0.4460 CE: 0.4445 Aux: 0.1543 Acc: 0.7469 AUC: 0.7710 Time: 58.42s
            match_train_epoch = re.search(r'Epoch \[(\d+)\] Training - Loss: ([\d\.]+) CE: ([\d\.]+) Aux: ([\d\.]+) Acc: ([\d\.]+) AUC: ([\d\.]+)', line)
            if match_train_epoch:
                train_epoch_stats['epoch'].append(int(match_train_epoch.group(1)))
                train_epoch_stats['loss'].append(float(match_train_epoch.group(2)))
                train_epoch_stats['ce'].append(float(match_train_epoch.group(3)))
                train_epoch_stats['aux'].append(float(match_train_epoch.group(4)))
                train_epoch_stats['acc'].append(float(match_train_epoch.group(5)))
                train_epoch_stats['auc'].append(float(match_train_epoch.group(6)))
                continue

            # Parse Validation Epoch Stats
            # Epoch [1] Validation - Loss: 0.3652 CE: 0.3652 Aux: 0.0000 Acc: 0.8020 AUC: 0.9580
            match_val_epoch = re.search(r'Epoch \[(\d+)\] Validation - Loss: ([\d\.]+) CE: ([\d\.]+) Aux: ([\d\.]+) Acc: ([\d\.]+) AUC: ([\d\.]+)', line)
            if match_val_epoch:
                val_epoch_stats['epoch'].append(int(match_val_epoch.group(1)))
                val_epoch_stats['loss'].append(float(match_val_epoch.group(2)))
                val_epoch_stats['ce'].append(float(match_val_epoch.group(3)))
                val_epoch_stats['aux'].append(float(match_val_epoch.group(4)))
                val_epoch_stats['acc'].append(float(match_val_epoch.group(5)))
                val_epoch_stats['auc'].append(float(match_val_epoch.group(6)))
                continue

            # Parse Training Batch Stats
            # Epoch [1] Batch [10/818] Loss: 0.9632 CE: 0.9619 Aux: 0.1291
            match_batch = re.search(r'Epoch \[(\d+)\] Batch \[(\d+)/(\d+)\] Loss: ([\d\.]+) CE: ([\d\.]+) Aux: ([\d\.]+)', line)
            if match_batch:
                epoch = int(match_batch.group(1))
                batch = int(match_batch.group(2))
                train_batch_stats['epoch'].append(epoch)
                train_batch_stats['batch'].append(batch)
                train_batch_stats['loss'].append(float(match_batch.group(4)))
                train_batch_stats['ce'].append(float(match_batch.group(5)))
                train_batch_stats['aux'].append(float(match_batch.group(6)))
                
                # Calculate absolute step number (approximate if we don't know total batches per epoch previous to this)
                # But here we do know batch count from the log line itself: 818
                # So step = (epoch - 1) * 818 + batch
                batch_total = int(match_batch.group(3))
                step = (epoch - 1) * batch_total + batch
                train_batch_stats['total_steps'].append(step)

    return train_epoch_stats, val_epoch_stats, train_batch_stats

def plot_metrics(train_epoch, val_epoch, train_batch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss (Epoch)
    plt.figure(figsize=(10, 6))
    plt.plot(train_epoch['epoch'], train_epoch['loss'], label='Train Loss', marker='o')
    plt.plot(val_epoch['epoch'], val_epoch['loss'], label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_epoch.png'))
    plt.close()

    # 2. Accuracy (Epoch)
    plt.figure(figsize=(10, 6))
    plt.plot(train_epoch['epoch'], train_epoch['acc'], label='Train Acc', marker='o')
    plt.plot(val_epoch['epoch'], val_epoch['acc'], label='Val Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_epoch.png'))
    plt.close()
    
    # 3. AUC (Epoch)
    plt.figure(figsize=(10, 6))
    plt.plot(train_epoch['epoch'], train_epoch['auc'], label='Train AUC', marker='o')
    plt.plot(val_epoch['epoch'], val_epoch['auc'], label='Val AUC', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'auc_epoch.png'))
    plt.close()

    # 4. Batch Loss
    plt.figure(figsize=(12, 6))
    plt.plot(train_batch['total_steps'], train_batch['loss'], label='Batch Loss', alpha=0.3)
    # Moving average
    window_size = 50
    if len(train_batch['loss']) > window_size:
        moving_avg = [sum(train_batch['loss'][i:i+window_size])/window_size for i in range(len(train_batch['loss'])-window_size)]
        plt.plot(train_batch['total_steps'][window_size:], moving_avg, label=f'Moving Avg (win={window_size})', color='red')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Batch Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_batch.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot training logs')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--output_dir', type=str, default='plots', help='Directory to save plots')
    args = parser.parse_args()

    train_epoch, val_epoch, train_batch = parse_log(args.log_file)
    
    print(f"Parsed {len(train_epoch['epoch'])} training epochs")
    print(f"Parsed {len(val_epoch['epoch'])} validation epochs")
    print(f"Parsed {len(train_batch['total_steps'])} batches")
    
    plot_metrics(train_epoch, val_epoch, train_batch, args.output_dir)
    print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
