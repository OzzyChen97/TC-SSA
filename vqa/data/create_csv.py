
import os
import pandas as pd
from pathlib import Path

def create_dataset_csv(root_dir, output_dir):
    root_path = Path(root_dir)
    classes = sorted([d.name for d in root_path.iterdir() if d.is_dir() and not d.name.startswith('.')])
    
    print(f"Found classes: {classes}")
    
    data = []
    
    label_map = {cls_name: i for i, cls_name in enumerate(classes)}
    print(f"Label mapping: {label_map}")
    
    for cls_name in classes:
        cls_dir = root_path / cls_name
        # Check if there is a nested directory with the same name
        nested_dir = cls_dir / cls_name
        if nested_dir.exists() and nested_dir.is_dir():
            search_dir = nested_dir
        else:
            search_dir = cls_dir
            
        print(f"Searching in {search_dir} for class {cls_name}")
        
        # files = list(search_dir.glob("*.npy"))
        # Using rglob just in case, but keeping it simple first
        files = list(search_dir.glob("*.npy"))
        
        print(f"Found {len(files)} files for class {cls_name}")
        
        for file_path in files:
            # We use the stem as slide_id, but the file naming seems complex.
            # Example: TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291_0_1024.npy
            # We should probably store the full path or relative path, or just the stem if the dataset class expects it.
            # train.py uses WSIFeatureDataset which loads {features_dir}/{slide_id}.pt
            # Our data is .npy and scattered.
            # So standard WSIFeatureDataset won't work directly because it expects a flat directory.
            # We should write a train_moe.py that can handle the specific CSV which includes full paths.
            
            data.append({
                'slide_id': file_path.stem,
                'file_path': str(file_path),
                'label': label_map[cls_name],
                'label_name': cls_name
            })
            
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split 80/20
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    train_csv_path = os.path.join(output_dir, 'train.csv')
    val_csv_path = os.path.join(output_dir, 'val.csv')
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"Saved train.csv with {len(train_df)} samples to {train_csv_path}")
    print(f"Saved val.csv with {len(val_df)} samples to {val_csv_path}")
    
    # Also save label map
    import json
    with open(os.path.join(output_dir, 'label_map.json'), 'w') as f:
        json.dump(label_map, f, indent=2)

if __name__ == "__main__":
    root = "/workspace/ETC/vqa/data/GTEx-TCGA-Embeddings"
    output = "/workspace/ETC/vqa/data"
    create_dataset_csv(root, output)
