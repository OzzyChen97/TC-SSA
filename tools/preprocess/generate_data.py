"""
Script to split the TCGA BRCA dataset into train/val/test sets with 8:1:1 ratio.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import random
import sys

# --- 核心工具函數 ---

def extract_patient_id(slide_id):
    # 提取病人ID，確保同一病人的切片不分家
    parts = slide_id.split('-')

    # CPTAC格式: C3L-00001-21 -> C3L-00001
    if len(parts) >= 2 and parts[0] in ['C3L', 'C3N']:
        return '-'.join(parts[:2]).upper()

    # TCGA格式: TCGA-3C-AALI-01Z-00-DX1 -> TCGA-3C-AALI
    if len(parts) >= 3 and parts[0] == 'TCGA':
        return '-'.join(parts[:3]).upper()

    # 默认：取前12个字符
    return slide_id[:12].upper()

def get_stratified_split_indices(labels, test_ratio, seed):
    # 分層採樣：保證訓練集和測試集中 IDC/ILC 的比例一致
    np.random.seed(seed)
    classes = np.unique(labels)
    test_indices = []
    train_indices = []
    
    for c in classes:
        cls_indices = np.where(labels == c)[0]
        np.random.shuffle(cls_indices)
        n_test = int(len(cls_indices) * test_ratio)
        
        # 處理邊界情況：如果某類樣本太少，優先保證訓練集有數據
        if n_test > 0:
            test_indices.extend(cls_indices[:n_test])
            train_indices.extend(cls_indices[n_test:])
        else:
            train_indices.extend(cls_indices)
            
    return np.array(train_indices), np.array(test_indices)

def split_by_patient(input_csv, output_dir='data', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, stratify=True):
    """
    核心拆分邏輯：
    1. 提取 Patient ID
    2. 按 Patient ID 進行 8:1:1 劃分
    3. 映射回 Slide ID
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Loading dataset from: {input_csv}")
    if not Path(input_csv).exists():
        print(f"Error: Input file {input_csv} not found!")
        sys.exit(1)
        
    df = pd.read_csv(input_csv)
    
    # 1. 提取病人 ID
    df['patient_id'] = df['slide_id'].apply(extract_patient_id)
    
    # 2. 按病人聚合標籤 (確保同一病人多張切片標籤一致)
    patient_labels_series = df.groupby('patient_id')['label'].agg(lambda x: x.mode()[0])
    unique_patients = np.array(patient_labels_series.index.tolist())
    labels = patient_labels_series.values
    
    print(f"Total slides: {len(df)}")
    print(f"Unique patients: {len(unique_patients)}")
    print(f"Split Ratio -> Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

    # 3. 第一步拆分：(Train + Val) vs Test
    # Test 取總量的 0.1
    if stratify:
        train_val_idx, test_idx = get_stratified_split_indices(labels, test_ratio, seed)
    else:
        indices = np.arange(len(unique_patients))
        np.random.shuffle(indices)
        n_test = int(len(unique_patients) * test_ratio)
        test_idx = indices[:n_test]
        train_val_idx = indices[n_test:]
        
    train_val_patients = unique_patients[train_val_idx]
    test_patients = unique_patients[test_idx]
    train_val_labels = labels[train_val_idx]
    
    # 4. 第二步拆分：Train vs Val
    # 剩下的數據是總量的 0.9
    # Val 需要佔總量的 0.1，所以在剩下的數據中佔比為 0.1 / 0.9 = 1/9
    val_ratio_adjusted = val_ratio / (1.0 - test_ratio)
    
    if stratify:
        train_idx, val_idx = get_stratified_split_indices(train_val_labels, val_ratio_adjusted, seed)
    else:
        indices = np.arange(len(train_val_patients))
        np.random.shuffle(indices)
        n_val = int(len(indices) * val_ratio_adjusted)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
    train_patients = train_val_patients[train_idx]
    val_patients = train_val_patients[val_idx]
    
    # 5. 驗證無重疊 (Double Check)
    train_set, val_set, test_set = set(train_patients), set(val_patients), set(test_patients)
    assert len(train_set & val_set) == 0, "Train/Val Overlap!"
    assert len(train_set & test_set) == 0, "Train/Test Overlap!"
    assert len(val_set & test_set) == 0, "Val/Test Overlap!"
    
    # 6. 生成結果 DataFrame
    train_df = df[df['patient_id'].isin(train_set)][['slide_id', 'label']]
    val_df = df[df['patient_id'].isin(val_set)][['slide_id', 'label']]
    test_df = df[df['patient_id'].isin(test_set)][['slide_id', 'label']]
    
    # 7. 保存
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)
    
    print("\n✅ Splitting Complete (8:1:1 Patient-Level).")
    print(f"Train: {len(train_df)} slides ({len(train_patients)} patients)")
    print(f"Val:   {len(val_df)} slides ({len(val_patients)} patients)")
    print(f"Test:  {len(test_df)} slides ({len(test_patients)} patients)")

def verify_no_leakage(output_dir):
    output_path = Path(output_dir)
    train = pd.read_csv(output_path / 'train.csv')
    val = pd.read_csv(output_path / 'val.csv')
    test = pd.read_csv(output_path / 'test.csv')
    
    train_p = set(train['slide_id'].apply(extract_patient_id))
    val_p = set(val['slide_id'].apply(extract_patient_id))
    test_p = set(test['slide_id'].apply(extract_patient_id))
    
    overlap = (train_p & test_p) | (train_p & val_p) | (val_p & test_p)
    if len(overlap) == 0:
        print("\n🎉 Verification Passed: No Patient Leakage!")
    else:
        print(f"\n❌ LEAKAGE DETECTED: {overlap}")

def main():
    parser = argparse.ArgumentParser(description="Split Dataset by Patient ID (8:1:1)")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to full dataset CSV')
    parser.add_argument('--output_dir', type=str, default='dataset_splits', help='Directory to save splits')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 執行拆分 (默認 8:1:1)
    split_by_patient(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )
    
    verify_no_leakage(args.output_dir)

if __name__ == '__main__':
    main()