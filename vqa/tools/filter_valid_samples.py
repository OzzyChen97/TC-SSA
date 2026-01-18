#!/usr/bin/env python3
"""
过滤训练数据集，只保留有有效特征文件的样本
"""

import json
import os
import sys
from pathlib import Path

def find_feature_file(slide_id, cancer_type, features_base_dir):
    """
    查找特征文件

    Args:
        slide_id: Slide ID
        cancer_type: 癌症类型
        features_base_dir: 特征文件基础目录

    Returns:
        特征文件路径或None
    """
    # 映射癌症类型到TCGA目录
    tcga_mapping = {
        'BLCA': 'TCGA-BLCA',
        'BRCA': 'TCGA-BR',
        'COAD': 'TCGA-COAD',
        'READ': 'TCGA-COAD',
        'GBM': 'TCGA-GBM',
        'HNSC': 'TCGA-HNSC',
        'LGG': 'TCGA-LGG',
        'LUAD': 'TCGA-LUNG',
        'LUSC': 'TCGA-LUNG',
        'SKCM': 'TCGA-SKCM',
    }

    tcga_dir = tcga_mapping.get(cancer_type, f'TCGA-{cancer_type}')

    # 构建可能的搜索目录
    possible_dirs = [
        os.path.join(features_base_dir, tcga_dir, tcga_dir),
        os.path.join(features_base_dir, tcga_dir, 'feat'),
        os.path.join(features_base_dir, tcga_dir),
        os.path.join(features_base_dir, 'TCGA-Rest', 'TCGA-Rest'),
        os.path.join(features_base_dir, 'TCGA-Rest', 'feat'),
        os.path.join(features_base_dir, 'TCGA-Rest'),
    ]

    # 在每个可能的目录中搜索匹配的文件
    for search_dir in possible_dirs:
        if not os.path.exists(search_dir):
            continue

        try:
            files = os.listdir(search_dir)
            matching_files = [f for f in files if f.startswith(slide_id)]

            if matching_files:
                # 优先选择特定模式的文件
                for priority_pattern in ['_0_1024.npy', '_1_1024.npy', '_0_512.npy', '_1_512.npy', '.pt', '.pth']:
                    for fname in matching_files:
                        if priority_pattern in fname:
                            return os.path.join(search_dir, fname)

                # 如果没有优先匹配，使用第一个匹配的文件
                if matching_files:
                    return os.path.join(search_dir, matching_files[0])
        except (OSError, PermissionError):
            continue

    return None


def filter_dataset(input_json, output_json, features_base_dir):
    """
    过滤数据集，只保留有特征文件的样本

    Args:
        input_json: 输入JSON文件路径
        output_json: 输出JSON文件路径
        features_base_dir: 特征文件基础目录
    """
    print(f"\n读取数据集: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)

    total = len(data)
    valid_samples = []
    invalid_samples = []

    print(f"总样本数: {total}")
    print("正在过滤...")

    for idx, sample in enumerate(data):
        if (idx + 1) % 1000 == 0:
            print(f"  处理进度: {idx + 1}/{total} ({(idx+1)/total*100:.1f}%)")

        sample_id = sample.get('id', f'sample_{idx}')
        image_paths = sample.get('image', [])

        if not image_paths:
            invalid_samples.append(sample_id)
            continue

        # 解析路径
        img_path = image_paths[0].lstrip('./')
        parts = img_path.split('/')

        if len(parts) >= 2:
            cancer_type = parts[0]
            csv_filename = parts[1]
            slide_id = csv_filename.replace('.csv', '')
        else:
            invalid_samples.append(sample_id)
            continue

        # 查找特征文件
        feature_path = find_feature_file(slide_id, cancer_type, features_base_dir)

        if feature_path:
            valid_samples.append(sample)
        else:
            invalid_samples.append(sample_id)

    # 保存过滤后的数据集
    with open(output_json, 'w') as f:
        json.dump(valid_samples, f, indent=2, ensure_ascii=False)

    print(f"\n过滤完成！")
    print(f"  原始样本数: {total}")
    print(f"  ✅ 有效样本: {len(valid_samples)} ({len(valid_samples)/total*100:.2f}%)")
    print(f"  ❌ 无效样本: {len(invalid_samples)} ({len(invalid_samples)/total*100:.2f}%)")
    print(f"\n保存到: {output_json}")

    if invalid_samples[:10]:
        print(f"\n无效样本示例 (前10个):")
        for sid in invalid_samples[:10]:
            print(f"  - {sid}")

    return len(valid_samples), len(invalid_samples)


def main():
    features_base_dir = '/workspace/ETC/vqa/data/GTEx-TCGA-Embeddings'

    datasets = [
        {
            'input': '/workspace/ETC/vqa/data/SlideChat/SlideInstruct_train_stage1_caption.json',
            'output': '/workspace/ETC/vqa/data/SlideChat/SlideInstruct_train_stage1_caption_filtered.json',
            'name': 'Stage 1 (Caption)'
        },
        {
            'input': '/workspace/ETC/vqa/data/SlideChat/SlideInstruct_train_stage2_vqa.json',
            'output': '/workspace/ETC/vqa/data/SlideChat/SlideInstruct_train_stage2_vqa_filtered.json',
            'name': 'Stage 2 (VQA)'
        }
    ]

    print("="*80)
    print("过滤SlideChat训练数据集")
    print("="*80)

    total_valid = 0
    total_invalid = 0

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"处理: {dataset['name']}")
        print(f"{'='*80}")

        valid, invalid = filter_dataset(
            dataset['input'],
            dataset['output'],
            features_base_dir
        )

        total_valid += valid
        total_invalid += invalid

    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}")
    print(f"总有效样本: {total_valid}")
    print(f"总无效样本: {total_invalid}")
    print(f"总体完整率: {total_valid/(total_valid+total_invalid)*100:.2f}%")

    print(f"\n{'='*80}")
    print("✅ 过滤完成！现在可以使用过滤后的数据集进行训练：")
    print(f"{'='*80}")
    for dataset in datasets:
        print(f"  {dataset['name']}: {dataset['output']}")


if __name__ == '__main__':
    main()
