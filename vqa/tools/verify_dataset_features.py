#!/usr/bin/env python3
"""
验证SlideChat训练数据集中的所有样本是否都有对应的特征文件
"""

import json
import os
from pathlib import Path
from collections import defaultdict

def verify_dataset(json_path, features_base_dir, dataset_name):
    """
    验证数据集中的特征文件是否存在

    Args:
        json_path: 数据集JSON文件路径
        features_base_dir: 特征文件基础目录
        dataset_name: 数据集名称（用于输出）
    """
    print(f"\n{'='*80}")
    print(f"验证数据集: {dataset_name}")
    print(f"数据集路径: {json_path}")
    print(f"特征目录: {features_base_dir}")
    print(f"{'='*80}\n")

    # 读取数据集
    with open(json_path, 'r') as f:
        data = json.load(f)

    total_samples = len(data)
    missing_features = []
    found_features = []
    cancer_type_stats = defaultdict(lambda: {'total': 0, 'missing': 0, 'found': 0})

    print(f"总样本数: {total_samples}\n")
    print("正在检查特征文件...")

    for idx, sample in enumerate(data):
        sample_id = sample.get('id', f'sample_{idx}')
        image_paths = sample.get('image', [])

        if not image_paths:
            print(f"⚠️  样本 {sample_id}: 没有image字段")
            continue

        # SlideChat格式中image是相对路径，如 "./BLCA/TCGA-GV-A40G-01Z-00-DX1.csv"
        for img_path in image_paths:
            # 提取癌症类型和slide_id
            parts = img_path.strip('./').split('/')
            if len(parts) >= 2:
                cancer_type = parts[0]  # 如 "BLCA"
                csv_filename = parts[1]  # 如 "TCGA-GV-A40G-01Z-00-DX1.csv"
                slide_id = csv_filename.replace('.csv', '')  # 如 "TCGA-GV-A40G-01Z-00-DX1"
            else:
                print(f"⚠️  样本 {sample_id}: 无法解析路径 {img_path}")
                continue

            cancer_type_stats[cancer_type]['total'] += 1

            # 构建可能的特征文件路径
            # 根据GTEx-TCGA-Embeddings目录结构，路径应该是:
            # TCGA-{CANCER}/feat/{SLIDE_ID}.pt 或 TCGA-{CANCER}/feat/{SLIDE_ID}.npy

            # 映射癌症类型到TCGA目录
            tcga_mapping = {
                'BLCA': 'TCGA-BLCA',
                'BRCA': 'TCGA-BR',  # 注意是BR不是BRCA
                'COAD': 'TCGA-COAD',
                'READ': 'TCGA-COAD',  # READ通常与COAD合并
                'GBM': 'TCGA-GBM',
                'HNSC': 'TCGA-HNSC',
                'LGG': 'TCGA-LGG',
                'LUAD': 'TCGA-LUNG',
                'LUSC': 'TCGA-LUNG',
                'SKCM': 'TCGA-SKCM',
            }

            tcga_dir = tcga_mapping.get(cancer_type, f'TCGA-{cancer_type}')

            # 构建可能的搜索目录（包括双层嵌套结构）
            possible_dirs = [
                os.path.join(features_base_dir, tcga_dir, tcga_dir),  # 双层嵌套
                os.path.join(features_base_dir, tcga_dir, 'feat'),    # 标准结构
                os.path.join(features_base_dir, tcga_dir),             # 直接路径
                os.path.join(features_base_dir, 'TCGA-Rest', 'TCGA-Rest'),  # Rest双层嵌套
                os.path.join(features_base_dir, 'TCGA-Rest', 'feat'),
                os.path.join(features_base_dir, 'TCGA-Rest'),
            ]

            feature_found = False
            found_path = None
            possible_paths = []

            # 在每个可能的目录中搜索匹配的文件
            for search_dir in possible_dirs:
                if not os.path.exists(search_dir):
                    continue

                try:
                    files = os.listdir(search_dir)
                    # 查找以slide_id开头的文件
                    matching_files = [f for f in files if f.startswith(slide_id)]

                    if matching_files:
                        # 优先选择特定模式的文件
                        for priority_pattern in ['_0_1024.npy', '_1_1024.npy', '_0_512.npy', '_1_512.npy', '.pt', '.pth']:
                            for fname in matching_files:
                                if priority_pattern in fname:
                                    found_path = os.path.join(search_dir, fname)
                                    feature_found = True
                                    break
                            if feature_found:
                                break

                        # 如果没有优先匹配，使用第一个匹配的文件
                        if not feature_found and matching_files:
                            found_path = os.path.join(search_dir, matching_files[0])
                            feature_found = True

                    if feature_found:
                        break
                except (OSError, PermissionError):
                    continue

            # 记录搜索过的路径（用于调试）
            possible_paths = [
                os.path.join(features_base_dir, tcga_dir, tcga_dir, f'{slide_id}*.npy'),
                os.path.join(features_base_dir, tcga_dir, 'feat', f'{slide_id}*.npy'),
                os.path.join(features_base_dir, 'TCGA-Rest', 'TCGA-Rest', f'{slide_id}*.npy'),
            ]

            if feature_found:
                found_features.append({
                    'sample_id': sample_id,
                    'cancer_type': cancer_type,
                    'slide_id': slide_id,
                    'feature_path': found_path
                })
                cancer_type_stats[cancer_type]['found'] += 1
            else:
                missing_features.append({
                    'sample_id': sample_id,
                    'cancer_type': cancer_type,
                    'slide_id': slide_id,
                    'searched_paths': possible_paths[:3]  # 只显示前3个主要路径
                })
                cancer_type_stats[cancer_type]['missing'] += 1

    # 输出统计结果
    print(f"\n{'='*80}")
    print("验证结果汇总")
    print(f"{'='*80}")
    print(f"总样本数: {total_samples}")
    print(f"✅ 找到特征文件: {len(found_features)} ({len(found_features)/total_samples*100:.2f}%)")
    print(f"❌ 缺失特征文件: {len(missing_features)} ({len(missing_features)/total_samples*100:.2f}%)")

    # 按癌症类型统计
    print(f"\n{'='*80}")
    print("按癌症类型统计")
    print(f"{'='*80}")
    print(f"{'类型':<10} {'总数':>8} {'找到':>8} {'缺失':>8} {'完整率':>10}")
    print(f"{'-'*80}")

    for cancer_type in sorted(cancer_type_stats.keys()):
        stats = cancer_type_stats[cancer_type]
        completion_rate = stats['found'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"{cancer_type:<10} {stats['total']:>8} {stats['found']:>8} {stats['missing']:>8} {completion_rate:>9.2f}%")

    # 显示缺失的详细信息（最多显示20个）
    if missing_features:
        print(f"\n{'='*80}")
        print(f"缺失特征文件详情（显示前20个）")
        print(f"{'='*80}")
        for i, missing in enumerate(missing_features[:20]):
            print(f"\n{i+1}. 样本ID: {missing['sample_id']}")
            print(f"   癌症类型: {missing['cancer_type']}")
            print(f"   Slide ID: {missing['slide_id']}")
            print(f"   查找路径:")
            for path in missing['searched_paths']:
                print(f"     - {path}")

        if len(missing_features) > 20:
            print(f"\n... 还有 {len(missing_features) - 20} 个缺失的特征文件未显示")

    return {
        'total': total_samples,
        'found': len(found_features),
        'missing': len(missing_features),
        'cancer_type_stats': dict(cancer_type_stats),
        'missing_list': missing_features
    }


def main():
    # 设置路径
    features_base_dir = '/workspace/ETC/vqa/data/GTEx-TCGA-Embeddings'
    stage1_json = '/workspace/ETC/vqa/data/SlideChat/SlideInstruct_train_stage1_caption.json'
    stage2_json = '/workspace/ETC/vqa/data/SlideChat/SlideInstruct_train_stage2_vqa.json'

    # 验证Stage 1数据集
    stage1_results = verify_dataset(
        stage1_json,
        features_base_dir,
        'Stage 1 - Caption Pretraining'
    )

    # 验证Stage 2数据集
    stage2_results = verify_dataset(
        stage2_json,
        features_base_dir,
        'Stage 2 - VQA Finetuning'
    )

    # 总结
    print(f"\n{'='*80}")
    print("总体验证结果")
    print(f"{'='*80}")
    print(f"\nStage 1 (Caption):")
    print(f"  总样本: {stage1_results['total']}")
    print(f"  ✅ 找到: {stage1_results['found']} ({stage1_results['found']/stage1_results['total']*100:.2f}%)")
    print(f"  ❌ 缺失: {stage1_results['missing']} ({stage1_results['missing']/stage1_results['total']*100:.2f}%)")

    print(f"\nStage 2 (VQA):")
    print(f"  总样本: {stage2_results['total']}")
    print(f"  ✅ 找到: {stage2_results['found']} ({stage2_results['found']/stage2_results['total']*100:.2f}%)")
    print(f"  ❌ 缺失: {stage2_results['missing']} ({stage2_results['missing']/stage2_results['total']*100:.2f}%)")

    # 检查特征目录结构
    print(f"\n{'='*80}")
    print("特征目录结构检查")
    print(f"{'='*80}")

    if os.path.exists(features_base_dir):
        print(f"✅ 特征基础目录存在: {features_base_dir}\n")

        # 列出所有TCGA目录
        tcga_dirs = [d for d in os.listdir(features_base_dir)
                     if os.path.isdir(os.path.join(features_base_dir, d)) and d.startswith('TCGA-')]

        print("找到的TCGA目录:")
        for tcga_dir in sorted(tcga_dirs):
            feat_dir = os.path.join(features_base_dir, tcga_dir, 'feat')
            if os.path.exists(feat_dir):
                num_features = len([f for f in os.listdir(feat_dir)
                                   if f.endswith(('.pt', '.pth', '.npy'))])
                print(f"  ✅ {tcga_dir}/feat/ - {num_features} 个特征文件")
            else:
                print(f"  ⚠️  {tcga_dir}/ - 缺少feat子目录")
    else:
        print(f"❌ 特征基础目录不存在: {features_base_dir}")

    print(f"\n{'='*80}")

    # 返回状态码
    if stage1_results['missing'] == 0 and stage2_results['missing'] == 0:
        print("✅ 所有数据集的特征文件都已正确对应！")
        return 0
    else:
        print("⚠️  存在缺失的特征文件，请检查上述详情。")
        return 1


if __name__ == '__main__':
    exit(main())
