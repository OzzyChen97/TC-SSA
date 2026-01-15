"""
按照 8:1:1 比例将 tcga_brca_dataset.csv 分割为训练集、验证集和测试集。
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ================= 配置区域 =================
# 输入文件
INPUT_CSV = '/workspace/ETC/tcga_brca_dataset.csv'

# 输出目录
OUTPUT_DIR = '/workspace/ETC/dataset'

# 分割比例 (train:val:test = 8:1:1)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 随机种子（保证可重复性）
RANDOM_SEED = 42
# ===========================================


def split_dataset():
    """按照指定比例分割数据集，并保持类别平衡"""

    print("=" * 50)
    print("TCGA BRCA 数据集分割脚本")
    print("=" * 50)
    print(f"输入文件: {INPUT_CSV}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"分割比例: Train={TRAIN_RATIO:.1%}, Val={VAL_RATIO:.1%}, Test={TEST_RATIO:.1%}")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 50)
    print()

    # 1. 读取数据集
    print("正在读取数据集...")
    df = pd.read_csv(INPUT_CSV)
    print(f"✓ 加载完成: {len(df)} 个样本")

    # 2. 查看类别分布
    print("\n原始数据集类别分布:")
    class_counts = df['label'].value_counts().sort_index()
    for label, count in class_counts.items():
        label_name = "IDC (浸润性导管癌)" if label == 0 else "ILC (浸润性小叶癌)"
        print(f"  Class {label} ({label_name}): {count} 样本 ({count/len(df)*100:.1f}%)")
    print()

    # 3. 设置随机种子
    np.random.seed(RANDOM_SEED)

    # 4. 手动实现分层抽样分割
    print("正在进行数据集分割（保持类别比例）...")

    train_dfs = []
    val_dfs = []
    test_dfs = []

    # 对每个类别分别进行分割
    for label in df['label'].unique():
        # 获取该类别的所有样本
        class_df = df[df['label'] == label].copy()
        n_samples = len(class_df)

        # 打乱顺序
        class_df = class_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        # 计算各部分的样本数
        n_train = int(n_samples * TRAIN_RATIO)
        n_val = int(n_samples * VAL_RATIO)
        # test 取剩余的，避免舍入误差

        # 分割
        train_dfs.append(class_df[:n_train])
        val_dfs.append(class_df[n_train:n_train + n_val])
        test_dfs.append(class_df[n_train + n_val:])

    # 合并各类别的分割结果
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)

    # 再次打乱（可选，使不同类别混合）
    train_df = train_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print("✓ 分割完成！")
    print()

    # 6. 打印分割后的统计信息
    print("=" * 50)
    print("分割后各数据集统计:")
    print("=" * 50)

    for split_name, split_df in [
        ("训练集 (Train)", train_df),
        ("验证集 (Val)", val_df),
        ("测试集 (Test)", test_df)
    ]:
        print(f"\n{split_name}:")
        print(f"  总样本数: {len(split_df)} ({len(split_df)/len(df)*100:.1f}%)")
        print(f"  类别分布:")

        for label in sorted(split_df['label'].unique()):
            count = (split_df['label'] == label).sum()
            label_name = "IDC (浸润性导管癌)" if label == 0 else "ILC (浸润性小叶癌)"
            print(f"    Class {label} ({label_name}): {count} 样本 ({count/len(split_df)*100:.1f}%)")

    print()
    print("=" * 50)

    # 7. 创建输出目录
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # 8. 保存分割后的数据集
    print("\n正在保存分割后的CSV文件...")

    train_path = output_path / 'train.csv'
    val_path = output_path / 'val.csv'
    test_path = output_path / 'test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✓ 训练集已保存: {train_path} ({len(train_df)} 样本)")
    print(f"✓ 验证集已保存: {val_path} ({len(val_df)} 样本)")
    print(f"✓ 测试集已保存: {test_path} ({len(test_df)} 样本)")
    print()

    # 9. 验证分割比例
    print("=" * 50)
    print("分割比例验证:")
    print("=" * 50)
    total = len(df)
    print(f"训练集: {len(train_df)}/{total} = {len(train_df)/total:.1%} (目标: {TRAIN_RATIO:.1%})")
    print(f"验证集: {len(val_df)}/{total} = {len(val_df)/total:.1%} (目标: {VAL_RATIO:.1%})")
    print(f"测试集: {len(test_df)}/{total} = {len(test_df)/total:.1%} (目标: {TEST_RATIO:.1%})")
    print()

    print("🎉 数据集分割完成！可以开始训练了！")
    print()


if __name__ == '__main__':
    split_dataset()
