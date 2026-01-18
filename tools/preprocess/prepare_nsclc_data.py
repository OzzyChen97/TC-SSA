import pandas as pd
import os

# ================= 配置区域 (修改这里) =================
# 1. 你的 .pt 特征文件所在的文件夹路径
FEATURE_DIR = '/workspace/ETC/CPathPatchFeature/nsclc/uni/pt_files'

# 2. LUAD和LUSC临床数据 CSV 文件路径
LUAD_CLINICAL_FILE = '/workspace/ETC/TCGA-LUAD _clinfile .csv'
LUSC_CLINICAL_FILE = '/workspace/ETC/TCGA-LUSC _clinfile .csv'

# 3. 输出结果文件名
OUTPUT_CSV = '/workspace/ETC/tcga_nsclc_dataset.csv'
# =======================================================

def prepare_dataset():
    print(f"正在读取临床数据...")

    # 1. 读取LUAD临床数据 CSV 文件
    try:
        df_luad = pd.read_csv(LUAD_CLINICAL_FILE)
        print(f"-> 加载LUAD临床数据: {len(df_luad)} 例")

        # 2. 读取LUSC临床数据 CSV 文件
        df_lusc = pd.read_csv(LUSC_CLINICAL_FILE)
        print(f"-> 加载LUSC临床数据: {len(df_lusc)} 例")

        # 3. 提取 bcr_patient_barcode 列，并添加标签
        # LUAD -> Label 0
        df_luad_filtered = df_luad[['bcr_patient_barcode']].copy()
        df_luad_filtered['label'] = 0  # LUAD -> Label 0

        # LUSC -> Label 1
        df_lusc_filtered = df_lusc[['bcr_patient_barcode']].copy()
        df_lusc_filtered['label'] = 1  # LUSC -> Label 1

        print(f"-> LUAD (Label 0 - 肺腺癌): {len(df_luad_filtered)} 例")
        print(f"-> LUSC (Label 1 - 肺鳞癌): {len(df_lusc_filtered)} 例")

        # 4. 合并两个数据集
        all_patients = pd.concat([df_luad_filtered, df_lusc_filtered], ignore_index=True)
        all_patients['bcr_patient_barcode'] = all_patients['bcr_patient_barcode'].astype(str).str.upper().str.strip()
        patient_to_label = dict(zip(all_patients['bcr_patient_barcode'], all_patients['label']))

    except Exception as e:
        print(f"读取临床数据文件失败: {e}")
        return

    print(f"-> 合并临床数据完成，共 {len(patient_to_label)} 个病人标签。")
    print("-" * 30)

    # 5. 扫描你的特征文件夹，进行匹配
    print(f"正在扫描特征文件夹: {FEATURE_DIR} ...")
    if not os.path.exists(FEATURE_DIR):
        print("错误：特征文件夹不存在！请检查路径。")
        return

    pt_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.pt')]
    print(f"-> 找到 {len(pt_files)} 个 .pt 文件。开始匹配...")

    matched_data = []
    unmatched_count = 0

    for filename in pt_files:
        # 文件名示例: TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01.pt
        # 提取 slide_id 和 patient_id
        slide_id = filename.replace('.pt', '')

        # 根据TCGA文件命名规则提取 patient_id (前三个部分，用-连接)
        # 例如: TCGA-05-4244
        parts = slide_id.split('-')
        if len(parts) >= 3:
            patient_id = '-'.join(parts[:3]).upper()
        else:
            patient_id = slide_id.upper()

        if patient_id in patient_to_label:
            label = patient_to_label[patient_id]
            matched_data.append({
                'slide_id': slide_id,
                'patient_id': patient_id,  # 重要：保存patient_id用于后续数据划分
                'label': label
            })
        else:
            # 未找到匹配的patient_id
            unmatched_count += 1
            # print(f"未找到标签: {patient_id}")

    print(f"-> 匹配成功: {len(matched_data)} 个文件")
    print(f"-> 未匹配: {unmatched_count} 个文件")

    # 6. 保存结果
    final_df = pd.DataFrame(matched_data)
    final_df.to_csv(OUTPUT_CSV, index=False)

    # 7. 统计信息
    print("-" * 30)
    print(f"🎉 成功生成数据集文件: {OUTPUT_CSV}")
    print(f"📊 总样本数: {len(final_df)}")
    print(f"   Class 0 (LUAD - 肺腺癌): {len(final_df[final_df['label']==0])}")
    print(f"   Class 1 (LUSC - 肺鳞癌): {len(final_df[final_df['label']==1])}")

    # 统计唯一病人数
    unique_patients = final_df['patient_id'].nunique()
    print(f"\n👤 唯一病人数: {unique_patients}")
    print(f"   平均每个病人的切片数: {len(final_df) / unique_patients:.2f}")

    print("\n⚠️  重要提示：")
    print("   TCGA数据集必须按Patient ID划分训练/验证/测试集！")
    print("   同一个病人的所有切片必须在同一个集合中，避免数据泄露。")
    print("\n现在你可以直接在 train.py 中加载这个 CSV 了！")
    print(f"特征文件路径: {FEATURE_DIR}")

if __name__ == '__main__':
    prepare_dataset()
