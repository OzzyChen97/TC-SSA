import pandas as pd
import os

# ================= 配置区域 (修改这里) =================
# 1. 你的 .pt 特征文件所在的文件夹路径
FEATURE_DIR = '/workspace/moe/CPathPatchFeature/brca/uni/pt_files'

# 2. BRCA临床数据 TSV 文件路径
CLINICAL_DATA_FILE = '/workspace/ETC/brca_tcga_clinical_data-2.tsv'

# 3. 输出结果文件名
OUTPUT_CSV = '/workspace/ETC/tcga_brca_dataset.csv'
# =======================================================

def prepare_dataset():
    print(f"正在读取临床数据...")

    # 1. 读取临床数据 TSV 文件
    try:
        df_clinical = pd.read_csv(CLINICAL_DATA_FILE, sep='\t')
        print(f"-> 加载临床数据: {len(df_clinical)} 例")

        # 2. 提取 Patient ID 和 Neoplasm Histologic Type Name 列
        # 只保留 IDC 和 ILC 的数据
        df_filtered = df_clinical[['Patient ID', 'Neoplasm Histologic Type Name']].copy()

        # 过滤出 IDC 和 ILC
        df_idc = df_filtered[df_filtered['Neoplasm Histologic Type Name'] == 'Infiltrating Ductal Carcinoma'].copy()
        df_idc['label'] = 0  # IDC -> Label 0

        df_ilc = df_filtered[df_filtered['Neoplasm Histologic Type Name'] == 'Infiltrating Lobular Carcinoma'].copy()
        df_ilc['label'] = 1  # ILC -> Label 1

        print(f"-> IDC (Label 0): {len(df_idc)} 例")
        print(f"-> ILC (Label 1): {len(df_ilc)} 例")

        # 合并
        all_patients = pd.concat([df_idc, df_ilc])
        all_patients = all_patients[['Patient ID', 'label']].copy()
        all_patients['Patient ID'] = all_patients['Patient ID'].astype(str).str.upper().str.strip()
        patient_to_label = dict(zip(all_patients['Patient ID'], all_patients['label']))

    except Exception as e:
        print(f"读取临床数据文件失败: {e}")
        return

    # 3. 制作一个字典: {'patient_id': label, ...}
    # 确保 ID 是大写且去空格

    print(f"-> 合并临床数据完成，共 {len(patient_to_label)} 个病人标签。")
    print("-" * 30)

    # 4. 扫描你的特征文件夹，进行匹配
    print(f"正在扫描特征文件夹: {FEATURE_DIR} ...")
    if not os.path.exists(FEATURE_DIR):
        print("错误：特征文件夹不存在！请检查路径。")
        return

    pt_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.pt')]
    print(f"-> 找到 {len(pt_files)} 个 .pt 文件。开始匹配...")

    matched_data = []
    unmatched_count = 0

    for filename in pt_files:
        # 文件名示例: TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.pt
        # 提取 slide_id 和 patient_id
        slide_id = filename.replace('.pt', '')

        # 根据TCGA文件命名规则提取 patient_id (前三个部分，用-连接)
        # 例如: TCGA-3C-AALI
        parts = slide_id.split('-')
        if len(parts) >= 3:
            patient_id = '-'.join(parts[:3]).upper()
        else:
            patient_id = slide_id.upper()

        if patient_id in patient_to_label:
            label = patient_to_label[patient_id]
            matched_data.append({
                'slide_id': slide_id,
                'label': label
            })
        else:
            # 未找到匹配的patient_id（可能患者不是IDC或ILC类型）
            unmatched_count += 1
            # print(f"未找到标签: {patient_id}")

    print(f"-> 匹配成功: {len(matched_data)} 个文件")
    print(f"-> 未匹配: {unmatched_count} 个文件（可能不是IDC或ILC类型）")

    # 5. 保存结果
    final_df = pd.DataFrame(matched_data)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("-" * 30)
    print(f"🎉 成功生成数据集文件: {OUTPUT_CSV}")
    print(f"📊 总样本数: {len(final_df)}")
    print(f"   Class 0 (IDC - 浸润性导管癌): {len(final_df[final_df['label']==0])}")
    print(f"   Class 1 (ILC - 浸润性小叶癌): {len(final_df[final_df['label']==1])}")
    print("\n现在你可以直接在 train.py 中加载这个 CSV 了！")
    print(f"特征文件路径: {FEATURE_DIR}")

if __name__ == '__main__':
    prepare_dataset()