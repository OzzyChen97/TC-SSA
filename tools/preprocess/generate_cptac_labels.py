import os
import pandas as pd
import re
import requests
import time

# ================= 配置区域 =================
# 你的 CPTAC 特征文件夹路径
FEATURE_DIR = "/workspace/ETC/CPathPatchFeature/cptac_nsclc/uni/pt_files"

# CPTAC 临床数据 Excel 文件路径
LUAD_DATA_FILE = '/workspace/ETC/data/cptac_nsclc/TCIA-CPTAC-LUAD_v13_20250801-nbia-digest.xlsx'
LSCC_DATA_FILE = '/workspace/ETC/data/cptac_nsclc/TCIA-CPTAC-LSCC_v15-nbia-digest.xlsx'

OUTPUT_CSV = "/workspace/ETC/data/cptac_nsclc_dataset.csv"
# ===========================================

def load_patient_labels():
    """
    从本地 Excel 文件加载病例标签
    CPTAC-LUAD -> label = 0 (LUAD腺癌)
    CPTAC-LSCC -> label = 1 (LUSC鳞癌)
    """
    print("正在读取临床数据...")

    patient_to_label = {}

    # 读取 LUAD 数据
    try:
        df_luad = pd.read_excel(LUAD_DATA_FILE)
        luad_patients = df_luad['Patient ID'].unique()
        print(f"-> LUAD (Label 0): {len(luad_patients)} 例")
        for patient_id in luad_patients:
            patient_to_label[str(patient_id).upper().strip()] = 0  # LUAD
    except Exception as e:
        print(f"读取 LUAD 文件失败: {e}")

    # 读取 LSCC 数据
    try:
        df_lscc = pd.read_excel(LSCC_DATA_FILE)
        lscc_patients = df_lscc['Patient ID'].unique()
        print(f"-> LSCC/LUSC (Label 1): {len(lscc_patients)} 例")
        for patient_id in lscc_patients:
            patient_to_label[str(patient_id).upper().strip()] = 1  # LUSC
    except Exception as e:
        print(f"读取 LSCC 文件失败: {e}")

    print(f"-> 本地Excel数据: {len(patient_to_label)} 个病人标签")
    print("-" * 50)

    return patient_to_label

def get_all_cptac_lung_cases():
    """
    从GDC API获取CPTAC-3项目中所有肺癌病例的标签
    """
    print("\n正在从GDC API查询CPTAC-3所有肺癌病例...")

    url = "https://api.gdc.cancer.gov/cases"

    # 查询CPTAC-3项目中所有肺癌相关的病例
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "project.project_id", "value": ["CPTAC-3"]}},
            {"op": "in", "content": {"field": "primary_site", "value": ["Bronchus and lung"]}}
        ]
    }

    params = {
        "filters": filters,
        "fields": "submitter_id,project.project_id,disease_type,primary_site,diagnoses.primary_diagnosis",
        "format": "json",
        "size": 1000  # 获取大量数据
    }

    try:
        response = requests.post(url, json=params)

        if response.status_code != 200:
            print(f"⚠️ API 返回错误码: {response.status_code}")
            return {}

        data = response.json()
        hits = data.get('data', {}).get('hits', [])
        total = data.get('data', {}).get('pagination', {}).get('total', 0)

        print(f"-> GDC API返回: {total} 个肺癌病例，获取了 {len(hits)} 个")

        # 分析并标注
        patient_labels = {}
        luad_count = 0
        lusc_count = 0

        for hit in hits:
            submitter_id = hit.get('submitter_id', '').upper()
            disease_type = hit.get('disease_type', '')
            primary_diagnosis = hit.get('diagnoses', [{}])[0].get('primary_diagnosis', '') if hit.get('diagnoses') else ''

            # 判断类型
            label = None
            if 'Adenocarcinoma' in disease_type or 'Adenocarcinoma' in primary_diagnosis or 'adenocarcinoma' in disease_type.lower():
                label = 0  # LUAD
                luad_count += 1
            elif 'Squamous' in disease_type or 'Squamous' in primary_diagnosis or 'squamous' in disease_type.lower():
                label = 1  # LUSC
                lusc_count += 1

            if label is not None:
                patient_labels[submitter_id] = label

        print(f"-> GDC标注结果: LUAD={luad_count}, LUSC={lusc_count}")

        return patient_labels

    except Exception as e:
        print(f"查询GDC API出错: {e}")
        return {}

def main():
    # 1. 加载标签：本地Excel + GDC API
    local_labels = load_patient_labels()
    gdc_labels = get_all_cptac_lung_cases()

    # 合并标签（本地优先）
    all_labels = {**gdc_labels, **local_labels}  # local_labels会覆盖gdc_labels
    print(f"\n合并后总标签数: {len(all_labels)} 个病人")
    print(f"  LUAD (Label 0): {sum(1 for v in all_labels.values() if v == 0)}")
    print(f"  LUSC (Label 1): {sum(1 for v in all_labels.values() if v == 1)}")
    print("-" * 50)

    # 2. 扫描特征文件
    print(f"\n正在扫描特征文件夹: {FEATURE_DIR} ...")
    if not os.path.exists(FEATURE_DIR):
        print("错误：特征文件夹不存在！请检查路径。")
        return

    pt_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.pt')]
    print(f"-> 找到 {len(pt_files)} 个 .pt 文件。开始匹配...")

    matched_data = []
    unmatched_count = 0

    for filename in pt_files:
        # 文件名示例: C3L-00001-21.pt
        slide_id = filename.replace('.pt', '')

        # 提取 patient_id
        match = re.search(r'(C3[LN]-[A-Z0-9]+)', filename)
        if match:
            patient_id = match.group(1).upper()
        else:
            print(f"⚠️ 跳过无法解析 ID 的文件: {filename}")
            unmatched_count += 1
            continue

        # 匹配标签
        if patient_id in all_labels:
            label = all_labels[patient_id]
            matched_data.append({
                'slide_id': slide_id,
                'label': label
            })
        else:
            # 未找到匹配的patient_id
            unmatched_count += 1

    print(f"\n匹配结果:")
    print(f"-> 匹配成功: {len(matched_data)} 个文件")
    print(f"-> 未匹配: {unmatched_count} 个文件")

    # 3. 保存结果
    if len(matched_data) == 0:
        print("\n❌ 错误：没有找到任何匹配的标签！")
        return

    final_df = pd.DataFrame(matched_data)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print("-" * 50)
    print(f"✅ 成功生成数据集文件: {OUTPUT_CSV}")
    print(f"📊 总样本数: {len(final_df)}")
    print(f"   Class 0 (LUAD - 腺癌): {len(final_df[final_df['label']==0])}")
    print(f"   Class 1 (LUSC - 鳞癌): {len(final_df[final_df['label']==1])}")
    print("\n现在你可以使用 generate_data.py 切分数据集了！")
    print(f"特征文件路径: {FEATURE_DIR}")

if __name__ == "__main__":
    main()