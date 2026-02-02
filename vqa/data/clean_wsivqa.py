"""
数据清洗脚本：
1. 从 WsiVQA_test.json 中移除已存在于 SlideInstruct_train_stage2_vqa_filtered.json 的图片
2. 移除在 GTEx-TCGA-Embeddings 中没有对应特征的图片
"""

import json
import os
from pathlib import Path

# 路径配置
BASE_DIR = Path("/workspace/zhuo/ETC/vqa/data")
WSIVQA_TEST_PATH = BASE_DIR / "WsiVQA_test.json"
SLIDECHAT_PATH = BASE_DIR / "SlideChat/SlideInstruct_train_stage2_vqa_filtered.json"
EMBEDDINGS_DIR = BASE_DIR / "GTEx-TCGA-Embeddings"
OUTPUT_PATH = BASE_DIR / "WsiVQA_test_cleaned.json"

def extract_case_id(s):
    """从字符串中提取 TCGA case ID (如 TCGA-LD-A74U)"""
    # 匹配 TCGA-XX-XXXX 格式
    import re
    match = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', s)
    return match.group(0) if match else None

def extract_slide_id(s):
    """从字符串中提取完整的 slide ID (如 TCGA-LD-A74U-01Z-00-DX1)"""
    import re
    match = re.search(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2,3}-[A-Z0-9]{2,3}-[A-Z0-9]+', s)
    return match.group(0) if match else None

def main():
    print("=" * 60)
    print("数据清洗脚本")
    print("=" * 60)
    
    # Step 1: 加载 WsiVQA_test.json
    print("\n[1/4] 加载 WsiVQA_test.json...")
    with open(WSIVQA_TEST_PATH, 'r') as f:
        wsivqa_data = json.load(f)
    print(f"  - 原始数据条数: {len(wsivqa_data)}")
    
    # 获取 WsiVQA 中的唯一 case IDs
    wsivqa_case_ids = set()
    for item in wsivqa_data:
        case_id = item.get('Id')
        if case_id:
            wsivqa_case_ids.add(case_id)
    print(f"  - 唯一 case ID 数量: {len(wsivqa_case_ids)}")
    
    # Step 2: 加载 SlideInstruct 并提取 case IDs
    print("\n[2/4] 加载 SlideInstruct_train_stage2_vqa_filtered.json...")
    with open(SLIDECHAT_PATH, 'r') as f:
        slidechat_data = json.load(f)
    print(f"  - SlideChat 数据条数: {len(slidechat_data)}")
    
    # 提取 SlideChat 中的所有 case IDs
    slidechat_case_ids = set()
    for item in slidechat_data:
        images = item.get('image', [])
        for img in images:
            case_id = extract_case_id(img)
            if case_id:
                slidechat_case_ids.add(case_id)
    print(f"  - SlideChat 唯一 case ID 数量: {len(slidechat_case_ids)}")
    
    # Step 3: 获取 Embeddings 中存在的 case IDs
    print("\n[3/4] 扫描 GTEx-TCGA-Embeddings 目录...")
    embedding_case_ids = set()
    
    for root, dirs, files in os.walk(EMBEDDINGS_DIR):
        for f in files:
            if f.endswith('.npy'):
                case_id = extract_case_id(f)
                if case_id:
                    embedding_case_ids.add(case_id)
    print(f"  - Embeddings 唯一 case ID 数量: {len(embedding_case_ids)}")
    
    # Step 4: 过滤数据
    print("\n[4/4] 过滤数据...")
    
    # 统计
    removed_in_slidechat = 0
    removed_no_embedding = 0
    kept = 0
    
    cleaned_data = []
    removed_case_ids_slidechat = set()
    removed_case_ids_no_embedding = set()
    
    for item in wsivqa_data:
        case_id = item.get('Id')
        
        # 条件1: 如果 case_id 已经在 SlideChat 中，跳过
        if case_id in slidechat_case_ids:
            removed_in_slidechat += 1
            removed_case_ids_slidechat.add(case_id)
            continue
        
        # 条件2: 如果 case_id 在 Embeddings 中不存在，跳过
        if case_id not in embedding_case_ids:
            removed_no_embedding += 1
            removed_case_ids_no_embedding.add(case_id)
            continue
        
        # 保留
        cleaned_data.append(item)
        kept += 1
    
    print(f"\n结果统计:")
    print(f"  - 原始数据条数: {len(wsivqa_data)}")
    print(f"  - 因在 SlideChat 中已存在而移除: {removed_in_slidechat} 条 (涉及 {len(removed_case_ids_slidechat)} 个 case ID)")
    print(f"  - 因无 Embedding 特征而移除: {removed_no_embedding} 条 (涉及 {len(removed_case_ids_no_embedding)} 个 case ID)")
    print(f"  - 最终保留: {kept} 条")
    
    # 保存结果
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(cleaned_data, f, indent=4)
    print(f"\n已保存清洗后的数据到: {OUTPUT_PATH}")
    
    # 打印一些被移除的样例
    print("\n被移除的 case ID 样例 (SlideChat 中已存在):")
    for cid in list(removed_case_ids_slidechat)[:5]:
        print(f"  - {cid}")
    
    print("\n被移除的 case ID 样例 (无 Embedding):")
    for cid in list(removed_case_ids_no_embedding)[:5]:
        print(f"  - {cid}")

if __name__ == "__main__":
    main()
