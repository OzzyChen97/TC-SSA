#!/usr/bin/env python
"""
Filter SlideChat training data to only include samples with available feature files.
"""

import json
import os
import argparse
from pathlib import Path


def get_features_path(image_path, features_dir):
    """
    Try to find the feature file for a given image path.
    Returns the path if found, None otherwise.
    """
    # Parse the image path: "./LUAD/TCGA-05-5425-01Z-00-DX1.csv"
    rel_path = image_path.lstrip('./')
    parts = rel_path.split('/')
    
    if len(parts) >= 2:
        cancer_type = parts[0]  # e.g., "LUAD"
        csv_filename = parts[1]  # e.g., "TCGA-05-5425-01Z-00-DX1.csv"
        slide_id = csv_filename.replace('.csv', '')
    else:
        slide_id = os.path.basename(rel_path).replace('.csv', '')
        cancer_type = None
    
    # Map cancer types to TCGA directory names
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
        'OV': 'TCGA-Rest',
        'KIRC': 'TCGA-Rest',
        'KIRP': 'TCGA-Rest',
        'LIHC': 'TCGA-Rest',
        'STAD': 'TCGA-Rest',
        'UCEC': 'TCGA-Rest',
        'THCA': 'TCGA-Rest',
        'PRAD': 'TCGA-Rest',
        'PAAD': 'TCGA-Rest',
    }
    
    # Build possible directory paths
    possible_dirs = []
    if cancer_type:
        tcga_dir = tcga_mapping.get(cancer_type, f'TCGA-{cancer_type}')
        possible_dirs.extend([
            os.path.join(features_dir, tcga_dir, tcga_dir),
            os.path.join(features_dir, tcga_dir, 'feat'),
            os.path.join(features_dir, tcga_dir),
        ])
    
    # Also check TCGA-Rest
    possible_dirs.extend([
        os.path.join(features_dir, 'TCGA-Rest', 'TCGA-Rest'),
        os.path.join(features_dir, 'TCGA-Rest', 'feat'),
        os.path.join(features_dir, 'TCGA-Rest'),
    ])
    
    # Search for feature files
    for search_dir in possible_dirs:
        if not os.path.exists(search_dir):
            continue
        
        try:
            files = os.listdir(search_dir)
            matching_files = [f for f in files if f.startswith(slide_id)]
            
            if matching_files:
                # Prefer 512 dim files (ConCH)
                for priority_pattern in ['_0_512.npy', '_1_512.npy', '_0_1024.npy', '_1_1024.npy', '.pt', '.pth', '.npy']:
                    for fname in matching_files:
                        if priority_pattern in fname or fname.endswith(priority_pattern):
                            return os.path.join(search_dir, fname)
                
                return os.path.join(search_dir, matching_files[0])
        except (OSError, PermissionError):
            continue
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Filter training data for available features')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Input SlideInstruct JSON file')
    parser.add_argument('--features_dir', type=str, required=True,
                        help='Directory containing feature files')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Output filtered JSON file')
    args = parser.parse_args()
    
    # Load input data
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    print(f"Total samples in input: {len(data)}")
    
    # Filter samples
    available_samples = []
    missing_samples = []
    
    for item in data:
        image_path = item['image'][0] if isinstance(item['image'], list) else item['image']
        feat_path = get_features_path(image_path, args.features_dir)
        
        if feat_path and os.path.exists(feat_path):
            available_samples.append(item)
        else:
            missing_samples.append({
                'id': item['id'],
                'image': image_path
            })
    
    print(f"Samples with available features: {len(available_samples)}")
    print(f"Samples with missing features: {len(missing_samples)}")
    
    # Save filtered data
    with open(args.output_json, 'w') as f:
        json.dump(available_samples, f, indent=2)
    
    print(f"Saved filtered data to: {args.output_json}")
    
    # Save missing samples list for reference
    missing_file = args.output_json.replace('.json', '_missing.json')
    with open(missing_file, 'w') as f:
        json.dump(missing_samples, f, indent=2)
    
    print(f"Saved missing samples list to: {missing_file}")
    
    # Print some missing examples
    if missing_samples:
        print("\nExample missing samples:")
        for sample in missing_samples[:5]:
            print(f"  - {sample['image']}")


if __name__ == '__main__':
    main()
