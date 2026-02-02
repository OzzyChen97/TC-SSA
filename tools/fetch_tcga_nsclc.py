"""
Fetch TCGA NSCLC histological subtype data from GDC API.

This script retrieves clinical data for NSCLC samples and extracts
histological diagnosis information for LUAD vs LUSC classification.

LUAD (Lung Adenocarcinoma) = Label 0
LUSC (Lung Squamous Cell Carcinoma) = Label 1

Usage:
    python tools/fetch_tcga_nsclc.py --output_dir data/nsclc
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# GDC API endpoints
GDC_API_BASE = "https://api.gdc.cancer.gov"
CASES_ENDPOINT = f"{GDC_API_BASE}/cases"


def fetch_nsclc_clinical_data(size=2000):
    """
    Fetch NSCLC (LUAD + LUSC) clinical data from GDC API.
    
    Returns:
        List of case dictionaries with clinical information
    """
    print("Fetching NSCLC clinical data from GDC API...")
    
    all_cases = []
    
    for project_id in ["TCGA-LUAD", "TCGA-LUSC"]:
        print(f"  Fetching {project_id}...")
        
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": project_id
                    }
                }
            ]
        }
        
        fields = [
            "case_id",
            "submitter_id",
            "project.project_id",
            "diagnoses.primary_diagnosis",
            "diagnoses.morphology",
            "samples.sample_id",
            "samples.submitter_id",
            "samples.sample_type",
            "samples.portions.slides.slide_id",
            "samples.portions.slides.submitter_id",
        ]
        
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": size
        }
        
        response = requests.get(CASES_ENDPOINT, params=params)
        
        if response.status_code != 200:
            print(f"Error: API request failed with status {response.status_code}")
            print(response.text)
            continue
        
        data = response.json()
        cases = data['data']['hits']
        
        # Add project_id to each case for label assignment
        for case in cases:
            case['project_id'] = project_id
        
        all_cases.extend(cases)
        print(f"    Retrieved {len(cases)} cases")
    
    print(f"Total: {len(all_cases)} cases")
    return all_cases


def extract_nsclc_labels(cases):
    """
    Extract NSCLC labels based on project.
    
    LUAD (Lung Adenocarcinoma) = Label 0
    LUSC (Lung Squamous Cell Carcinoma) = Label 1
    
    Args:
        cases: List of case dictionaries from GDC API
        
    Returns:
        DataFrame with slide_id and label
    """
    records = []
    
    for case in cases:
        case_id = case.get('case_id', '')
        submitter_id = case.get('submitter_id', '')
        project_id = case.get('project_id', '')
        
        # Assign label based on project
        if project_id == 'TCGA-LUAD':
            label = 0
            cancer_type = 'LUAD'
        elif project_id == 'TCGA-LUSC':
            label = 1
            cancer_type = 'LUSC'
        else:
            continue
        
        # Get slide information
        samples = case.get('samples', [])
        for sample in samples:
            portions = sample.get('portions', [])
            for portion in portions:
                slides = portion.get('slides', [])
                for slide in slides:
                    slide_submitter_id = slide.get('submitter_id', '')
                    if slide_submitter_id:
                        records.append({
                            'case_id': case_id,
                            'submitter_id': submitter_id,
                            'slide_submitter_id': slide_submitter_id,
                            'project_id': project_id,
                            'cancer_type': cancer_type,
                            'label': label
                        })
    
    df = pd.DataFrame(records)
    return df


def match_with_features(df, features_dir):
    """
    Match the clinical data with available feature files.
    """
    print(f"\nMatching with features in: {features_dir}")
    
    # Get all feature files
    feature_files = list(Path(features_dir).glob("*.pt"))
    print(f"Found {len(feature_files)} feature files")
    
    # Extract slide IDs from feature file names
    feature_slides = {}
    for f in feature_files:
        name = f.stem  # Remove .pt
        parts = name.split('.')
        if len(parts) >= 2:
            slide_part = parts[0]
            feature_slides[slide_part] = name  # Store without .pt
    
    print(f"Extracted {len(feature_slides)} unique slide identifiers")
    
    # Match with clinical data
    matched_records = []
    for _, row in df.iterrows():
        slide_submitter = row['slide_submitter_id']
        
        if slide_submitter in feature_slides:
            matched_records.append({
                'slide_id': feature_slides[slide_submitter],
                'submitter_id': row['submitter_id'],
                'cancer_type': row['cancer_type'],
                'label': row['label']
            })
    
    matched_df = pd.DataFrame(matched_records)
    return matched_df


def create_splits(df, train_ratio=0.7, seed=42):
    """
    Create train/test splits stratified by label (7:3).
    """
    df_valid = df.copy()
    df_valid['label'] = df_valid['label'].astype(int)
    
    print(f"\nCreating 7:3 splits from {len(df_valid)} samples")
    
    train_df, test_df = train_test_split(
        df_valid, 
        test_size=1-train_ratio,
        stratify=df_valid['label'],
        random_state=seed
    )
    
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description='Fetch TCGA NSCLC data for LUAD vs LUSC classification'
    )
    parser.add_argument('--output_dir', type=str, default='data/nsclc',
                        help='Output directory for CSV files')
    parser.add_argument('--features_dir', type=str, 
                        default='data/CPathPatchFeature/nsclc/uni/pt_files',
                        help='Directory containing feature .pt files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fetch clinical data
    cases = fetch_nsclc_clinical_data()
    if not cases:
        print("Failed to fetch clinical data")
        return
    
    # Step 2: Extract labels
    clinical_df = extract_nsclc_labels(cases)
    print(f"\nExtracted clinical data for {len(clinical_df)} slides")
    
    print("\nCancer type distribution:")
    print(clinical_df['cancer_type'].value_counts())
    
    # Step 3: Match with feature files
    if os.path.exists(args.features_dir):
        matched_df = match_with_features(clinical_df, args.features_dir)
        print(f"\nMatched {len(matched_df)} slides with features")
        
        if len(matched_df) > 0:
            print("\nMatched cancer type distribution:")
            print(matched_df['cancer_type'].value_counts())
            
            # Step 4: Create train/test splits (7:3)
            train_df, test_df = create_splits(matched_df, seed=args.seed)
            
            # Save splits
            train_df[['slide_id', 'label']].to_csv(
                os.path.join(args.output_dir, 'train.csv'), 
                index=False
            )
            test_df[['slide_id', 'label']].to_csv(
                os.path.join(args.output_dir, 'test.csv'), 
                index=False
            )
            
            # Save full dataset
            matched_df.to_csv(
                os.path.join(args.output_dir, 'tcga_nsclc_full.csv'),
                index=False
            )
            
            print(f"\n=== Split Statistics (7:3) ===")
            print(f"Train: {len(train_df)} samples (70%)")
            print(f"  - LUAD (label=0): {(train_df['label']==0).sum()}")
            print(f"  - LUSC (label=1): {(train_df['label']==1).sum()}")
            print(f"Test: {len(test_df)} samples (30%)")
            print(f"  - LUAD (label=0): {(test_df['label']==0).sum()}")
            print(f"  - LUSC (label=1): {(test_df['label']==1).sum()}")
            
            print(f"\nFiles saved to {args.output_dir}/")
        else:
            print("No matches found between clinical data and feature files")
    else:
        print(f"\nFeatures directory not found: {args.features_dir}")
    
    # Save raw clinical data
    clinical_df.to_csv(
        os.path.join(args.output_dir, 'tcga_nsclc_clinical_all.csv'),
        index=False
    )
    print(f"\nRaw clinical data saved to {args.output_dir}/tcga_nsclc_clinical_all.csv")


if __name__ == '__main__':
    main()
