"""
Fetch TCGA BRCA histological subtype data from GDC API.

This script retrieves clinical data for BRCA samples and extracts
histological diagnosis information for IDC vs ILC classification.

Usage:
    python tools/fetch_tcga_histology.py --output_dir data/brca
"""

import os
import sys
import json
import argparse
import requests
import pandas as pd
from pathlib import Path

# GDC API endpoints
GDC_API_BASE = "https://api.gdc.cancer.gov"
CASES_ENDPOINT = f"{GDC_API_BASE}/cases"
FILES_ENDPOINT = f"{GDC_API_BASE}/files"


def fetch_brca_clinical_data(size=2000):
    """
    Fetch BRCA clinical data including histological diagnosis from GDC API.
    
    Returns:
        List of case dictionaries with clinical information
    """
    print("Fetching BRCA clinical data from GDC API...")
    
    # Query parameters for BRCA cases
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": "TCGA-BRCA"
                }
            }
        ]
    }
    
    # Fields to retrieve
    fields = [
        "case_id",
        "submitter_id",
        "diagnoses.primary_diagnosis",
        "diagnoses.morphology",
        "diagnoses.tissue_or_organ_of_origin",
        "diagnoses.site_of_resection_or_biopsy",
        "diagnoses.prior_malignancy",
        "diagnoses.ajcc_pathologic_stage",
        "demographic.gender",
        "demographic.race",
        "demographic.ethnicity",
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
        return None
    
    data = response.json()
    
    print(f"Retrieved {len(data['data']['hits'])} cases")
    return data['data']['hits']


def fetch_slide_file_mapping(size=5000):
    """
    Fetch mapping between slide submitter IDs and file names from GDC API.
    
    Returns:
        DataFrame with slide_id to file_name mapping
    """
    print("Fetching slide to file mapping from GDC API...")
    
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": "TCGA-BRCA"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Slide Image"
                }
            }
        ]
    }
    
    fields = [
        "file_name",
        "file_id",
        "cases.submitter_id",
        "cases.samples.submitter_id",
        "cases.samples.portions.slides.submitter_id"
    ]
    
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": size
    }
    
    response = requests.get(FILES_ENDPOINT, params=params)
    
    if response.status_code != 200:
        print(f"Error: API request failed with status {response.status_code}")
        return None
    
    data = response.json()
    print(f"Retrieved {len(data['data']['hits'])} slide files")
    
    return data['data']['hits']


def extract_histology_labels(cases):
    """
    Extract histological diagnosis and map to binary labels.
    
    IDC (Infiltrating duct carcinoma) = Label 0
    ILC (Infiltrating lobular carcinoma) = Label 1
    
    Args:
        cases: List of case dictionaries from GDC API
        
    Returns:
        DataFrame with case_id, submitter_id, diagnosis, and label
    """
    records = []
    
    for case in cases:
        case_id = case.get('case_id', '')
        submitter_id = case.get('submitter_id', '')
        
        # Get diagnosis information
        diagnoses = case.get('diagnoses', [])
        if not diagnoses:
            continue
            
        diagnosis = diagnoses[0]  # Usually single diagnosis per case
        primary_diagnosis = diagnosis.get('primary_diagnosis', '')
        morphology = diagnosis.get('morphology', '')
        
        # Get slide information
        samples = case.get('samples', [])
        slide_ids = []
        for sample in samples:
            portions = sample.get('portions', [])
            for portion in portions:
                slides = portion.get('slides', [])
                for slide in slides:
                    slide_submitter_id = slide.get('submitter_id', '')
                    if slide_submitter_id:
                        slide_ids.append(slide_submitter_id)
        
        # Classify based on morphology code or primary diagnosis
        # ICD-O-3 Morphology codes:
        # 8500/3 - Infiltrating duct carcinoma (IDC)
        # 8520/3 - Lobular carcinoma (ILC)
        # 8522/3 - Infiltrating duct and lobular carcinoma (Mixed)
        
        label = None
        histology_type = "Unknown"
        
        if morphology:
            morph_code = morphology.split('/')[0] if '/' in morphology else morphology
            if morph_code == '8500':
                label = 0
                histology_type = "IDC"
            elif morph_code == '8520':
                label = 1
                histology_type = "ILC"
            elif morph_code == '8522':
                histology_type = "Mixed"
                # Can decide to include or exclude mixed cases
        
        # Also check primary diagnosis text
        if label is None and primary_diagnosis:
            primary_lower = primary_diagnosis.lower()
            if 'infiltrating duct' in primary_lower or 'ductal' in primary_lower:
                label = 0
                histology_type = "IDC"
            elif 'lobular' in primary_lower:
                label = 1
                histology_type = "ILC"
        
        for slide_id in slide_ids:
            records.append({
                'case_id': case_id,
                'submitter_id': submitter_id,
                'slide_submitter_id': slide_id,
                'primary_diagnosis': primary_diagnosis,
                'morphology': morphology,
                'histology_type': histology_type,
                'label': label
            })
    
    df = pd.DataFrame(records)
    return df


def match_with_features(df, features_dir):
    """
    Match the clinical data with available feature files.
    
    Args:
        df: DataFrame with clinical data
        features_dir: Directory containing .pt feature files
        
    Returns:
        DataFrame with matched slide_id (feature file name) and labels
    """
    print(f"\nMatching with features in: {features_dir}")
    
    # Get all feature files
    feature_files = list(Path(features_dir).glob("*.pt"))
    print(f"Found {len(feature_files)} feature files")
    
    # Extract slide IDs from feature file names
    # Format: TCGA-XX-XXXX-01Z-00-DX1.UUID.pt
    feature_slides = {}
    for f in feature_files:
        # Get the part before the UUID
        name = f.stem  # Remove .pt
        parts = name.split('.')
        if len(parts) >= 2:
            slide_part = parts[0]  # TCGA-XX-XXXX-01Z-00-DX1
            feature_slides[slide_part] = f.name
    
    print(f"Extracted {len(feature_slides)} unique slide identifiers")
    
    # Match with clinical data
    matched_records = []
    for _, row in df.iterrows():
        slide_submitter = row['slide_submitter_id']
        
        # Try to match
        if slide_submitter in feature_slides:
            matched_records.append({
                'slide_id': feature_slides[slide_submitter],
                'submitter_id': row['submitter_id'],
                'histology_type': row['histology_type'],
                'label': row['label']
            })
    
    matched_df = pd.DataFrame(matched_records)
    return matched_df


def create_splits(df, train_ratio=0.7, seed=42):
    """
    Create train/test splits stratified by label.
    
    Args:
        df: DataFrame with slide_id and label
        train_ratio: Ratio of training data (default: 0.7 for 7:3 split)
        seed: Random seed
        
    Returns:
        train_df, test_df
    """
    from sklearn.model_selection import train_test_split
    
    # Filter to only IDC and ILC (valid labels)
    df_valid = df[df['label'].notna()].copy()
    df_valid['label'] = df_valid['label'].astype(int)
    
    print(f"\nCreating 7:3 splits from {len(df_valid)} samples with valid labels")
    
    # Single split: train vs test
    train_df, test_df = train_test_split(
        df_valid, 
        test_size=1-train_ratio,
        stratify=df_valid['label'],
        random_state=seed
    )
    
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description='Fetch TCGA BRCA histological subtype data'
    )
    parser.add_argument('--output_dir', type=str, default='data/brca',
                        help='Output directory for CSV files')
    parser.add_argument('--features_dir', type=str, 
                        default='data/CPathPatchFeature/brca/r50/pt_files',
                        help='Directory containing feature .pt files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for splits')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fetch clinical data
    cases = fetch_brca_clinical_data()
    if cases is None:
        print("Failed to fetch clinical data")
        return
    
    # Step 2: Extract histology labels
    clinical_df = extract_histology_labels(cases)
    print(f"\nExtracted clinical data for {len(clinical_df)} slides")
    
    # Print histology distribution
    print("\nHistology type distribution:")
    print(clinical_df['histology_type'].value_counts())
    
    # Step 3: Match with feature files
    if os.path.exists(args.features_dir):
        matched_df = match_with_features(clinical_df, args.features_dir)
        print(f"\nMatched {len(matched_df)} slides with features")
        
        if len(matched_df) > 0:
            print("\nMatched histology distribution:")
            print(matched_df['histology_type'].value_counts())
            
            print("\nMatched label distribution:")
            print(matched_df['label'].value_counts())
            
            # Step 4: Create train/test splits (7:3)
            train_df, test_df = create_splits(matched_df, seed=args.seed)
            
            # Remove .pt suffix from slide_id for compatibility with dataset.py
            train_df = train_df.copy()
            test_df = test_df.copy()
            train_df['slide_id'] = train_df['slide_id'].str.replace('.pt', '', regex=False)
            test_df['slide_id'] = test_df['slide_id'].str.replace('.pt', '', regex=False)
            
            # Save splits
            train_df[['slide_id', 'label']].to_csv(
                os.path.join(args.output_dir, 'train_histology.csv'), 
                index=False
            )
            test_df[['slide_id', 'label']].to_csv(
                os.path.join(args.output_dir, 'test_histology.csv'), 
                index=False
            )
            
            # Save full dataset
            matched_df.to_csv(
                os.path.join(args.output_dir, 'tcga_brca_histology_full.csv'),
                index=False
            )
            
            print(f"\n=== Split Statistics (7:3) ===")
            print(f"Train: {len(train_df)} samples ({len(train_df)/len(matched_df)*100:.1f}%)")
            print(f"  - IDC (label=0): {(train_df['label']==0).sum()}")
            print(f"  - ILC (label=1): {(train_df['label']==1).sum()}")
            print(f"Test: {len(test_df)} samples ({len(test_df)/len(matched_df)*100:.1f}%)")
            print(f"  - IDC (label=0): {(test_df['label']==0).sum()}")
            print(f"  - ILC (label=1): {(test_df['label']==1).sum()}")
            
            print(f"\nFiles saved to {args.output_dir}/")
        else:
            print("No matches found between clinical data and feature files")
    else:
        print(f"\nFeatures directory not found: {args.features_dir}")
        print("Saving raw clinical data only...")
        
        clinical_df.to_csv(
            os.path.join(args.output_dir, 'tcga_brca_clinical_raw.csv'),
            index=False
        )
    
    # Also save raw clinical data for reference
    clinical_df.to_csv(
        os.path.join(args.output_dir, 'tcga_brca_clinical_all.csv'),
        index=False
    )
    print(f"\nRaw clinical data saved to {args.output_dir}/tcga_brca_clinical_all.csv")


if __name__ == '__main__':
    main()
