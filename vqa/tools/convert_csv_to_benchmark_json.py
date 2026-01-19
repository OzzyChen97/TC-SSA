"""
Convert SlideBench CSV to JSON format for benchmark evaluation.
"""

import csv
import json
import os
import argparse


def convert_csv_to_json(csv_path, features_dir, output_path):
    """
    Convert CSV benchmark to JSON format.
    
    Args:
        csv_path: Path to SlideBench CSV file
        features_dir: Directory containing patch features
        output_path: Path to save JSON file
    """
    samples = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            slide_id = row['Slide']
            question = row['Question']
            
            # Format choices
            choices = [
                f"A) {row['A']}",
                f"B) {row['B']}",
                f"C) {row['C']}",
                f"D) {row['D']}"
            ]
            
            answer = row['Answer']
            broad_category = row['Broad Category']
            narrow_category = row['Narrow Category']
            category = f"{broad_category} - {narrow_category}"
            
            # Construct features path
            features_path = os.path.join(features_dir, f"{slide_id}.pt")
            
            sample = {
                'slide_id': slide_id,
                'question': question,
                'choices': choices,
                'answer': answer,
                'category': category,
                'features_path': features_path
            }
            
            samples.append(sample)
    
    # Save JSON
    output_data = {'samples': samples}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Converted {len(samples)} samples from {csv_path}")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert CSV to JSON')
    
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to CSV file')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directory containing patch features')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Path to save JSON file')
    
    args = parser.parse_args()
    
    convert_csv_to_json(args.csv_path, args.features_dir, args.output_path)


if __name__ == '__main__':
    main()
