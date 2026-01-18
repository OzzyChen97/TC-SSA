#!/usr/bin/env python
"""
Download WSI slides from GDC Data Portal.

Usage:
    python tools/download_gdc_slides.py --slide_ids TCGA-05-4402-01Z-00-DX1... --output_dir ./slides
"""

import argparse
import os
import json
import requests
from pathlib import Path


def get_gdc_file_uuid(slide_id):
    """Query GDC API to get file UUID from slide ID."""
    files_endpt = "https://api.gdc.cancer.gov/files"
    
    # Try with .svs extension first
    for ext in ['.svs', '.SVS', '']:
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "=",
                    "content": {
                        "field": "file_name",
                        "value": f"{slide_id}{ext}"
                    }
                }
            ]
        }
        
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size",
            "format": "JSON",
            "size": "1"
        }
        
        try:
            response = requests.get(files_endpt, params=params, timeout=30)
            data = response.json()
            
            if data.get('data', {}).get('hits'):
                file_info = data['data']['hits'][0]
                return file_info['file_id'], file_info.get('file_size', 0), file_info.get('file_name', '')
        except Exception as e:
            print(f"  Error querying GDC API: {e}")
    
    return None, 0, ''


def download_slide_from_gdc(file_uuid, output_path):
    """Download slide from GDC using file UUID."""
    data_endpt = f"https://api.gdc.cancer.gov/data/{file_uuid}"
    
    try:
        print(f"  Downloading from GDC...")
        response = requests.get(data_endpt, stream=True, timeout=600)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            mb_down = downloaded / (1024*1024)
                            mb_total = total_size / (1024*1024)
                            print(f"\r  Progress: {pct:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end='', flush=True)
            print()
            return True
        else:
            print(f"  HTTP Error: {response.status_code}")
    except Exception as e:
        print(f"  Error downloading: {e}")
    
    return False


def main():
    parser = argparse.ArgumentParser(description='Download WSI slides from GDC')
    parser.add_argument('--slide_ids', type=str, nargs='+', required=True,
                        help='Slide IDs to download')
    parser.add_argument('--output_dir', type=str, default='./slides',
                        help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Downloading {len(args.slide_ids)} slides to {args.output_dir}")
    print("=" * 60)
    
    for slide_id in args.slide_ids:
        print(f"\nSlide: {slide_id}")
        
        # Check if already downloaded
        output_path = os.path.join(args.output_dir, f"{slide_id}.svs")
        if os.path.exists(output_path):
            print(f"  Already exists, skipping...")
            continue
        
        # Get file UUID
        file_uuid, file_size, file_name = get_gdc_file_uuid(slide_id)
        
        if file_uuid:
            print(f"  Found: {file_name} ({file_size / (1024*1024*1024):.2f} GB)")
            print(f"  UUID: {file_uuid}")
            
            # Download
            if download_slide_from_gdc(file_uuid, output_path):
                print(f"  Success: {output_path}")
            else:
                print(f"  Failed to download")
        else:
            print(f"  Not found on GDC")
    
    print("\n" + "=" * 60)
    print("Download complete!")


if __name__ == '__main__':
    main()
