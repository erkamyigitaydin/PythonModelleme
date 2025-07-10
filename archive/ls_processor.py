#!/usr/bin/env python3
"""
Label Studio to YOLO Format Converter
Converts Label Studio JSON exports to YOLO detection format.
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
import yaml

def create_directories(output_dir):
    """Create necessary directories for YOLO dataset."""
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    return images_dir, labels_dir

def get_image_dimensions(image_path):
    """Get image width and height."""
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None

def convert_bbox_to_yolo(bbox, img_width, img_height):
    """
    Convert Label Studio bbox to YOLO format.
    Label Studio: [x, y, width, height] in percentages (0-100)
    YOLO: [x_center, y_center, width, height] in normalized coordinates (0-1)
    """
    # Convert percentages to pixels
    x_px = (bbox['x'] / 100.0) * img_width
    y_px = (bbox['y'] / 100.0) * img_height
    w_px = (bbox['width'] / 100.0) * img_width
    h_px = (bbox['height'] / 100.0) * img_height
    
    # Calculate center coordinates
    x_center = (x_px + w_px / 2) / img_width
    y_center = (y_px + h_px / 2) / img_height
    
    # Normalize width and height
    width_norm = w_px / img_width
    height_norm = h_px / img_height
    
    return x_center, y_center, width_norm, height_norm

def process_label_studio_json(json_path, images_dir_source, images_dir_target, labels_dir, classes):
    """Process Label Studio JSON export and convert to YOLO format."""
    
    class_to_id = {class_name: idx for idx, class_name in enumerate(classes)}
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_count = 0
    
    for item in data:
        # Get image info
        if 'data' not in item:
            continue
        
        # Try different possible fields for image URL
        image_url = None
        for field in ['ocr', 'image', 'data']:
            if field in item['data']:
                image_url = item['data'][field]
                break
        
        if image_url is None:
            print(f"Warning: No image URL found in data fields: {list(item['data'].keys())}")
            continue
        
        # Extract filename from URL (usually in format /data/upload/2/prefix-filename.jpg)
        if '?d=' in image_url:
            filename = image_url.split('?d=')[-1]
        else:
            filename = os.path.basename(image_url)
        
        # Remove prefix from Label Studio filename (e.g., "48aae7ef-WhatsApp_Image..." -> "WhatsApp_Image...")
        if '-' in filename:
            filename = filename.split('-', 1)[1]  # Remove everything before first dash
        
        # Convert underscores to spaces and parentheses format
        # "WhatsApp_Image_2025-07-03_at_14.07.58_1.jpeg" -> "WhatsApp Image 2025-07-03 at 14.07.58 (1).jpeg"
        original_filename = filename.replace('_at_', ' at ').replace('_', ' ')
        
        # Handle numbered files: " 1.jpeg" -> " (1).jpeg"
        import re
        original_filename = re.sub(r' (\d+)\.(\w+)$', r' (\1).\2', original_filename)
        
        # Try multiple possible filenames
        possible_filenames = [
            filename,           # Original Label Studio name
            original_filename,  # Converted name with spaces and parentheses
            filename.replace('_', ' '),  # Simple underscore to space conversion
        ]
        
        source_image_path = None
        for possible_name in possible_filenames:
            test_path = Path(images_dir_source) / possible_name
            if test_path.exists():
                source_image_path = test_path
                filename = possible_name  # Use the found filename
                break
        
        if source_image_path is None:
            print(f"Warning: Source image not found. Tried: {possible_filenames}")
            continue
        
        # Copy image to target directory
        target_image_path = images_dir_target / filename
        shutil.copy2(source_image_path, target_image_path)
        
        # Get image dimensions
        img_width, img_height = get_image_dimensions(target_image_path)
        if img_width is None or img_height is None:
            continue
        
        # Process annotations
        annotations = []
        if 'annotations' in item and item['annotations']:
            for annotation in item['annotations']:
                if 'result' not in annotation:
                    continue
                
                # Group results by ID to match rectangles with labels
                results_by_id = {}
                for result in annotation['result']:
                    result_id = result.get('id', '')
                    if result_id not in results_by_id:
                        results_by_id[result_id] = {}
                    results_by_id[result_id][result.get('type', '')] = result
                
                # Process each bounding box with its corresponding label
                for result_id, grouped_results in results_by_id.items():
                    # Look for rectangle/bbox and labels
                    bbox_result = None
                    label_result = None
                    
                    # Check for different possible types
                    if 'rectangle' in grouped_results:
                        bbox_result = grouped_results['rectangle']
                    elif 'rectanglelabels' in grouped_results:
                        bbox_result = grouped_results['rectanglelabels']
                    
                    if 'labels' in grouped_results:
                        label_result = grouped_results['labels']
                    elif 'rectanglelabels' in grouped_results:
                        label_result = grouped_results['rectanglelabels']
                    
                    # Skip if we don't have both bbox and label
                    if not bbox_result or not label_result:
                        continue
                    
                    # Get class name from labels
                    class_names = label_result['value'].get('labels', [])
                    if not class_names:
                        # Try rectanglelabels format
                        class_names = label_result['value'].get('rectanglelabels', [])
                    
                    if not class_names:
                        continue
                    
                    class_name = class_names[0]  # Take first class if multiple
                    if class_name not in class_to_id:
                        print(f"Warning: Unknown class '{class_name}' in {filename}")
                        continue
                    
                    class_id = class_to_id[class_name]
                    
                    # Get bounding box coordinates
                    bbox = bbox_result['value']
                    
                    # Skip if it's a polygon (we only handle rectangles)
                    if 'points' in bbox:
                        print(f"Warning: Skipping polygon annotation for class '{class_name}' in {filename}")
                        continue
                    
                    x_center, y_center, width, height = convert_bbox_to_yolo(
                        bbox, img_width, img_height
                    )
                    
                    # Format: class_id x_center y_center width height
                    annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Write label file
        label_filename = Path(filename).stem + '.txt'
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(annotations))
        
        processed_count += 1
        print(f"Processed: {filename} ({len(annotations)} annotations)")
    
    return processed_count

def create_yaml_config(output_dir, classes):
    """Create YOLO dataset configuration file."""
    
    config = {
        'path': str(Path(output_dir).absolute()),  # Dataset root dir
        'train': 'images',  # Train images (relative to 'path')
        'val': 'images',    # Val images (relative to 'path') - same as train for now
        'test': 'images',   # Test images (optional)
        'nc': len(classes), # Number of classes
        'names': classes    # Class names
    }
    
    yaml_path = Path(output_dir) / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"Created YOLO config: {yaml_path}")
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description='Convert Label Studio JSON to YOLO format')
    parser.add_argument('--ls-json', required=True, help='Path to Label Studio JSON export')
    parser.add_argument('--images-dir', required=True, help='Directory containing source images')
    parser.add_argument('--output-dir', required=True, help='Output directory for YOLO dataset')
    parser.add_argument('--classes', nargs='+', required=True, help='List of class names')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.ls_json):
        print(f"Error: Label Studio JSON file not found: {args.ls_json}")
        return
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return
    
    print(f"Processing Label Studio export: {args.ls_json}")
    print(f"Source images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Classes ({len(args.classes)}): {args.classes}")
    
    # Create output directories
    images_dir, labels_dir = create_directories(args.output_dir)
    
    # Process JSON and convert to YOLO
    processed_count = process_label_studio_json(
        args.ls_json, 
        args.images_dir, 
        images_dir, 
        labels_dir, 
        args.classes
    )
    
    # Create YAML config
    yaml_path = create_yaml_config(args.output_dir, args.classes)
    
    print(f"\n=== Conversion Complete ===")
    print(f"Processed {processed_count} images")
    print(f"Dataset structure:")
    print(f"  {args.output_dir}/")
    print(f"  ├── images/     ({len(list(images_dir.glob('*')))} files)")
    print(f"  ├── labels/     ({len(list(labels_dir.glob('*')))} files)")
    print(f"  └── data.yaml   (YOLO config)")
    print(f"\nReady for YOLO training with: {yaml_path}")

if __name__ == "__main__":
    main() 