# Label Studio to YOLO Dataset Converter

## Overview
This project converts Label Studio JSON exports to YOLO format for training object detection models on receipt/slip data.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Class List:**
   The system recognizes these 13 classes (see `classes.py`):
   - company_name
   - address
   - tax_office
   - tax_number
   - date
   - time
   - category
   - tax_amount
   - total_amount
   - payment_method
   - merchant_number
   - receipt_number
   - currency

## Usage

### Step 1: Prepare Your Data
- Export your Label Studio project as JSON
- Have your original images in a directory
- Update the paths in the command below

### Step 2: Convert to YOLO Format

**Manual Command:**
```bash
source env/bin/activate

python ls_processor.py \
  --ls-json /path/to/your/labelstudio-export.json \
  --images-dir /path/to/your/slip_images \
  --output-dir dataset \
  --classes company_name address tax_office tax_number date time category tax_amount total_amount payment_method merchant_number receipt_number currency
```

**Using the provided script:**
```bash
# Update paths in run_dataset_creation.sh first, then:
./run_dataset_creation.sh
```

### Step 3: Train YOLO Model
```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8s.pt \
  epochs=50 \
  imgsz=640 \
  batch=16
```

## Output Structure
After running the converter, you'll get:
```
dataset/
├── images/          # All training images
├── labels/          # YOLO format annotations (.txt files)
└── data.yaml        # YOLO configuration file
```

## Script Details

### `ls_processor.py`
- Converts Label Studio JSON exports to YOLO format
- Handles bounding box coordinate conversion
- Creates proper directory structure
- Generates YOLO configuration file

### `classes.py`
- Contains the class definitions and mappings
- 13 classes for receipt field detection

### `run_dataset_creation.sh`
- Example script with the exact command structure
- Update paths as needed for your data

## Notes
- Make sure your Label Studio export includes bounding box annotations
- Images should be accessible at the specified `--images-dir` path
- The script will copy images to the output directory
- YOLO training requires GPU for reasonable performance 