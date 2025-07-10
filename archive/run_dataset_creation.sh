#!/bin/bash

# Activate virtual environment
source env/bin/activate

# Run the Label Studio processor with the specified classes
python ls_processor.py \
  --ls-json project-2-at-2025-07-03-12-45-1e153692.json \
  --images-dir /Users/erkamaydin/Desktop/sliptr \
  --output-dir dataset \
  --classes company_name address tax_office tax_number date time category tax_amount total_amount payment_method merchant_number receipt_number currency

echo "Dataset creation completed!"
echo "Dataset structure created in 'dataset/' directory"
echo ""
echo "To train YOLO model, run:"
echo "yolo detect train data=dataset/data.yaml model=models/yolov8s.pt epochs=50 imgsz=640 batch=16" 