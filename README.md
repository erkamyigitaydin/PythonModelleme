# Slip OCR & Field Detection with YOLO

This project uses YOLOv8 to detect specific fields on a receipt (like total amount, date, etc.) and then uses Tesseract OCR to extract the text from those detected fields.

## 🚀 Features

- **Field Detection:** Trains a YOLOv8 model to identify 13 different fields on a Turkish receipt.
- **OCR Extraction:** Extracts text from each detected field using Tesseract.
- **Label Studio Integration:** Includes a script to convert Label Studio JSON exports into YOLO-compatible datasets.
- **End-to-End Pipeline:** A complete pipeline from data annotation to prediction.

### 📋 Recognized Fields
The model is trained to detect the following 13 classes:
1. `company_name`
2. `address`
3. `tax_office`
4. `tax_number`
5. `date`
6. `time`
7. `category`
8. `tax_amount`
9. `total_amount`
10. `payment_method`
11. `merchant_number`
12. `receipt_number`
13. `currency`


## 🔧 Setup

### 1. Prerequisites
- Python 3.8+
- Tesseract OCR Engine
  - **macOS:** `brew install tesseract tesseract-lang`
  - **Ubuntu:** `sudo apt-get install tesseract-ocr tesseract-ocr-tur`

### 2. Installation
Clone the repository and install the required Python packages:
```bash
git clone <your-repo-url>
cd myocr
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## ⚙️ Usage

The project has three main workflows: creating a dataset, training the model, and running predictions.

### 1. Creating the Dataset (from Label Studio)

If you have annotations from Label Studio, you can convert them into a YOLO dataset.

1.  Export your data from Label Studio in **JSON** format.
2.  Place your original images in a directory.
3.  Run the `ls_processor.py` script:

```bash
python ls_processor.py \
  --ls-json /path/to/your/label-studio-export.json \
  --images-dir /path/to/your/images \
  --output-dir dataset \
  --classes company_name address tax_office tax_number date time category tax_amount total_amount payment_method merchant_number receipt_number currency
```
This will create a `dataset/` folder with `images`, `labels`, and a `data.yaml` file.

### 2. Training the Model

Once you have a dataset, you can train the YOLO model:
```bash
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8s.pt \
  epochs=50 \
  imgsz=640 \
  batch=16
```
The trained model weights will be saved in the `runs/detect/train/weights/` directory (e.g., `best.pt`).

### 3. Running Predictions

Use the `predict.py` script to run the full pipeline (YOLO detection + OCR extraction) on a new receipt image.

```bash
python predict.py /path/to/your/receipt.jpg
```

You can also specify a custom model:
```bash
python predict.py /path/to/your/receipt.jpg --model /path/to/your/best.pt
```

#### Example Output (JSON)
The script will output a JSON object with the detected fields and the extracted text.
```json
{
    "image_path": "slip.jpg",
    "detections_count": 1,
    "extracted_text": {
        "address": "TERMINAL NO: 123456.\n€)"
    },
    "raw_detections": [
        {
            "class_name": "address",
            "class_id": 1,
            "coordinates": [
                163.63,
                88.48,
                274.42,
                115.31
            ],
            "confidence": 0.4445
        }
    ]
}
```

## 📂 Project Structure
```
.
├── classes.py              # Defines the 13 classes for detection
├── core.py                 # Original Tesseract OCR helper functions
├── ls_processor.py         # Label Studio to YOLO dataset converter
├── predict.py              # Main script for prediction (YOLO + OCR)
├── requirements.txt        # Python dependencies
├── .gitignore              # Files to be ignored by Git
└── README.md               # This file
```
The `dataset/`, `runs/`, and `env/` directories will be created during the workflow but are excluded by `.gitignore`. 