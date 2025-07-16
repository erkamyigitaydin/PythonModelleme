# MyOCR - Gelişmiş OCR Sistemi

Bu proje, fiş/fatura gibi belgelerdeki kritik bilgileri yüksek doğrulukla otomatik olarak çıkartmak için geliştirilmiştir. YOLO tabanlı bölge tespiti ile OCR sonuçlarının doğruluğu artırılmıştır. **YOLO object detection** ve **OCR text extraction** teknolojilerini birleştiren gelişmiş bir Optical Character Recognition (OCR) sistemidir. Özellikle Türkçe fiş, fatura ve belge analizi için optimize edilmiştir.

## 🎯 Özellikler

### Temel Özellikler
- **YOLO-tabanlı Object Detection**: Belgede önemli alanları otomatik tespit
- **OCR Text Extraction**: Tespit edilen alanlardan metin çıkarma
- **Label Studio Entegrasyonu**: Kolay veri etiketleme ve ML backend desteği
- **Ground Truth Validation**: OCR doğruluğunu ölçme ve raporlama
- **Batch Processing**: Toplu görüntü analizi
- **Enhanced Training**: OCR accuracy ile gelişmiş model eğitimi

### Tespit Edilen Alanlar
Sistem aşağıdaki 13 farklı alanı tespit edebilir:

```python
classes = [
    "company_name",      # Şirket adı
    "address",           # Adres
    "tax_office",        # Vergi dairesi
    "tax_number",        # Vergi numarası
    "date",              # Tarih
    "time",              # Saat
    "category",          # Kategori
    "tax_amount",        # Vergi tutarı
    "total_amount",      # Toplam tutar
    "payment_method",    # Ödeme yöntemi
    "merchant_number",   # İşyeri numarası
    "receipt_number",    # Fiş numarası
    "currency"           # Para birimi
]
```

## 🏗️ Sistem Mimarisi

```
MyOCR/
├── myocr_lib/          # Ana OCR kütüphanesi
├── label_studio_ml/    # Label Studio ML Backend
├── enhanced_dataset/   # Eğitim veri seti
├── runs/              # Model eğitim sonuçları
└── archive/           # Arşiv dosyalar
```

### Bileşenler

#### 1. MyOCR Kütüphanesi (`myocr_lib/`)
- **core.py**: Ana OCR fonksiyonları (YOLO + OCR integration)
- **classes.py**: Sınıf tanımları ve mapping'ler
- **__init__.py**: Kütüphane API'sı

#### 2. Label Studio ML Backend (`label_studio_ml/`)
- **model.py**: ML model sınıfı ve tahmin logics
- **server.py**: Flask-tabanlı ML backend server
- **runs/**: Model training sonuçları

#### 3. Ana Scriptler
- **train_model.py**: OCR accuracy ile gelişmiş model eğitimi
- **predict.py**: Görüntü analizi ve doğrulama
- **ls_processor.py**: Label Studio'dan YOLO formatına dönüştürme
- **test_ml_backend.py**: ML backend test suite

## 📦 Kurulum

### Gereksinimler
- Python 3.8+
- CUDA destekli GPU (önerilen)
- 8GB+ RAM

### 1. Projeyi Klonlayın
```bash
git clone <repository-url>
cd myocr
```

### 2. Virtual Environment Oluşturun
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# veya
env\Scripts\activate  # Windows
```

### 3. Bağımlılıkları Kurun
```bash
pip install -r requirements.txt
```

### 4. Tesseract OCR Kurun
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-tur

# macOS
brew install tesseract tesseract-lang

# Windows
# Tesseract-OCR installer'ı indirin ve kurun
```

## 🚀 Kullanım

### 1. Web Arayüzü (Önerilen)

En kolay kullanım şekli web arayüzüdür:

```bash
# Web uygulamasını başlat
cd myocr_web
./start_web_app.sh

# Tarayıcıda açın: http://localhost:5000
```

**Web Arayüzü Özellikleri:**
- 🖱️ Drag & drop ile dosya yükleme
- 🎛️ Canlı confidence threshold ayarı
- 🎯 Görsel üzerinde tespit edilen alanları gösterme
- 📊 Detaylı analiz sonuçları ve istatistikler
- 📱 Responsive tasarım (mobil uyumlu)
- 🔄 Gerçek zamanlı sonuç güncelleme

### 2. Komut Satırı Kullanımı

```bash
# Basit analiz
python predict.py resim.jpg

# Confidence threshold ile
python predict.py resim.jpg 0.25

# Ground truth validation ile
python predict.py resim.jpg 0.25 --validate
```

**Örnek Çıktı:**
```
📄 Enhanced OCR Analizi: resim.jpg
🎯 Confidence threshold: 0.25
==================================================

🎯 Tespit Edilen Alanlar (5):
----------------------------------------------------------------------
 1. company_name    | Güven: 0.856 | Konum: (123,45)-(456,78)
     📝 Text: 'ABC Market Ltd. Şti.'

 2. total_amount    | Güven: 0.923 | Konum: (789,123)-(890,156)
     📝 Text: '125,50 TL'

 3. date           | Güven: 0.745 | Konum: (234,567)-(345,590)
     📝 Text: '03.07.2025'
```

### 3. Batch Processing

```bash
# Tüm klasörü analiz et
python predict.py --batch enhanced_dataset/images

# Maksimum 10 görüntü
python predict.py --batch enhanced_dataset/images 0.25 10
```

### 4. Model Eğitimi

```bash
# Enhanced training (OCR accuracy ile)
python train_model.py --dataset dataset/data.yaml --epochs 50

# Standard YOLO training
yolo detect train data=dataset/data.yaml model=yolov8s.pt epochs=50 imgsz=640 batch=16
```

### 5. Label Studio ML Backend

```bash
# ML Backend'i başlat
./start_ml_backend.sh

# Test et
python test_ml_backend.py
```

**ML Backend URL'leri:**
- Ana sayfa: `http://localhost:9090/`
- Health check: `http://localhost:9090/health`
- Prediction: `http://localhost:9090/predict`

## 🏷️ Veri Hazırlama

### Label Studio ile Veri Etiketleme

1. **Label Studio'yu Kurun:**
```bash
pip install label-studio
```

2. **Projeyi Başlatın:**
```bash
label-studio start
```

3. **ML Backend'i Bağlayın:**
   - Settings → Machine Learning
   - URL: `http://localhost:9090`

4. **Veriyi Dışa Aktarın:**
   - Export → JSON format

### YOLO Formatına Dönüştürme

```bash
python ls_processor.py \
  --ls-json exported_data.json \
  --images-dir /path/to/images \
  --output-dir dataset \
  --classes company_name address tax_office tax_number date time category tax_amount total_amount payment_method merchant_number receipt_number currency
```

## 📊 Performance ve Doğrulama

### OCR Accuracy Ölçümü

```python
from myocr_lib.core import validate_prediction_with_ground_truth

result = validate_prediction_with_ground_truth(
    model, 
    "image.jpg", 
    "enhanced_dataset/texts", 
    confidence=0.25
)

print(f"Overall OCR Accuracy: {result['overall_ocr_accuracy']:.3f}")
```

### Batch Validation

```python
from myocr_lib.core import batch_validate_with_ground_truth

results = batch_validate_with_ground_truth(
    model,
    "enhanced_dataset/images",
    "enhanced_dataset/texts",
    confidence=0.25,
    max_images=50
)
```

## 🛠️ API Kullanımı

### Python API

```python
from myocr_lib import load_model, predict_with_text_extraction

# Model yükle
model = load_model("runs/detect/train/weights/best.pt")

# Tahmin yap
detections = predict_with_text_extraction(model, "image.jpg", confidence=0.25)

for detection in detections:
    print(f"Sınıf: {detection['class_name']}")
    print(f"Güven: {detection['confidence']:.3f}")
    print(f"Metin: {detection['extracted_text']}")
    print(f"Koordinatlar: {detection['coordinates']}")
```

### REST API (Label Studio ML Backend)

```python
import requests

# Tahmin request'i
payload = {
    "tasks": [
        {
            "id": 1,
            "data": {
                "image": "/path/to/image.jpg"
            }
        }
    ]
}

response = requests.post("http://localhost:9090/predict", json=payload)
results = response.json()["results"]
```

## 📁 Proje Yapısı

```
myocr/
├── myocr_lib/                    # Ana OCR kütüphanesi
│   ├── __init__.py              # API exports
│   ├── core.py                  # OCR core functions
│   └── classes.py               # Sınıf tanımları
├── myocr_web/                   # Web Uygulaması (Ayrı Modül)
│   ├── web_app.py              # Flask uygulaması
│   ├── templates/              # HTML şablonları
│   │   └── index.html         # Ana sayfa
│   ├── static/                 # Statik dosyalar
│   │   ├── uploads/           # Yüklenen dosyalar
│   │   ├── css/               # CSS dosyaları
│   │   └── js/                # JavaScript dosyaları
│   ├── requirements.txt        # Web app bağımlılıkları
│   ├── start_web_app.sh       # Web uygulaması başlatma
│   └── README.md              # Web app dokümantasyonu
├── label_studio_ml/             # Label Studio ML Backend
│   ├── model.py                 # ML model sınıfı
│   ├── server.py                # Flask server
│   └── runs/                    # Training sonuçları
├── enhanced_dataset/            # Eğitim veri seti
│   ├── images/                  # Görüntü dosyaları
│   ├── labels/                  # YOLO label dosyaları
│   ├── texts/                   # Ground truth metin dosyaları
│   └── data.yaml               # YOLO dataset config
├── runs/                        # Model eğitim sonuçları
│   ├── detect/                  # YOLO detection training
│   └── enhanced_training/       # Enhanced training logs
├── train_model.py               # Model eğitim scripti
├── predict.py                   # Tahmin scripti
├── ls_processor.py              # Label Studio processor
├── test_ml_backend.py           # ML backend test
├── start_ml_backend.sh          # ML backend başlatma
├── run_dataset_creation.sh      # Dataset oluşturma
└── requirements.txt             # Ana proje bağımlılıkları
```

## 🔧 Konfigürasyon

### Environment Variables

```bash
# ML Backend confidence threshold
export CONFIDENCE_THRESHOLD=0.2

# AWS S3 credentials (opsiyonel)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
```

### Model Ayarları

```python
# myocr_lib/core.py dosyasında
CONFIDENCE_THRESHOLD = 0.25      # Tespit confidence eşiği
OCR_CONFIG = "--oem 3 --psm 6"   # Tesseract OCR konfigürasyonu
YOLO_MODEL_PATH = "runs/detect/train/weights/best.pt"
```

## 📈 Performance Optimizasyon

### GPU Kullanımı
```python
# CUDA kullanılabilirliğini kontrol et
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

### Batch Processing Optimizasyon
```python
# Parallel processing için
from concurrent.futures import ThreadPoolExecutor

def process_images_parallel(image_paths, model, confidence=0.25):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(
            lambda img: predict_with_text_extraction(model, img, confidence),
            image_paths
        ))
    return results
```

## 🧪 Test

### Unit Tests
```bash
# ML Backend test
python test_ml_backend.py

# OCR accuracy test
python -m pytest tests/ -v
```

### Manuel Test
```bash
# Web uygulaması test
cd myocr_web
./start_web_app.sh
# Tarayıcıda http://localhost:5000 açın ve fiş yükleyin

# Tek görüntü test
python predict.py test_images/sample.jpg 0.25 --validate

# Batch test
python predict.py --batch test_images/ 0.25 5
```

## 🐛 Troubleshooting

### Yaygın Sorunlar

1. **YOLO Model Yükleme Hatası**
```bash
Error: Model yüklenemedi
Solution: Model dosyasının varlığını kontrol edin: runs/detect/train/weights/best.pt
```

2. **Tesseract OCR Hatası**
```bash
Error: tesseract is not installed
Solution: Tesseract-OCR'ı yükleyin ve PATH'e ekleyin
```

3. **Label Studio ML Backend Bağlantı Hatası**
```bash
Error: Connection refused
Solution: ML Backend'in çalıştığını kontrol edin: ./start_ml_backend.sh
```

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📞 İletişim

Proje hakkında sorularınız için:
- GitHub Issues
- Email: [erkamyigitaydin@gmail.com]

## 🙏 Teşekkürler

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection framework
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - OCR engine
- [Label Studio](https://github.com/heartexlabs/label-studio) - Data labeling platform
- [OpenCV](https://opencv.org/) - Computer vision library

---

**Not:** Bu sistem sürekli geliştirilmektedir. En son güncellemeler için repository'yi takip edin. 