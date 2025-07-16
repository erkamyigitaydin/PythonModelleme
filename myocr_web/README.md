# MyOCR Web Application

MyOCR sistemi için modern ve kullanıcı dostu web arayüzü. Fiş ve fatura görüntülerini kolayca analiz etmek için tasarlanmıştır.

## 🌟 Özellikler

- 🖱️ **Drag & Drop**: Dosyaları sürükle-bırak ile yükle
- 🎛️ **Canlı Ayarlar**: Confidence threshold'u gerçek zamanlı ayarla  
- 🎯 **Görsel Analiz**: Tespit edilen alanları görüntü üzerinde göster
- 📊 **Detaylı Sonuçlar**: Çıkarılan metinler ve güven skorları
- 📱 **Responsive**: Mobil ve tablet uyumlu tasarım
- 🔄 **Gerçek Zamanlı**: Anında sonuç güncelleme

## 🚀 Hızlı Başlangıç

### 1. Kurulum

```bash
# Ana proje dizinindeki virtual environment'ı kullan
cd ../
source env/bin/activate
cd myocr_web

# Veya web uygulaması için ayrı environment oluştur
python -m venv web_env
source web_env/bin/activate
pip install -r requirements.txt
```

### 2. Başlatma

```bash
# Script ile (önerilen)
./start_web_app.sh

# Veya direkt Python ile
python web_app.py
```

### 3. Kullanım

1. Tarayıcıda `http://localhost:5000` adresine git
2. Fiş/fatura görüntüsünü yükle
3. Confidence threshold'u ayarla (isteğe bağlı)
4. "Analiz Et" butonuna bas
5. Sonuçları incele

## 📁 Dosya Yapısı

```
myocr_web/
├── web_app.py              # Flask uygulaması
├── templates/              # HTML şablonları
│   └── index.html         # Ana sayfa
├── static/                # Statik dosyalar
│   ├── uploads/           # Yüklenen dosyalar
│   ├── css/              # CSS dosyaları
│   └── js/               # JavaScript dosyaları
├── requirements.txt       # Python bağımlılıkları
├── start_web_app.sh      # Başlatma scripti
└── README.md             # Bu dosya
```

## 🔧 Konfigürasyon

### Port Değiştirme
```python
# web_app.py dosyasının sonunda
app.run(host='0.0.0.0', port=5001, debug=True)  # Port 5001
```

### Upload Limiti
```python
# web_app.py dosyasında
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB
```

### Desteklenen Formatlar
```python
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
```

## 🎨 Arayüz Özelleştirme

Web arayüzü `templates/index.html` dosyasında CSS variables ile özelleştirilebilir:

```css
:root {
    --primary-color: #3498db;    /* Ana renk */
    --secondary-color: #2c3e50;  /* İkincil renk */
    --success-color: #27ae60;    /* Başarı rengi */
    --danger-color: #e74c3c;     /* Hata rengi */
}
```

## 🔍 API Endpoints

### POST /upload
Dosya yükleme ve analiz
```json
{
    "file": "dosya",
    "confidence": "0.25"
}
```

### GET /health
Sistem durumu kontrolü
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2025-01-01T12:00:00"
}
```

### GET /uploads/<filename>
Yüklenen dosyaları serve et

## 📊 Örnek Analiz Sonucu

```json
{
    "success": true,
    "detections": [
        {
            "class_name": "company_name",
            "class_name_tr": "Şirket Adı",
            "confidence": 0.856,
            "coordinates": [123, 45, 456, 78],
            "extracted_text": "ABC Market Ltd. Şti."
        },
        {
            "class_name": "total_amount",
            "class_name_tr": "Toplam Tutar", 
            "confidence": 0.923,
            "coordinates": [789, 123, 890, 156],
            "extracted_text": "125,50 TL"
        }
    ],
    "summary": {
        "total_detections": 2,
        "avg_confidence": 0.889,
        "detected_fields": ["Şirket Adı", "Toplam Tutar"]
    }
}
```

## 🐛 Sorun Giderme

### Model Yükleme Hatası
```bash
❌ Model yükleme hatası: No such file or directory
```
**Çözüm**: Ana proje dizininde model dosyasının olduğundan emin olun:
```bash
ls ../runs/detect/train/weights/best.pt
```

### Port Zaten Kullanımda
```bash
❌ Port 5000 is already in use
```
**Çözüm**: Farklı port kullanın veya mevcut servisi durdurun:
```bash
lsof -ti:5000 | xargs kill -9
```

### Upload Hatası
```bash
❌ Desteklenmeyen dosya formatı
```
**Çözüm**: Desteklenen format kullanın (JPG, PNG, JPEG, GIF, BMP, TIFF)

## 🔒 Güvenlik

- ✅ Dosya uzantısı kontrolü
- ✅ Dosya boyutu limiti (16MB)
- ✅ Güvenli dosya adı oluşturma
- ✅ Temporary dosya temizleme
- ✅ Input validation

## 📈 Performance

- ⚡ Asenkron dosya işleme
- 🗂️ Automatic cleanup
- 💾 Base64 image encoding
- 🎯 Optimized detection rendering

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Push edin
5. Pull Request açın

## 📄 Lisans

Ana MyOCR projesi ile aynı lisans altında.

---

**Not**: Bu web uygulaması ana MyOCR sisteminin bir parçasıdır. Ana sistem için `../README.md` dosyasına bakın. 