#!/usr/bin/env python3
"""
MyOCR Web Application
Kullanıcı dostu web arayüzü ile fiş/fatura analizi
"""

import os
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import base64
import io

# Ana proje dizinini path'e ekle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# MyOCR kütüphanesini import et
from myocr_lib.core import load_model, predict_with_text_extraction

app = Flask(__name__)
app.config['SECRET_KEY'] = 'myocr-web-app-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Upload klasörü
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Klasörleri oluştur
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('templates', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global model instance
model = None

def allowed_file(filename):
    """Dosya uzantısı kontrolü"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_ocr_model():
    """OCR modelini yükle"""
    global model
    try:
        # Ana proje dizinindeki model yolu
        model_path = Path(__file__).parent.parent / "runs/detect/train/weights/best.pt"
        model = load_model(str(model_path))
        print("✅ OCR Model başarıyla yüklendi")
        return True
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        return False

def draw_detections_on_image(image_path, detections):
    """Tespit edilen alanları görüntü üzerine çiz"""
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Her tespit için çerçeve çiz
    for i, detection in enumerate(detections):
        coords = detection['coordinates']
        x1, y1, x2, y2 = map(int, coords)
        
        # Renk paleti (her sınıf için farklı renk)
        colors = [
            (255, 0, 0),    # Kırmızı
            (0, 255, 0),    # Yeşil
            (0, 0, 255),    # Mavi
            (255, 255, 0),  # Sarı
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Mor
            (255, 165, 0),  # Turuncu
            (128, 128, 128), # Gri
            (255, 192, 203), # Pembe
            (0, 128, 0),    # Koyu Yeşil
            (128, 0, 0),    # Koyu Kırmızı
            (0, 0, 128)     # Koyu Mavi
        ]
        
        color = colors[i % len(colors)]
        
        # Çerçeve çiz
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Label yazısı
        label = f"{i+1}. {detection['class_name']}"
        confidence = detection['confidence']
        
        # Text background
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        cv2.rectangle(image, (x1, y1 - text_height - 10), 
                     (x1 + text_width, y1), color, -1)
        
        # Text
        cv2.putText(image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Base64 encode et
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Dosya upload ve analiz"""
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Desteklenmeyen dosya formatı'}), 400
    
    if file and file.filename and allowed_file(file.filename):
        # Güvenli dosya adı oluştur
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Dosyayı kaydet
        file.save(file_path)
        
        try:
            # OCR analizi yap
            if model is None:
                return jsonify({'error': 'OCR modeli yüklenmedi'}), 500
            
            confidence = float(request.form.get('confidence', 0.25))
            detections = predict_with_text_extraction(model, file_path, confidence)
            
            # Sonuçları işle
            results = []
            total_confidence = 0
            
            for detection in detections:
                result = {
                    'class_name': detection['class_name'],
                    'class_name_tr': get_turkish_class_name(detection['class_name']),
                    'confidence': round(detection['confidence'], 3),
                    'coordinates': detection['coordinates'],
                    'extracted_text': detection.get('extracted_text', ''),
                }
                results.append(result)
                total_confidence += detection['confidence']
            
            avg_confidence = total_confidence / len(detections) if detections else 0
            
            # Görüntü üzerine tespit alanlarını çiz
            annotated_image = draw_detections_on_image(file_path, detections)
            
            # Başarılı yanıt
            response = {
                'success': True,
                'filename': unique_filename,
                'image_url': url_for('uploaded_file', filename=unique_filename),
                'annotated_image': annotated_image,
                'detections': results,
                'summary': {
                    'total_detections': len(detections),
                    'avg_confidence': round(avg_confidence, 3),
                    'detected_fields': [r['class_name_tr'] for r in results]
                }
            }
            
            return jsonify(response)
            
        except Exception as e:
            # Hata durumunda dosyayı sil
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': f'OCR analizi hatası: {str(e)}'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Upload edilmiş dosyaları serve et"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def get_turkish_class_name(class_name):
    """İngilizce sınıf adını Türkçe'ye çevir"""
    translations = {
        'company_name': 'Şirket Adı',
        'address': 'Adres',
        'tax_office': 'Vergi Dairesi',
        'tax_number': 'Vergi Numarası',
        'date': 'Tarih',
        'time': 'Saat',
        'category': 'Kategori',
        'tax_amount': 'Vergi Tutarı',
        'total_amount': 'Toplam Tutar',
        'payment_method': 'Ödeme Yöntemi',
        'merchant_number': 'İşyeri Numarası',
        'receipt_number': 'Fiş Numarası',
        'currency': 'Para Birimi'
    }
    return translations.get(class_name, class_name.title())

@app.route('/health')
def health():
    """Sistem durumu kontrolü"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 MyOCR Web Application başlatılıyor...")
    
    # Model yükle
    if load_ocr_model():
        print("✅ Sistem hazır!")
    else:
        print("⚠️ Model yüklenemedi, web app kısıtlı modda çalışacak")
    
    print("🌐 Web arayüzü: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True) 