#!/bin/bash
# MyOCR Web Application Başlatma Scripti

echo "🚀 MyOCR Web Application başlatılıyor..."

# Ana proje dizinine git
cd "$(dirname "$0")/.."

# Virtual environment'ı aktif et
source env/bin/activate

# Web uygulaması dizinine geri dön
cd myocr_web

# Gerekli klasörleri oluştur
mkdir -p static/uploads
mkdir -p static/css
mkdir -p static/js

# Web uygulamasını başlat
python web_app.py

echo "✅ Web uygulaması başlatıldı: http://localhost:5000" 