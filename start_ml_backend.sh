#!/bin/bash
# MyOCR Label Studio ML Backend Başlatma Scripti

echo "🚀 MyOCR Label Studio ML Backend Başlatılıyor..."

# Virtual environment'ı aktif et
source env/bin/activate

# ML Backend'i başlat
cd label_studio_ml
python server.py --host localhost --port 9090 --confidence 0.2

echo "✅ ML Backend başlatıldı: http://localhost:9090" 