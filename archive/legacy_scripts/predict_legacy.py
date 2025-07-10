#!/usr/bin/env python3
"""
Basit OCR Tahmin Scripti
YOLO ile alan tespiti + Tesseract ile metin çıkarımı
"""

import json
import sys
from myocr_lib.core import load_model, predict_on_image, extract_text_from_box

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <resim_yolu> [confidence]")
        print("Örnek: python predict.py fiş.jpg 0.25")
        return
    
    image_path = sys.argv[1]
    confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    
    print(f"📄 OCR Analizi: {image_path}")
    print(f"🎯 Confidence threshold: {confidence}")
    print("=" * 50)
    
    # Model yükle
    model = load_model()
    if model is None:
        print("❌ Model yüklenemedi!")
        return
    
    # Tahmin yap
    detections = predict_on_image(model, image_path, confidence)
    
    if not detections:
        print(f"❌ Hiç alan tespit edilemedi! (conf>={confidence})")
        print("💡 Daha düşük confidence deneyin: python predict.py", image_path, "0.1")
        return
    
    print(f"✅ {len(detections)} alan tespit edildi")
    
    # Metinleri çıkar
    extracted_data = {}
    
    for detection in detections:
        class_name = detection['class_name']
        confidence_score = detection['confidence']
        
        print(f"📍 {class_name} (güven: {confidence_score:.2f})")
        
        text = extract_text_from_box(image_path, detection['coordinates'])
        
        if class_name in extracted_data:
            if isinstance(extracted_data[class_name], list):
                extracted_data[class_name].append(text)
            else:
                extracted_data[class_name] = [extracted_data[class_name], text]
        else:
            extracted_data[class_name] = text
        
        print(f"   📝 Metin: '{text}'")
    
    # JSON sonuç
    result = {
        "image_path": image_path,
        "confidence_threshold": confidence,
        "detections_count": len(detections),
        "extracted_text": extracted_data,
        "raw_detections": detections
    }
    
    print("\n" + "=" * 50)
    print("📊 JSON SONUÇ:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result

if __name__ == "__main__":
    main() 