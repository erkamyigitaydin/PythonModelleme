import cv2
import argparse
from ultralytics import YOLO
import json

# Sınıf isimlerini ve OCR fonksiyonunu YENİ PAKETTEN import et
from myocr_lib import id_to_class, extract_text_from_box

def predict_and_extract(image_path, model_path='runs/detect/train/weights/best.pt'):
    """
    Verilen görüntü üzerinde YOLO ile alan tespiti yapar ve
    her alan için OCR ile metin çıkarımı yapar.
    """
    # Modeli yükle
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Hata: Model yüklenemedi. Model yolu doğru mu?: {model_path}")
        print(f"Detay: {e}")
        return
        
    # Görüntüyü yükle
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Görüntü dosyası bulunamadı veya okunamadı: {image_path}")
    except Exception as e:
        print(e)
        return

    # YOLO ile tahmin yap
    results = model(image_path, verbose=False)
    
    # Tespit edilen her bir kutu için işlem yap
    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            # Sınıf ID'sini al
            class_id = int(box.cls[0])
            # Sınıf adını al
            class_name = id_to_class.get(class_id, "bilinmeyen_sinif")
            
            # Koordinatları al
            coords = box.xyxy[0].tolist()
            
            # Güven skorunu al
            confidence = float(box.conf[0])
            
            detections.append({
                "class_name": class_name,
                "class_id": class_id,
                "coordinates": coords,
                "confidence": confidence
            })

    # Tespit edilen alanlardan metinleri çıkar
    extracted_data = {}
    
    # Orijinal görüntüyü OCR için kullan
    original_image_for_ocr = cv2.imread(image_path)
    
    for detection in detections:
        # YENİ: core.py'deki gelişmiş fonksiyonu kullan
        text = extract_text_from_box(original_image_for_ocr, detection['coordinates'])
        
        # Aynı sınıftan birden fazla tespit varsa listeye ekle
        if detection['class_name'] in extracted_data:
            # Eğer mevcut değer bir liste değilse, listeye çevir
            if not isinstance(extracted_data[detection['class_name']], list):
                extracted_data[detection['class_name']] = [extracted_data[detection['class_name']]]
            extracted_data[detection['class_name']].append(text)
        else:
            extracted_data[detection['class_name']] = text


    # Sonuçları JSON formatında birleştir
    final_result = {
        "image_path": image_path,
        "detections_count": len(detections),
        "extracted_text": extracted_data,
        "raw_detections": detections,
    }

    return final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO ve OCR ile fiş bilgilerini çıkarma")
    parser.add_argument("image_path", help="İşlenecek fiş görüntüsünün yolu")
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt", help="Eğitilmiş YOLO modelinin yolu")
    args = parser.parse_args()

    # İşlemi başlat
    results_json = predict_and_extract(args.image_path, args.model)
    
    # Sonuçları düzgün formatlı JSON olarak yazdır
    if results_json:
        print(json.dumps(results_json, indent=4, ensure_ascii=False)) 