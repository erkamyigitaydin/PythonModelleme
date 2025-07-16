#!/usr/bin/env python3

import json
import sys
import os
from pathlib import Path
from typing import Optional
from myocr_lib.core import (
    load_model, 
    predict_with_text_extraction, 
    validate_prediction_with_ground_truth,
    batch_validate_with_ground_truth
)

def predict_single_image(image_path: str, confidence: float = 0.25, 
                        validate_with_ground_truth: bool = False):
    """Tek görüntü için enhanced prediction"""
    
    print(f"📄 Enhanced OCR Analizi: {image_path}")
    print(f"🎯 Confidence threshold: {confidence}")
    print("=" * 50)
    
    # Model yükle
    model = load_model()
    if model is None:
        print("❌ Model yüklenemedi!")
        return
    
    if validate_with_ground_truth:
        # Ground truth ile validation
        texts_dir = "enhanced_dataset/texts"
        
        if os.path.exists(texts_dir):
            print("🔍 Ground truth ile validation yapılıyor...")
            
            result = validate_prediction_with_ground_truth(
                model, image_path, texts_dir, confidence
            )
            
            if result['status'] == 'validated':
                print(f"\n📊 OCR Validation Sonuçları:")
                print(f"   Overall Accuracy: {result['overall_ocr_accuracy']:.3f}")
                print(f"   Validated Fields: {result['validated_fields']}/{result['ground_truth_fields']}")
                
                print(f"\n🔍 Detaylı Karşılaştırma:")
                for validation in result['validation_results']:
                    print(f"   📌 {validation['class_name']}:")
                    print(f"      Predicted: '{validation['predicted_text']}'")
                    print(f"      Ground Truth: '{validation['ground_truth_text']}'")
                    print(f"      Accuracy: {validation['ocr_accuracy']:.3f}")
                    print(f"      Detection Conf: {validation['detection_confidence']:.3f}")
                
            else:
                print("⚠️ Bu görüntü için ground truth bulunamadı")
                # Normal prediction yap
                detections = predict_with_text_extraction(model, image_path, confidence)
                display_predictions(detections)
        else:
            print("⚠️ Ground truth dizini bulunamadı, normal prediction yapılıyor")
            detections = predict_with_text_extraction(model, image_path, confidence)
            display_predictions(detections)
    else:
        # Normal prediction
        detections = predict_with_text_extraction(model, image_path, confidence)
        display_predictions(detections)

def display_predictions(detections):
    """Prediction sonuçlarını göster"""
    if not detections:
        print("❌ Hiç alan tespit edilmedi!")
        return
    
    print(f"\n🎯 Tespit Edilen Alanlar ({len(detections)}):")
    print("-" * 70)
    
    for i, detection in enumerate(detections, 1):
        coords = detection['coordinates']
        print(f"{i:2d}. {detection['class_name']:15s} "
              f"| Güven: {detection['confidence']:.3f} "
              f"| Konum: ({coords[0]:.0f},{coords[1]:.0f})-({coords[2]:.0f},{coords[3]:.0f})")
        
        if 'extracted_text' in detection:
            text = detection['extracted_text'][:50] + "..." if len(detection['extracted_text']) > 50 else detection['extracted_text']
            print(f"     📝 Text: '{text}'")
        print()

def batch_prediction_with_validation(images_dir: str, confidence: float = 0.25, 
                                   max_images: Optional[int] = None):
    """Batch prediction + validation"""
    
    print(f"📁 Batch Enhanced OCR Analizi: {images_dir}")
    print(f"🎯 Confidence threshold: {confidence}")
    if max_images:
        print(f"🔢 Max images: {max_images}")
    print("=" * 70)
    
    # Model yükle
    model = load_model()
    if model is None:
        print("❌ Model yüklenemedi!")
        return
    
    texts_dir = "enhanced_dataset/texts"
    
    if not os.path.exists(texts_dir):
        print("⚠️ Ground truth dizini bulunamadı")
        return
    
    # Batch validation
    print("🔍 Batch validation başlıyor...")
    
    result = batch_validate_with_ground_truth(
        model, images_dir, texts_dir, confidence, max_images
    )
    
    stats = result['statistics']
    
    print(f"\n📊 Batch Validation Sonuçları:")
    print(f"   Total Images: {stats['total_images']}")
    print(f"   Validated Images: {stats['validated_images']}")
    print(f"   Total Detections: {stats['total_detections']}")
    
    if stats['validated_images'] > 0:
        print(f"   Average OCR Accuracy: {stats['average_accuracy']:.3f}")
        
        print(f"\n📈 Class-wise Accuracies:")
        for class_name, accuracy_data in stats['class_accuracies'].items():
            print(f"   {class_name:15s} | Avg: {accuracy_data['average']:.3f} "
                  f"| Count: {accuracy_data['count']:3d} "
                  f"| Range: {accuracy_data['min']:.3f}-{accuracy_data['max']:.3f}")
    
    # Detaylı sonuçları kaydet
    output_file = "batch_validation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detaylı sonuçlar kaydedildi: {output_file}")

def main():
    """Ana fonksiyon"""
    if len(sys.argv) < 2:
        print("Kullanım:")
        print("  python predict.py <resim_yolu> [confidence] [--validate] [--json-only]")
        print("  python predict.py --batch <images_dir> [confidence] [max_images]")
        print("")
        print("Örnekler:")
        print("  python predict.py fiş.jpg 0.25                    # Default prediction")
        print("  python predict.py fiş.jpg 0.25 --validate         # With ground truth validation")
        print("  python predict.py fiş.jpg 0.25 --json-only        # Only JSON output")
        print("  python predict.py --batch enhanced_dataset/images 0.25 10  # Batch processing")
        return
    
    if sys.argv[1] == "--batch":
        # Batch mode
        if len(sys.argv) < 3:
            print("❌ Batch mode için images dizini gerekli!")
            return
        
        images_dir = sys.argv[2]
        confidence = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25
        max_images = int(sys.argv[4]) if len(sys.argv) > 4 else None
        
        batch_prediction_with_validation(images_dir, confidence, max_images)
        
    else:
        # Single image mode
        image_path = sys.argv[1]
        confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
        validate = "--validate" in sys.argv
        
        predict_single_image(image_path, confidence, validate)

if __name__ == "__main__":
    main() 
