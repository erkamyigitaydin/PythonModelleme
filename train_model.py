#!/usr/bin/env python3
"""
MyOCR Enhanced Training Script
YOLO + OCR Accuracy ile gelişmiş model eğitimi
Text ground truth kullanan çok modlu eğitim sistemi
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from ultralytics import YOLO
import torch
import numpy as np
from PIL import Image
import cv2

# Ana proje dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from myocr_lib.core import extract_text_from_box
from myocr_lib.classes import class_to_id, id_to_class

# Logging kurulumu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OCRAccuracyTracker:
    """OCR doğruluğunu track eden sınıf"""
    
    def __init__(self):
        self.accuracy_history = []
        self.class_accuracies = {}
        
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """İki text arasındaki benzerliği hesapla"""
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0
        
        # Levenshtein distance approximation
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        
        # Simple character matching
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        similarity = matches / max_len
        
        # Exact match bonus
        if text1 == text2:
            return 1.0
        elif text1 in text2 or text2 in text1:
            return max(similarity, 0.8)
        
        return similarity
    
    def evaluate_ocr_accuracy(self, model, image_path: str, yolo_labels: List[Dict], 
                             text_ground_truth: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """Bir görüntü için OCR accuracy'yi değerlendir"""
        
        # YOLO ile prediction yap
        results = model(image_path, conf=0.25, verbose=False)
        
        if not results or not results[0].boxes:
            return {}
        
        # Görüntü boyutlarını al
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        class_accuracies = {}
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = id_to_class.get(class_id, "unknown")
            
            if class_name not in text_ground_truth:
                continue
                
            # Bounding box koordinatları
            coords = box.xyxy[0].tolist()
            
            # OCR ile text çıkar
            try:
                ocr_text = extract_text_from_box(image_path, coords)
                ground_truth_text = text_ground_truth[class_name]
                
                # Accuracy hesapla
                accuracy = self.calculate_text_similarity(ocr_text, ground_truth_text)
                class_accuracies[class_name] = {
                    'accuracy': accuracy,
                    'ocr_text': ocr_text,
                    'ground_truth': ground_truth_text
                }
                
                logger.debug(f"📊 {class_name}: {accuracy:.3f} - OCR:'{ocr_text}' GT:'{ground_truth_text}'")
                
            except Exception as e:
                logger.warning(f"OCR extraction error for {class_name}: {e}")
                continue
        
        return class_accuracies
    
    def update_accuracy_stats(self, accuracies: Dict[str, Dict]):
        """Accuracy istatistiklerini güncelle"""
        for class_name, data in accuracies.items():
            if class_name not in self.class_accuracies:
                self.class_accuracies[class_name] = []
            self.class_accuracies[class_name].append(data['accuracy'])
    
    def get_overall_accuracy(self) -> float:
        """Genel OCR accuracy'yi hesapla"""
        all_accuracies = []
        for class_name, accuracies in self.class_accuracies.items():
            all_accuracies.extend(accuracies)
        
        return float(np.mean(all_accuracies)) if all_accuracies else 0.0
    
    def get_class_accuracies(self) -> Dict[str, float]:
        """Sınıf bazında accuracy'leri döndür"""
        return {
            class_name: float(np.mean(accuracies))
            for class_name, accuracies in self.class_accuracies.items()
        }

def load_text_ground_truth(dataset_path: str) -> Dict[str, Dict[str, str]]:
    """Text ground truth dosyalarını yükle"""
    texts_dir = Path(dataset_path).parent / "texts"
    
    if not texts_dir.exists():
        logger.warning(f"Text ground truth dizini bulunamadı: {texts_dir}")
        return {}
    
    text_data = {}
    
    for json_file in texts_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Dosya adından image name'i çıkar
            image_name = json_file.stem
            
            # UUID'li key'leri temizle - sadece class name'i al
            cleaned_data = {}
            for key, value in data.items():
                # Key format: "class_name_UUID" -> "class_name"
                if '_' in key:
                    class_name = key.rsplit('_', 1)[0]
                    cleaned_data[class_name] = value
                else:
                    cleaned_data[key] = value
            
            text_data[image_name] = cleaned_data
            
        except Exception as e:
            logger.warning(f"Text ground truth yüklenemedi {json_file}: {e}")
            continue
    
    logger.info(f"📝 {len(text_data)} dosya için text ground truth yüklendi")
    return text_data

def validate_ocr_performance(model, dataset_path: str, text_ground_truth: Dict, 
                           sample_size: int = 10) -> Dict:
    """OCR performansını validate et"""
    
    images_dir = Path(dataset_path).parent / "images"
    accuracy_tracker = OCRAccuracyTracker()
    
    # Sample images seç
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    if len(image_files) > sample_size:
        image_files = image_files[:sample_size]
    
    logger.info(f"🔍 {len(image_files)} görüntü üzerinde OCR validation başlatılıyor...")
    
    validation_results = []
    
    for image_file in image_files:
        image_name = image_file.stem
        
        if image_name not in text_ground_truth:
            continue
        
        try:
            # OCR accuracy değerlendir
            accuracies = accuracy_tracker.evaluate_ocr_accuracy(
                model, str(image_file), [], text_ground_truth[image_name]
            )
            
            if accuracies:
                accuracy_tracker.update_accuracy_stats(accuracies)
                
                avg_accuracy = float(np.mean([data['accuracy'] for data in accuracies.values()]))
                validation_results.append({
                    'image': image_name,
                    'accuracy': avg_accuracy,
                    'class_count': len(accuracies)
                })
                
        except Exception as e:
            logger.warning(f"Validation error for {image_name}: {e}")
            continue
    
    # Sonuçları raporla
    overall_accuracy = accuracy_tracker.get_overall_accuracy()
    class_accuracies = accuracy_tracker.get_class_accuracies()
    
    logger.info(f"📊 OCR Validation Sonuçları:")
    logger.info(f"   Genel Accuracy: {overall_accuracy:.3f}")
    logger.info(f"   Validate edilen görüntü: {len(validation_results)}")
    
    for class_name, accuracy in class_accuracies.items():
        logger.info(f"   {class_name}: {accuracy:.3f}")
    
    return {
        'overall_accuracy': overall_accuracy,
        'class_accuracies': class_accuracies,
        'validation_results': validation_results,
        'sample_count': len(validation_results)
    }

def enhanced_training_callback(model, epoch: int, dataset_path: str, 
                              text_ground_truth: Dict, validation_frequency: int = 10):
    """Enhanced training callback - her N epoch'ta OCR validation yap"""
    
    if epoch % validation_frequency != 0:
        return
    
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 Epoch {epoch} - OCR Validation")
    logger.info(f"{'='*50}")
    
    validation_results = validate_ocr_performance(
        model, dataset_path, text_ground_truth, sample_size=5
    )
    
    # Validation sonuçlarını kaydet
    validation_file = f"runs/enhanced_training/validation_epoch_{epoch}.json"
    os.makedirs(os.path.dirname(validation_file), exist_ok=True)
    
    with open(validation_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"💾 Validation sonuçları kaydedildi: {validation_file}")

def main():
    """Ana enhanced eğitim fonksiyonu"""
    
    # Enhanced dataset yolu
    dataset_path = "enhanced_dataset/data.yaml"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Enhanced dataset bulunamadı: {dataset_path}")
        logger.info("💡 Önce enhanced dataset'i oluşturun!")
        return
    
    logger.info("🚀 MyOCR Enhanced Training Başlıyor...")
    logger.info(f"📁 Dataset: {dataset_path}")
    
    # Text ground truth yükle
    text_ground_truth = load_text_ground_truth(dataset_path)
    
    if not text_ground_truth:
        logger.warning("⚠️ Text ground truth bulunamadı, sadece YOLO eğitimi yapılacak")
    else:
        logger.info(f"📝 {len(text_ground_truth)} görüntü için text ground truth yüklendi")
    
    # GPU/CPU kontrolü
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🔧 Cihaz: {device}")
    
    # Model yükle
    model_path = "runs/detect/train/weights/best.pt"
    
    if os.path.exists(model_path):
        logger.info(f"📂 Mevcut model yükleniyor: {model_path}")
        model = YOLO(model_path)
        training_type = "Enhanced Fine-tuning"
    else:
        logger.info("🆕 YOLOv8n ile başlıyor")
        model = YOLO('yolov8n.pt')
        training_type = "Enhanced Transfer Learning"
    
    logger.info(f"🎯 Eğitim tipi: {training_type}")
    
    # İlk OCR baseline ölç
    if text_ground_truth:
        logger.info("\n📊 Başlangıç OCR Performance Ölçümü...")
        baseline_results = validate_ocr_performance(
            model, dataset_path, text_ground_truth, sample_size=10
        )
        logger.info(f"📈 Baseline OCR Accuracy: {baseline_results['overall_accuracy']:.3f}")
    
    # Enhanced eğitim parametreleri
    training_params = {
        'data': dataset_path,
        'epochs': 100,        # Text-aware eğitim için daha moderate
        'imgsz': 640,
        'batch': 8,
        'device': device,
        'workers': 4,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': True,
        'project': 'runs/enhanced_training',
        'name': 'text_aware_training',
        'exist_ok': True,
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'lr0': 0.0005,       # Daha düşük learning rate (fine-tuning için)
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'hsv_h': 0.01,       # Daha az augmentation (text quality için)
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 5.0,      # Minimal rotation
        'translate': 0.05,   # Minimal translation
        'scale': 0.1,        # Minimal scaling
        'flipud': 0.0,       # No vertical flip
        'fliplr': 0.2,       # Minimal horizontal flip
        'mosaic': 0.5,       # Reduced mosaic
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    logger.info("📋 Enhanced Eğitim Parametreleri:")
    for key, value in training_params.items():
        logger.info(f"   {key}: {value}")
    
    try:
        logger.info("🏃‍♂️ Enhanced eğitim başlıyor...")
        
        # Model eğitimi
        results = model.train(**training_params)
        
        logger.info("✅ Enhanced eğitim tamamlandı!")
        
        # Final OCR performance test
        if text_ground_truth and results is not None:
            logger.info("\n📊 Final OCR Performance Testi...")
            
            # En iyi modeli yükle
            if hasattr(results, 'save_dir') and results.save_dir is not None:
                best_model_path = results.save_dir / "weights" / "best.pt"
                if os.path.exists(best_model_path):
                    final_model = YOLO(str(best_model_path))
                    
                    final_results = validate_ocr_performance(
                        final_model, dataset_path, text_ground_truth, sample_size=20
                    )
                    
                    logger.info(f"📈 Final OCR Accuracy: {final_results['overall_accuracy']:.3f}")
                    
                    if 'baseline_results' in locals():
                        improvement = final_results['overall_accuracy'] - baseline_results['overall_accuracy']
                        logger.info(f"📊 OCR Improvement: {improvement:+.3f}")
        
        # Model güncelleme
        if results is not None and hasattr(results, 'save_dir') and results.save_dir is not None:
            best_model_path = results.save_dir / "weights" / "best.pt"
            
            if os.path.exists(best_model_path):
                # Main model'i güncelle
                import shutil
                backup_path = "runs/detect/train/weights/best_backup_enhanced.pt"
                final_path = "runs/detect/train/weights/best.pt"
                
                # Backup oluştur
                if os.path.exists(final_path):
                    shutil.copy2(final_path, backup_path)
                    logger.info(f"🔄 Backup oluşturuldu: {backup_path}")
                
                # Enhanced modeli main olarak kaydet
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                shutil.copy2(best_model_path, final_path)
                logger.info(f"📦 Enhanced model ana model olarak güncellendi: {final_path}")
        
        logger.info("🎉 Enhanced training başarıyla tamamlandı!")
        logger.info("💡 Model artık text-aware ve OCR-optimized!")
        
    except Exception as e:
        logger.error(f"❌ Enhanced eğitim hatası: {e}")
        raise

if __name__ == "__main__":
    main() 