#!/usr/bin/env python3
"""
MyOCR YOLO Model Eğitim Scripti
50 fişlik veri ile yeniden eğitim
"""

import os
import sys
import logging
from pathlib import Path
from ultralytics import YOLO
import torch

# Logging kurulumu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Ana eğitim fonksiyonu"""
    
    # Dataset yolu
    dataset_path = "dataset/data.yaml"
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset bulunamadı: {dataset_path}")
        return
    
    logger.info("🚀 MyOCR Model Eğitimi Başlıyor...")
    logger.info(f"📁 Dataset: {dataset_path}")
    
    # GPU/CPU kontrolü
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"🔧 Cihaz: {device}")
    
    # Mevcut model varsa yükle (transfer learning)
    model_path = "runs/detect/train/weights/best.pt"
    
    if os.path.exists(model_path):
        logger.info(f"📂 Mevcut model yükleniyor: {model_path}")
        model = YOLO(model_path)
        training_type = "Fine-tuning"
    else:
        logger.info("🆕 YOLOv8n ile sıfırdan başlıyor")
        model = YOLO('yolov8n.pt')
        training_type = "Transfer Learning"
    
    logger.info(f"🎯 Eğitim tipi: {training_type}")
    
    # Eğitim parametreleri - 50 fişlik küçük dataset için optimize
    training_params = {
        'data': dataset_path,
        'epochs': 150,        # Küçük dataset için daha fazla epoch
        'imgsz': 640,         # Standard resolution
        'batch': 8,           # Küçük batch size
        'device': device,
        'workers': 4,
        'patience': 25,       # Early stopping patience
        'save': True,
        'save_period': 10,    # Her 10 epoch'ta kaydet
        'cache': True,        # Görüntüleri RAM'de cache et
        'project': 'runs/detect',
        'name': 'retrain_50_receipts',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,      # Cosine learning rate
        'close_mosaic': 10,  # Son 10 epoch'ta mosaic augmentation kapat
        'resume': False,
        'amp': True,         # Mixed precision training
        'fraction': 1.0,     # Tüm veriyi kullan
        'profile': False,
        'freeze': None,      # Hiçbir layer'ı freeze etme
        'lr0': 0.001,        # Düşük learning rate (küçük dataset)
        'lrf': 0.01,         # Final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,          # Box loss gain
        'cls': 0.5,          # Classification loss gain
        'dfl': 1.5,          # DFL loss gain
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,      # Hafif renk augmentation
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 10.0,     # Hafif rotasyon (fiş düz olmalı)
        'translate': 0.1,    # Hafif translate
        'scale': 0.2,        # Hafif scale
        'shear': 2.0,        # Hafif shear
        'perspective': 0.0,  # Perspective kapalı (fiş düz)
        'flipud': 0.0,       # Vertical flip kapalı
        'fliplr': 0.5,       # Horizontal flip
        'mosaic': 1.0,       # Mosaic augmentation
        'mixup': 0.0,        # Mixup kapalı (küçük dataset)
        'copy_paste': 0.0    # Copy-paste kapalı
    }
    
    logger.info("📋 Eğitim Parametreleri:")
    for key, value in training_params.items():
        logger.info(f"   {key}: {value}")
    
    try:
        logger.info("🏃‍♂️ Eğitim başlıyor...")
        
        # Model eğitimi
        results = model.train(**training_params)
        
        logger.info("✅ Eğitim tamamlandı!")
        logger.info(f"📊 Sonuçlar: {results}")
        
        # En iyi model yolunu bul
        try:
            # Type-safe model path extraction
            if results is not None and hasattr(results, 'save_dir') and results.save_dir is not None:
                best_model_path = results.save_dir / "weights" / "best.pt"
                last_model_path = results.save_dir / "weights" / "last.pt"
                
                logger.info(f"🏆 En iyi model: {best_model_path}")
                logger.info(f"📝 Son model: {last_model_path}")
                
                # Ana model dizinine kopyala
                import shutil
                if os.path.exists(best_model_path):
                    backup_path = "runs/detect/train/weights/best_backup.pt"
                    final_path = "runs/detect/train/weights/best.pt"
                    
                    # Backup oluştur
                    if os.path.exists(final_path):
                        shutil.copy2(final_path, backup_path)
                        logger.info(f"🔄 Backup oluşturuldu: {backup_path}")
                    
                    # Yeni modeli kopyala
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    shutil.copy2(best_model_path, final_path)
                    logger.info(f"📦 Model güncellendi: {final_path}")
            else:
                logger.warning("⚠️ Model save_dir bulunamadı, manuel model kopyalama gerekebilir")
                
        except Exception as e:
            logger.error(f"Model kopyalama hatası: {e}")
        
        # Validation skorları - type-safe access
        if results is not None and hasattr(results, 'results_dict') and results.results_dict is not None:
            try:
                metrics = results.results_dict
                logger.info("📊 Validation Metrikleri:")
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"   {key}: {value:.4f}")
            except Exception as e:
                logger.warning(f"⚠️ Validation metrikleri okunamadı: {e}")
        else:
            logger.info("📊 Validation metrikleri mevcut değil")
        
        logger.info("🎉 Eğitim başarıyla tamamlandı!")
        logger.info("💡 Model predict.py ve ML backend'de otomatik olarak güncellenecek")
        
    except Exception as e:
        logger.error(f"❌ Eğitim hatası: {e}")
        raise

if __name__ == "__main__":
    main() 