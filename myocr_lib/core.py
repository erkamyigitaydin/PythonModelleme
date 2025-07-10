import cv2
import pytesseract
from PIL import Image
from typing import List, Dict, Optional, Tuple
import json
import os
from pathlib import Path

# Optional ultralytics import - PyTorch may not be available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics/PyTorch not available. YOLO detection disabled.")

def load_model(model_path='runs/detect/train/weights/best.pt'):
    """Model yükle"""
    if not YOLO_AVAILABLE:
        print("YOLO model loading skipped - ultralytics not available")
        return None
        
    try:
        model = YOLO(model_path)
        print(f"Model yüklendi: {model_path}")
        return model
    except Exception as e:
        print(f"Model yüklenemedi: {e}")
        return None

def predict_on_image(model, image_path, confidence=0.25):
    """Görüntü üzerinde YOLO tahmin yap"""
    if model is None:
        return []
    
    results = model(image_path, conf=confidence, verbose=False)
    detections = []
    
    if results and results[0].boxes is not None:
        print(f"📊 {len(results[0].boxes)} kutu tespit edildi (conf>={confidence})")
        
        for box in results[0].boxes:
            from .classes import id_to_class
            
            class_id = int(box.cls[0])
            class_name = id_to_class.get(class_id, "bilinmeyen")
            coords = box.xyxy[0].tolist()
            confidence_score = float(box.conf[0])
            
            detections.append({
                "class_name": class_name,
                "class_id": class_id,
                "coordinates": coords,
                "confidence": confidence_score
            })
    else:
        print(f"❌ Hiç kutu tespit edilmedi (conf>={confidence})")
    
    return detections

def extract_text_from_box(image_path: str, box_coords: List[float], ocr_config: str = "") -> str:
    """Kutu koordinatlarından metin çıkar"""
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
    
    if image is None:
        return ""
    
    x1, y1, x2, y2 = map(int, box_coords)
    padding = 5
    h, w = image.shape[:2]
    
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)

    cropped_image = image[y1_pad:y2_pad, x1_pad:x2_pad]

    if cropped_image.size == 0:
        return ""

    try:
        pil_img = Image.fromarray(cropped_image)
        config = ocr_config if ocr_config.strip() else r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(pil_img, lang="tur+eng", config=config)
        return text.strip()
    except Exception as e:
        print(f"OCR hatası: {e}")
        return ""

def predict_with_text_extraction(model, image_path: str, confidence: float = 0.25) -> List[Dict]:
    """YOLO tahmin + OCR text extraction birleşik fonksiyon"""
    detections = predict_on_image(model, image_path, confidence)
    
    for detection in detections:
        try:
            text = extract_text_from_box(image_path, detection['coordinates'])
            detection['extracted_text'] = text
        except Exception as e:
            detection['extracted_text'] = ""
            print(f"Text extraction error for {detection['class_name']}: {e}")
    
    return detections

def load_text_ground_truth_for_image(image_name: str, texts_dir: str) -> Dict[str, str]:
    """Belirli bir görüntü için text ground truth yükle"""
    json_file = os.path.join(texts_dir, f"{image_name}.json")
    
    if not os.path.exists(json_file):
        return {}
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # UUID'li key'leri temizle
        cleaned_data = {}
        for key, value in data.items():
            if '_' in key:
                class_name = key.rsplit('_', 1)[0]
                cleaned_data[class_name] = value
            else:
                cleaned_data[key] = value
        
        return cleaned_data
        
    except Exception as e:
        print(f"Text ground truth yüklenemedi {json_file}: {e}")
        return {}

def calculate_ocr_accuracy(predicted_text: str, ground_truth_text: str) -> float:
    """OCR accuracy hesapla"""
    pred = predicted_text.strip().lower()
    truth = ground_truth_text.strip().lower()
    
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    
    # Character-level similarity
    max_len = max(len(pred), len(truth))
    if max_len == 0:
        return 1.0
    
    matches = sum(1 for a, b in zip(pred, truth) if a == b)
    similarity = matches / max_len
    
    # Exact match bonus
    if pred == truth:
        return 1.0
    elif pred in truth or truth in pred:
        return max(similarity, 0.8)
    
    return similarity

def validate_prediction_with_ground_truth(model, image_path: str, texts_dir: str, 
                                        confidence: float = 0.25) -> Dict:
    """Model prediction'ını ground truth ile validate et"""
    
    # Image name extract et
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Ground truth yükle
    ground_truth = load_text_ground_truth_for_image(image_name, texts_dir)
    
    if not ground_truth:
        return {
            'status': 'no_ground_truth',
            'detections': predict_with_text_extraction(model, image_path, confidence)
        }
    
    # Prediction + text extraction
    detections = predict_with_text_extraction(model, image_path, confidence)
    
    # Accuracy hesapla
    validation_results = []
    total_accuracy = 0
    valid_comparisons = 0
    
    for detection in detections:
        class_name = detection['class_name']
        
        if class_name in ground_truth:
            predicted_text = detection['extracted_text']
            truth_text = ground_truth[class_name]
            
            accuracy = calculate_ocr_accuracy(predicted_text, truth_text)
            
            validation_results.append({
                'class_name': class_name,
                'predicted_text': predicted_text,
                'ground_truth_text': truth_text,
                'ocr_accuracy': accuracy,
                'detection_confidence': detection['confidence']
            })
            
            total_accuracy += accuracy
            valid_comparisons += 1
    
    overall_accuracy = total_accuracy / valid_comparisons if valid_comparisons > 0 else 0.0
    
    return {
        'status': 'validated',
        'overall_ocr_accuracy': overall_accuracy,
        'validation_results': validation_results,
        'detections': detections,
        'ground_truth_fields': len(ground_truth),
        'validated_fields': valid_comparisons
    }

def batch_validate_with_ground_truth(model, images_dir: str, texts_dir: str, 
                                    confidence: float = 0.25, max_images: Optional[int] = None) -> Dict:
    """Batch olarak multiple image'ları validate et"""
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(images_dir).glob(ext))
    
    if max_images is not None:
        image_files = image_files[:max_images]
    
    all_results = []
    overall_stats = {
        'total_images': len(image_files),
        'validated_images': 0,
        'total_detections': 0,
        'total_accuracy': 0,
        'class_accuracies': {}
    }
    
    for image_file in image_files:
        try:
            result = validate_prediction_with_ground_truth(
                model, str(image_file), texts_dir, confidence
            )
            
            if result['status'] == 'validated':
                all_results.append(result)
                overall_stats['validated_images'] += 1
                overall_stats['total_detections'] += len(result['validation_results'])
                overall_stats['total_accuracy'] += result['overall_ocr_accuracy']
                
                # Class-wise accuracy tracking
                for validation in result['validation_results']:
                    class_name = validation['class_name']
                    if class_name not in overall_stats['class_accuracies']:
                        overall_stats['class_accuracies'][class_name] = []
                    overall_stats['class_accuracies'][class_name].append(validation['ocr_accuracy'])
                    
        except Exception as e:
            print(f"Validation error for {image_file}: {e}")
            continue
    
    # Final statistics
    if overall_stats['validated_images'] > 0:
        overall_stats['average_accuracy'] = overall_stats['total_accuracy'] / overall_stats['validated_images']
        
        # Class average accuracies
        for class_name, accuracies in overall_stats['class_accuracies'].items():
            overall_stats['class_accuracies'][class_name] = {
                'average': sum(accuracies) / len(accuracies),
                'count': len(accuracies),
                'min': min(accuracies),
                'max': max(accuracies)
            }
    
    return {
        'results': all_results,
        'statistics': overall_stats
    } 
