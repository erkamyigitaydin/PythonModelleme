import os
import sys
import logging
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

# Ana proje dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myocr_lib.core import load_model, predict_on_image

logger = logging.getLogger(__name__)

class MyOCRModel:
    """MyOCR sistemi için Label Studio ML Backend"""
    
    def __init__(self, **kwargs):
        # Model yolu
        self.model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "runs/detect/train/weights/best.pt"
        )
        
        # YOLO modelini yükle
        logger.info(f"Model yükleniyor: {self.model_path}")
        self.model = load_model(self.model_path)
        
        # Confidence threshold
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.2'))
        
        # S3 client (opsiyonel)
        try:
            self.s3_client = boto3.client('s3')
            logger.info("S3 client başarıyla oluşturuldu")
        except (NoCredentialsError, Exception) as e:
            logger.warning(f"S3 client oluşturulamadı: {e}")
            self.s3_client = None
        
        # Sınıf isimleri
        self.class_names = [
            'company_name', 'address', 'tax_office', 'tax_number', 
            'date', 'time', 'category', 'tax_amount', 'total_amount', 
            'payment_method', 'merchant_number', 'receipt_number', 'currency'
        ]
        
        logger.info(f"Model başarıyla yüklendi. Confidence threshold: {self.confidence_threshold}")
    
    def predict(self, tasks: List[Dict], **kwargs) -> List[Dict]:
        """Label Studio'dan gelen görevleri işle ve tahminleri döndür"""
        predictions = []
        
        for task in tasks:
            try:
                # Görüntü URL'ini al
                image_url = task['data'].get('ocr') or task['data'].get('image')
                if not image_url:
                    logger.warning("Görevde image bulunamadı")
                    predictions.append({
                        'result': [],
                        'score': 0.0
                    })
                    continue
                
                # Görüntüyü işle ve tahmin yap
                prediction = self._predict_single_image(image_url, task)
                predictions.append(prediction)
                
            except Exception as e:
                logger.error(f"Tahmin hatası: {e}")
                predictions.append({
                    'result': [],
                    'score': 0.0
                })
        
        return predictions
    
    def _download_s3_image(self, s3_url: str, task_id: int) -> str:
        """S3'ten görüntüyü indir ve geçici dosya yolunu döndür"""
        if not self.s3_client:
            raise Exception("S3 client mevcut değil")
        
        # S3 URL'yi parse et: s3://bucket/key
        s3_url = s3_url.replace('s3://', '')
        bucket, key = s3_url.split('/', 1)
        
        # Geçici dosya yolu
        temp_path = f"/tmp/s3_image_{task_id}_{os.path.basename(key)}"
        
        try:
            # S3'ten dosyayı indir
            self.s3_client.download_file(bucket, key, temp_path)
            logger.info(f"S3 görüntüsü indirildi: {s3_url} -> {temp_path}")
            return temp_path
        except ClientError as e:
            logger.error(f"S3 download hatası: {e}")
            raise
    
    def _predict_single_image(self, image_url: str, task: Dict) -> Dict:
        """Tek bir görüntü için tahmin yap"""
        temp_path = None
        try:
            # URL tipine göre görüntüyü al
            if image_url.startswith('s3://'):
                # S3 URL
                if not self.s3_client:
                    logger.error("S3 URL algılandı ancak S3 client mevcut değil")
                    return {'result': [], 'score': 0.0}
                
                temp_path = self._download_s3_image(image_url, task['id'])
                
            elif image_url.startswith('/data/'):
                # Label Studio'dan gelen yol formatı
                temp_path = image_url.replace('/data/', 'dataset/')
                
            elif image_url.startswith('http'):
                # HTTP URL
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
                temp_path = f"/tmp/temp_image_{task['id']}.jpg"
                image.save(temp_path)
                
            else:
                # Yerel dosya yolu
                temp_path = image_url
            
            # Dosyanın varlığını kontrol et
            if not os.path.exists(temp_path):
                logger.error(f"Görüntü dosyası bulunamadı: {temp_path}")
                return {'result': [], 'score': 0.0}
            
            # YOLO ile tahmin yap
            detections = predict_on_image(self.model, temp_path, self.confidence_threshold)
            
            if not detections:
                logger.info("Hiç tespit edilmedi")
                return {'result': [], 'score': 0.0}
            
            # Görüntü boyutlarını al
            img = Image.open(temp_path)
            img_width, img_height = img.size
            
            # Label Studio formatına çevir
            result = []
            total_score = 0
            
            for detection in detections:
                # Koordinatları normalize et (0-100 arası)
                x1, y1, x2, y2 = detection['coordinates']
                
                # Koordinatları yüzdeye çevir
                x = (x1 / img_width) * 100
                y = (y1 / img_height) * 100
                width = ((x2 - x1) / img_width) * 100
                height = ((y2 - y1) / img_height) * 100
                
                # Label Studio annotation formatı
                annotation = {
                    "from_name": "label",
                    "to_name": "image", 
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [detection['class_name']],
                        "x": x,
                        "y": y,
                        "width": width,
                        "height": height
                    },
                    "score": detection['confidence']
                }
                
                result.append(annotation)
                total_score += detection['confidence']
            
            avg_score = total_score / len(detections) if detections else 0.0
            
            return {
                'result': result,
                'score': avg_score
            }
            
        except Exception as e:
            logger.error(f"Görüntü işleme hatası: {e}")
            return {'result': [], 'score': 0.0}
        
        finally:
            # Geçici dosyaları temizle (sadece S3 ve HTTP için)
            if temp_path and (image_url.startswith('s3://') or image_url.startswith('http')) and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"Geçici dosya silindi: {temp_path}")
                except:
                    pass
    
    def fit(self, completions: List[Dict], **kwargs):
        """Model eğitimi için - Label Studio annotation'larıyla fine-tuning"""
        if not completions:
            logger.warning("Hiç annotation verisi gelmedi")
            return
        
        logger.info(f"🚀 Model eğitimi başlıyor: {len(completions)} annotation")
        
        try:
            # Annotation'ları YOLO formatına çevir
            training_data = self._prepare_training_data(completions)
            
            if not training_data:
                logger.warning("Geçerli eğitim verisi bulunamadı")
                return
            
            # İnkremental eğitim için dataset oluştur
            dataset_path = self._create_training_dataset(training_data)
            
            # YOLO fine-tuning
            self._fine_tune_model(dataset_path)
            
            logger.info("✅ Model eğitimi tamamlandı!")
            
        except Exception as e:
            logger.error(f"❌ Model eğitimi hatası: {e}")
    
    def _prepare_training_data(self, completions: List[Dict]) -> List[Dict]:
        """Label Studio annotation'larını eğitim formatına çevir"""
        training_data = []
        
        for completion in completions:
            try:
                # Task bilgilerini al
                task = completion.get('task', {})
                image_url = task.get('data', {}).get('ocr') or task.get('data', {}).get('image')
                
                if not image_url:
                    continue
                
                # Annotation sonuçlarını al
                annotations = completion.get('result', [])
                
                if not annotations:
                    continue
                
                # Annotations'ları ID'ye göre grupla (rectangle + text)
                regions_by_id = {}
                
                for ann in annotations:
                    ann_id = ann.get('id', '')
                    if ann_id not in regions_by_id:
                        regions_by_id[ann_id] = {}
                    
                    if ann.get('type') == 'rectanglelabels':
                        regions_by_id[ann_id]['rectangle'] = ann
                    elif ann.get('type') == 'textarea':
                        regions_by_id[ann_id]['transcription'] = ann
                
                # YOLO formatına çevir + text ground truth ekle
                yolo_annotations = []
                text_ground_truth = {}
                
                for region_id, region_data in regions_by_id.items():
                    rectangle = region_data.get('rectangle')
                    transcription = region_data.get('transcription')
                    
                    if not rectangle:
                        continue
                    
                    class_name = rectangle['value']['rectanglelabels'][0]
                    
                    # Sınıf ID'sini bul
                    class_id = self.class_names.index(class_name) if class_name in self.class_names else -1
                    
                    if class_id == -1:
                        continue
                    
                    # Bounding box koordinatları (Label Studio: 0-100%, YOLO: 0-1 normalized)
                    x = rectangle['value']['x'] / 100.0
                    y = rectangle['value']['y'] / 100.0 
                    width = rectangle['value']['width'] / 100.0
                    height = rectangle['value']['height'] / 100.0
                    
                    # YOLO center format'a çevir
                    x_center = x + width / 2
                    y_center = y + height / 2
                    
                    bbox_data = {
                        'class_id': class_id,
                        'class_name': class_name,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    }
                    
                    # Text transcription varsa ekle
                    if transcription:
                        manual_text = transcription['value']['text'][0] if transcription['value']['text'] else ""
                        bbox_data['ground_truth_text'] = manual_text.strip()
                        text_ground_truth[class_name] = manual_text.strip()
                        logger.info(f"📝 {class_name}: '{manual_text.strip()}'")
                    
                    yolo_annotations.append(bbox_data)
                
                if yolo_annotations:
                    training_data.append({
                        'image_url': image_url,
                        'annotations': yolo_annotations,
                        'text_ground_truth': text_ground_truth
                    })
                    
            except Exception as e:
                logger.error(f"Annotation işleme hatası: {e}")
                continue
        
        logger.info(f"📊 {len(training_data)} görüntü eğitim için hazırlandı")
        return training_data
    
    def _create_training_dataset(self, training_data: List[Dict]) -> str:
        """Eğitim dataset'i oluştur"""
        import os
        import tempfile
        from pathlib import Path
        
        # Geçici dataset dizini
        dataset_dir = Path(tempfile.mkdtemp(prefix="ls_training_"))
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        text_dir = dataset_dir / "text_ground_truth"
        
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        text_dir.mkdir(exist_ok=True)
        
        image_count = 0
        ocr_accuracy_stats = []
        
        for data in training_data:
            try:
                image_url = data['image_url']
                annotations = data['annotations']
                text_ground_truth = data.get('text_ground_truth', {})
                
                # Görüntüyü indir/kopyala
                if image_url.startswith('s3://'):
                    # S3'ten indir
                    temp_path = self._download_s3_image(image_url, image_count)
                    image_filename = f"image_{image_count}.jpg"
                    final_image_path = images_dir / image_filename
                    
                    import shutil
                    shutil.move(temp_path, final_image_path)
                    
                elif os.path.exists(image_url):
                    # Yerel dosya kopyala
                    image_filename = f"image_{image_count}.jpg"
                    final_image_path = images_dir / image_filename
                    
                    import shutil
                    shutil.copy2(image_url, final_image_path)
                else:
                    logger.warning(f"Görüntü bulunamadı: {image_url}")
                    continue
                
                # YOLO label dosyası oluştur
                label_filename = f"image_{image_count}.txt"
                label_path = labels_dir / label_filename
                
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        f.write(f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n")
                
                # Text ground truth dosyası oluştur (varsa)
                if text_ground_truth:
                    text_filename = f"image_{image_count}.json"
                    text_path = text_dir / text_filename
                    
                    import json
                    with open(text_path, 'w', encoding='utf-8') as f:
                        json.dump(text_ground_truth, f, ensure_ascii=False, indent=2)
                    
                    # OCR accuracy ölçümü yap
                    accuracy_stats = self._measure_ocr_accuracy(str(final_image_path), annotations, text_ground_truth)
                    if accuracy_stats:
                        ocr_accuracy_stats.extend(accuracy_stats)
                
                image_count += 1
                
            except Exception as e:
                logger.error(f"Dataset oluşturma hatası: {e}")
                continue
        
        # OCR accuracy raporunu kaydet
        if ocr_accuracy_stats:
            self._save_ocr_accuracy_report(dataset_dir, ocr_accuracy_stats)
        
        # data.yaml oluştur
        yaml_content = f"""
path: {dataset_dir}
train: images
val: images  # Küçük dataset için train=val
nc: {len(self.class_names)}
names: {self.class_names}
"""
        
        yaml_path = dataset_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        
        logger.info(f"📁 Dataset oluşturuldu: {dataset_dir} ({image_count} görüntü)")
        logger.info(f"📝 Text ground truth: {len([d for d in training_data if d.get('text_ground_truth')])} görüntü")
        
        return str(yaml_path)
    
    def _measure_ocr_accuracy(self, image_path: str, annotations: List[Dict], ground_truth: Dict[str, str]) -> List[Dict]:
        """OCR doğruluğunu ölç - manuel text vs otomatik OCR"""
        accuracy_stats = []
        
        try:
            from myocr_lib.core import extract_text_from_box
            
            for ann in annotations:
                class_name = ann.get('class_name')
                if not class_name or class_name not in ground_truth:
                    continue
                
                manual_text = ground_truth[class_name]
                if not manual_text.strip():
                    continue
                
                # Bounding box koordinatlarını pixel'e çevir
                img = Image.open(image_path)
                img_width, img_height = img.size
                
                x_center = ann['x_center'] * img_width
                y_center = ann['y_center'] * img_height
                width = ann['width'] * img_width
                height = ann['height'] * img_height
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # OCR ile text çıkar
                ocr_text = extract_text_from_box(image_path, [x1, y1, x2, y2])
                
                # Accuracy hesapla
                accuracy = self._calculate_text_similarity(manual_text, ocr_text)
                
                accuracy_stats.append({
                    'class_name': class_name,
                    'manual_text': manual_text,
                    'ocr_text': ocr_text,
                    'accuracy': accuracy,
                    'image_path': image_path
                })
                
                logger.info(f"📊 {class_name} OCR Accuracy: {accuracy:.1%}")
                logger.info(f"   Manual: '{manual_text}'")
                logger.info(f"   OCR: '{ocr_text}'")
                
        except Exception as e:
            logger.error(f"OCR accuracy ölçüm hatası: {e}")
        
        return accuracy_stats
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """İki text arasındaki benzerliği hesapla (0-1 arası)"""
        try:
            # Basit character-based similarity
            text1 = text1.strip().lower()
            text2 = text2.strip().lower()
            
            if not text1 and not text2:
                return 1.0
            if not text1 or not text2:
                return 0.0
            
            # Levenshtein distance yaklaşımı
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
            
            # Basit karakter karşılaştırması
            matches = sum(1 for a, b in zip(text1, text2) if a == b)
            similarity = matches / max_len
            
            # Daha sofistike: tam eşleşme bonus
            if text1 == text2:
                similarity = 1.0
            elif text1 in text2 or text2 in text1:
                similarity = max(similarity, 0.8)
            
            return similarity
            
        except Exception:
            return 0.0
    
    def _save_ocr_accuracy_report(self, dataset_dir: Path, accuracy_stats: List[Dict]):
        """OCR accuracy raporunu kaydet"""
        try:
            import json
            
            # Özet istatistikler
            if accuracy_stats:
                total_accuracy = sum(stat['accuracy'] for stat in accuracy_stats) / len(accuracy_stats)
                class_accuracies = {}
                
                for stat in accuracy_stats:
                    class_name = stat['class_name']
                    if class_name not in class_accuracies:
                        class_accuracies[class_name] = []
                    class_accuracies[class_name].append(stat['accuracy'])
                
                # Sınıf bazında ortalama
                class_avg_accuracies = {
                    class_name: sum(accuracies) / len(accuracies)
                    for class_name, accuracies in class_accuracies.items()
                }
                
                report = {
                    'overall_accuracy': total_accuracy,
                    'class_accuracies': class_avg_accuracies,
                    'total_samples': len(accuracy_stats),
                    'detailed_results': accuracy_stats
                }
                
                report_path = dataset_dir / "ocr_accuracy_report.json"
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
                
                logger.info(f"📊 OCR Accuracy Raporu:")
                logger.info(f"   Genel Doğruluk: {total_accuracy:.1%}")
                for class_name, accuracy in class_avg_accuracies.items():
                    logger.info(f"   {class_name}: {accuracy:.1%}")
                logger.info(f"   Rapor: {report_path}")
                
        except Exception as e:
            logger.error(f"Rapor kaydetme hatası: {e}")
    
    def _fine_tune_model(self, dataset_yaml_path: str):
        """YOLO modelini yeni verilerle fine-tune et"""
        try:
            from ultralytics import YOLO
            
            # Mevcut modeli yükle
            model = YOLO(self.model_path)
            
            # Fine-tuning parametreleri
            results = model.train(
                data=dataset_yaml_path,
                epochs=10,  # Az epoch (incremental)
                imgsz=640,
                batch=4,    # Küçük batch size
                patience=5,
                save=True,
                project="runs/incremental",
                name="ls_training",
                exist_ok=True,
                verbose=False
            )
            
            # Yeni model ağırlıklarını güncelle
            try:
                save_dir = getattr(results, 'save_dir', None)
                if save_dir:
                    new_model_path = os.path.join(str(save_dir), "weights", "best.pt")
                else:
                    new_model_path = "runs/incremental/ls_training/weights/best.pt"
            except:
                new_model_path = "runs/incremental/ls_training/weights/best.pt"
            
            if os.path.exists(new_model_path):
                # Backup oluştur
                backup_path = self.model_path.replace('.pt', '_backup.pt')
                import shutil
                shutil.copy2(self.model_path, backup_path)
                
                # Yeni modeli yükle
                shutil.copy2(new_model_path, self.model_path)
                self.model = YOLO(self.model_path)
                
                logger.info(f"✅ Model güncellendi: {self.model_path}")
                logger.info(f"🔄 Backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Fine-tuning hatası: {e}")
            raise 