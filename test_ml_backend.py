#!/usr/bin/env python3
"""
ML Backend Test Scripti
Label Studio entegrasyonunu test eder
"""

import requests
import json
import os

def test_ml_backend():
    """ML Backend'i test et"""
    base_url = "http://localhost:9090"
    
    print("🧪 ML Backend Test Başlatılıyor...")
    print("=" * 50)
    
    # 1. Health check
    print("1. Health check testi...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print(f"   ✅ Status: {health_data.get('status')}")
            print(f"   ✅ Model loaded: {health_data.get('model_loaded')}")
            print(f"   ✅ Confidence: {health_data.get('confidence_threshold')}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # 2. Prediction test
    print("\n2. Prediction test...")
    
    # Test görüntüsü yolu
    test_image = "dataset/train/images/slip_tr_119.jpg"
    if not os.path.exists(test_image):
        print(f"   ❌ Test görüntüsü bulunamadı: {test_image}")
        return False
    
    # Test payload'ı - Doğru Label Studio formatı
    test_payload = {
        "tasks": [
            {
                "id": 1,
                "data": {
                    "image": test_image
                }
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict", 
            json=test_payload, 
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            # Label Studio yanıt formatı: {"results": [...]}
            predictions = result.get('results', [])
            
            if predictions:
                prediction = predictions[0]
                results = prediction.get('result', [])
                score = prediction.get('score', 0)
                
                print(f"   ✅ Tahmin başarılı!")
                print(f"   ✅ Tespit edilen alan sayısı: {len(results)}")
                print(f"   ✅ Ortalama skor: {score:.3f}")
                
                # İlk birkaç tespit detayı
                for i, result in enumerate(results[:3]):
                    class_name = result['value']['rectanglelabels'][0]
                    confidence = result.get('score', 0)
                    print(f"   📍 Alan {i+1}: {class_name} (güven: {confidence:.2f})")
                
                return True
            else:
                print("   ⚠️ Hiç alan tespit edilmedi")
                return True
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            print(f"   ❌ Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"   ❌ Prediction error: {e}")
        return False
    
    return True

def main():
    """Ana test fonksiyonu"""
    print("🚀 MyOCR Label Studio ML Backend Testi")
    print("\n⚠️  Not: ML Backend'in çalışıyor olması gerekir!")
    print("   Başlatmak için: ./start_ml_backend.sh")
    print()
    
    # Backend'in çalışıp çalışmadığını kontrol et
    try:
        response = requests.get("http://localhost:9090/", timeout=3)
        if response.status_code != 200:
            print("❌ ML Backend çalışmıyor! Önce ./start_ml_backend.sh çalıştırın.")
            return
    except:
        print("❌ ML Backend çalışmıyor! Önce ./start_ml_backend.sh çalıştırın.")
        return
    
    # Test başlat
    success = test_ml_backend()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 ML Backend test başarılı!")
        print("✅ Label Studio entegrasyonu hazır!")
    else:
        print("❌ ML Backend test başarısız!")
        print("🔧 Loglara bakın ve hataları düzeltin.")

if __name__ == "__main__":
    main() 