#!/usr/bin/env python3
"""
Label Studio ML Backend Server
MyOCR sistemi için Flask sunucusu
"""

import os
import argparse
import logging
from flask import Flask, request, jsonify
from model import MyOCRModel

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
ml_model = None

def create_app():
    """Flask uygulamasını oluştur"""
    global ml_model
    
    app = Flask(__name__)
    
    # ML modelini başlat
    ml_model = MyOCRModel()
    
    @app.route('/predict', methods=['POST'])
    def predict():
        """Tahmin endpoint'i - Label Studio formatı"""
        try:
            if ml_model is None:
                return jsonify({'error': 'Model not loaded'}), 500
                
            data = request.get_json()
            logger.info(f"Received data: {data}")
            
            # Label Studio'dan gelen format: {"tasks": [...]}
            tasks = data.get('tasks', [])
            
            if not tasks:
                return jsonify({'error': 'No tasks provided'}), 400
            
            # ML modeli ile tahmin yap
            predictions = ml_model.predict(tasks)
            
            # Label Studio'nun beklediği format: {"results": [...]}
            response = {"results": predictions}
            
            logger.info(f"Returning {len(predictions)} predictions")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health():
        """Sağlık kontrolü"""
        return jsonify({
            'status': 'healthy',
            'model_loaded': ml_model is not None,
            'confidence_threshold': ml_model.confidence_threshold if ml_model else None
        })
    
    @app.route('/setup', methods=['POST'])
    def setup():
        """Label Studio setup endpoint'i"""
        return jsonify({
            'model_version': '1.0.0',
            'type': 'ml_backend'
        })
    
    @app.route('/', methods=['GET'])
    def index():
        """Ana sayfa"""
        return jsonify({
            'message': 'MyOCR Label Studio ML Backend',
            'version': '1.0.0',
            'endpoints': ['/predict', '/health', '/setup']
        })
    
    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MyOCR Label Studio ML Backend')
    parser.add_argument('--port', type=int, default=9090, help='Port (default: 9090)')
    parser.add_argument('--host', type=str, default='localhost', help='Host (default: localhost)')
    parser.add_argument('--confidence', type=float, default=0.2, help='Confidence threshold (default: 0.2)')
    
    args = parser.parse_args()
    
    # Environment variable ayarla
    os.environ['CONFIDENCE_THRESHOLD'] = str(args.confidence)
    
    # Flask uygulamasını başlat
    app = create_app()
    
    logger.info(f"🚀 MyOCR ML Backend başlatılıyor...")
    logger.info(f"   Host: {args.host}")
    logger.info(f"   Port: {args.port}")
    logger.info(f"   Confidence: {args.confidence}")
    logger.info(f"   URL: http://{args.host}:{args.port}")
    logger.info(f"   Health: http://{args.host}:{args.port}/health")
    logger.info(f"   Predict: http://{args.host}:{args.port}/predict")
    
    app.run(host=args.host, port=args.port, debug=False) 