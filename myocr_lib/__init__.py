# myocr_lib paketinden hangi fonksiyon ve değişkenlerin
# doğrudan import edilebileceğini tanımlar.

<<<<<<< HEAD
from .classes import id_to_class, class_to_id
from .core import load_model, predict_on_image, extract_text_from_box

__all__ = [
    'id_to_class', 
    'class_to_id',
    'load_model',
    'predict_on_image', 
    'extract_text_from_box'
=======
from .core import extract_text_from_box, preprocess_for_ocr
from .classes import classes, class_to_id, id_to_class

# "from myocr_lib import *" kullanıldığında nelerin import edileceğini belirler.
__all__ = [
    'extract_text_from_box',
    'preprocess_for_ocr',
    'classes',
    'class_to_id',
    'id_to_class'
>>>>>>> main
] 