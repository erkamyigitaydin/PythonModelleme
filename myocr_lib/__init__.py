# myocr_lib paketinden hangi fonksiyon ve değişkenlerin
# doğrudan import edilebileceğini tanımlar.

from .core import extract_text_from_box, preprocess_for_ocr
from .classes import classes, class_to_id, id_to_class

# "from myocr_lib import *" kullanıldığında nelerin import edileceğini belirler.
__all__ = [
    'extract_text_from_box',
    'preprocess_for_ocr',
    'classes',
    'class_to_id',
    'id_to_class'
] 