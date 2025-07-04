import cv2
import pytesseract
from PIL import Image

def preprocess_for_ocr(image):
    """
    Bir görüntüye OCR doğruluğunu artırmak için ön işleme adımları uygular.
    """
    # Görüntü BGR formatında değilse devam etme
    if len(image.shape) < 3 or image.shape[2] != 3:
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed_image

def extract_text_from_box(image, box_coords):
    """
    Bir görüntüyü sınırlayıcı kutu koordinatlarını kullanarak kırpar ve
    Tesseract kullanarak metin çıkarımı yapar.
    """
    x1, y1, x2, y2 = map(int, box_coords)
    padding = 2
    h, w = image.shape[:2]
    
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)

    cropped_image = image[y1_pad:y2_pad, x1_pad:x2_pad]

    if cropped_image.size == 0:
        return ""

    preprocessed_crop = preprocess_for_ocr(cropped_image)
    
    try:
        pil_img = Image.fromarray(preprocessed_crop)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(pil_img, lang="tur+eng", config=custom_config)
        text = text.replace('|', 'I').replace('€', 'E').strip()
        return text
    except Exception as e:
        print(f"Bir kutucukta OCR hatası oluştu: {e}")
        return "" 