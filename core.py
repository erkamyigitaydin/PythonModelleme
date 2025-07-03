import pytesseract
import cv2
from PIL import Image
import os
import re


def image_preprocess(image_path):
    if image_path == None:
        raise ValueError("Image path is required")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def extract_text_from_image(image_path):
    image = image_preprocess(image_path)
    text = pytesseract.image_to_string(image, lang="tur"+"eng")
    return text



def extract_info_from_slip(image_path):

    processed = image_preprocess(image_path)
    
    text = pytesseract.image_to_string(processed, lang="tur"+"eng")
   
    # İşyeri No
    isyeri_no = None
    isyeri_match = re.search(r"İ?S?YERİ\s*NO[:\s]*([0-9]+)", text, re.IGNORECASE)
    if isyeri_match:
        isyeri_no = isyeri_match.group(1)
    else:
        isyeri_match = re.search(r"İ?S?YERİ\s*NO[:\s]*([0-9]+)[\s\.]", text, re.IGNORECASE)
        if isyeri_match:
            isyeri_no = isyeri_match.group(1)

    # Terminal No
    terminal_no = None
    match = re.search(r"TERMINAL\s*NO[:\s]*([0-9]+)", text, re.IGNORECASE)
    if match:
        terminal_no = match.group(1)

    # Toplam Tutar
    tutar = None
    match = re.search(r"Toplam Tutar[:\s]*([\d.,]+)", text, re.IGNORECASE)
    if match:
        tutar = match.group(1)

    lines = text.strip().split("\n")
    # Boş satırları temizle
    
    lines = [line.strip() for line in lines if line.strip()]
    # Genelde ilk satır işyeri adı, sonraki satırlar adres ve şehir olur
    
    isyeri_adi = lines[0] if len(lines) > 0 else None
    adres_satirlari = []
    
    # İlk 3-4 satırı "adres" olarak birleştir
    for i in range(1, min(4, len(lines))):
        adres_satirlari.append(lines[i])
    adres = " ".join(adres_satirlari)
    
    return {
        "sonuc": text,
        "isyeri_no": isyeri_no,
        "terminal_no": terminal_no,
        "tutar": tutar,
        "raw_text": text,
        "isyeri_adi": isyeri_adi,
        "adres": adres
    }
