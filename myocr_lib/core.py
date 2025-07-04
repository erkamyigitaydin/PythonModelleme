import cv2
import pytesseract
from PIL import Image
import numpy as np

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Automatically deskews a cropped image region by finding the text angle.
    """
    # Use a copy to avoid modifying the original image
    image_copy = image.copy()
    
    # Convert to grayscale
    if len(image_copy.shape) == 3 and image_copy.shape[2] == 3:
        gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_copy

    # Invert the image and apply a threshold to isolate the text.
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Find the coordinates of all black pixels.
    coords = np.column_stack(np.where(thresh > 0))
    
    # If there aren't enough points, we can't reliably determine the angle.
    if len(coords) < 20: 
        return image 

    # Find the minimum area rectangle that encloses the points
    angle = cv2.minAreaRect(coords)[-1]
    
    # The angle from minAreaRect is in [-90, 0]. We adjust it for rotation.
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # If the angle is very small, no need to rotate.
    if abs(angle) < 0.5:
        return image

    # Rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Use white as the border color to avoid black edges after rotation
    rotated = cv2.warpAffine(
        image, M, (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=(255, 255, 255)
    )
    
    return rotated

def preprocess_for_ocr(image):
    """
    Applies advanced preprocessing: denoising and adaptive thresholding.
    Deskewing should be done before calling this function.
    """
    # The image is expected to be deskewed at this point.
    if len(image.shape) >= 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # It's likely already grayscale
        gray = image

    # Denoising using Non-Local Means
    # The 'h' parameter is the filter strength. A value of 10 is a good starting point.
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Thresholding to get a clean binary image
    processed_image = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed_image

def extract_text_from_box(image, box_coords):
    """
    Crops an image using bounding box coordinates, applies advanced
    preprocessing (deskewing, denoising), and extracts text using Tesseract.
    """
    x1, y1, x2, y2 = map(int, box_coords)
    # Increased padding to give more context for deskewing
    padding = 5 
    h, w = image.shape[:2]
    
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(w, x2 + padding)
    y2_pad = min(h, y2 + padding)

    cropped_image = image[y1_pad:y2_pad, x1_pad:x2_pad]

    if cropped_image.size == 0:
        return ""

    # 1. Deskew the cropped image
    deskewed_crop = deskew_image(cropped_image)

    # 2. Apply other preprocessing (denoising, thresholding)
    preprocessed_crop = preprocess_for_ocr(deskewed_crop)
    
    try:
        pil_img = Image.fromarray(preprocessed_crop)
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(pil_img, lang="tur+eng", config=custom_config)
        text = text.replace('|', 'I').replace('€', 'E').strip()
        return text
    except Exception as e:
        print(f"Bir kutucukta OCR hatası oluştu: {e}")
        return "" 