from PIL import Image
from core import extract_text_from_image, image_preprocess, extract_info_from_slip

def main():
    try:
        image_path = "/Users/erkamaydin/Desktop/Python/myocr/slip.jpg"
        sonuc = extract_info_from_slip(image_path)
        print("İşyeri No:", sonuc["isyeri_no"])
        print("Terminal No:", sonuc["terminal_no"])
        print("Tutar:", sonuc["tutar"])
        print("İşyeri Adı:", sonuc["isyeri_adi"])
        print("Adres:", sonuc["adres"])
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    main()
