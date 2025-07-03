from PIL import Image
from core import extract_text_from_image, image_preprocess, extract_info_from_slip
import argparse

def main(image_path):
    try:
        sonuc = extract_info_from_slip(image_path)
        print("sonuc:", sonuc["sonuc"])
        print("İşyeri No:", sonuc["isyeri_no"])
        print("Terminal No:", sonuc["terminal_no"])
        print("Tutar:", sonuc["tutar"])
        print("İşyeri Adı:", sonuc["isyeri_adi"])
        print("Adres:", sonuc["adres"])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract slip information from an image")
    parser.add_argument("image_path", help="Path to the slip image")
    args = parser.parse_args()
    main(args.image_path)
