# MyOCR Sınıf Tanımları
classes = [
    "company_name",
    "address",
    "tax_office",
    "tax_number",
    "date",
    "time",
    "category",
    "tax_amount",
    "total_amount",
    "payment_method",
    "merchant_number",
    "receipt_number",
    "currency"
]

# Class index mapping for YOLO
class_to_id = {class_name: idx for idx, class_name in enumerate(classes)}
id_to_class = {idx: class_name for idx, class_name in enumerate(classes)} 