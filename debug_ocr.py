"""Debug PaddleOCR output format in detail."""
import fitz
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import json

# Initialize PaddleOCR
print("Initializing PaddleOCR...")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Open PDF and get first page
print("Opening PDF...")
doc = fitz.open('Cam-Followers1.pdf')
page = doc[0]

# Convert to image at 300 DPI
zoom = 300 / 72.0
matrix = fitz.Matrix(zoom, zoom)
pixmap = page.get_pixmap(matrix=matrix, alpha=False)
img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
img_array = np.array(img)

# Save image to debug
img.save('debug_page1.png')
print(f"Image saved as debug_page1.png, size: {img_array.shape}")

# Run OCR
print("Running OCR...")
result = ocr.ocr(img_array)

print(f"\n=== OCR Result Debug ===")
print(f"Result type: {type(result)}")

if result is None:
    print("Result is None!")
elif isinstance(result, dict):
    print("Result is a dict!")
    for key, value in result.items():
        print(f"  Key: {key}, Type: {type(value)}")
elif isinstance(result, list):
    print(f"Result is a list with {len(result)} elements")
    
    for i, item in enumerate(result):
        print(f"\n--- Element {i} ---")
        print(f"Type: {type(item)}")
        
        if item is None:
            print("  Item is None")
        elif isinstance(item, list):
            print(f"  List with {len(item)} items")
            if len(item) > 0:
                for j, sub_item in enumerate(item[:3]):  # First 3 items
                    print(f"    Sub-item {j}: {type(sub_item)}")
                    if isinstance(sub_item, (list, tuple)) and len(sub_item) >= 2:
                        bbox = sub_item[0]
                        text_info = sub_item[1]
                        print(f"      bbox type: {type(bbox)}")
                        print(f"      text_info type: {type(text_info)}, value: {text_info}")
        elif isinstance(item, dict):
            print(f"  Dict with keys: {item.keys()}")
            for k, v in list(item.items())[:3]:
                print(f"    {k}: {type(v)} = {str(v)[:100]}")
        else:
            print(f"  Value: {item}")
else:
    print(f"Unknown result type: {type(result)}")
    print(f"Result: {result}")

doc.close()
print("\nDone!")
