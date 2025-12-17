"""
Simple Image Extractor - No OCR Dependencies
Extracts and processes images from the sample folder.
Works without Tesseract or EasyOCR.

This script:
1. Analyzes images (size, format)
2. Creates enhanced versions for better readability
3. Saves image info as JSON

For OCR text extraction, you'll need to install Tesseract separately.

Usage:
    python simple_extract.py
"""

import os
import json
from pathlib import Path
from datetime import datetime

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def ensure_directory(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(Path(path).absolute())


def analyze_image(image_path, page_number):
    """Analyze an image and extract metadata."""
    img = Image.open(image_path)
    
    return {
        "page_number": page_number,
        "filename": Path(image_path).name,
        "path": str(image_path),
        "width": img.width,
        "height": img.height,
        "mode": img.mode,
        "format": img.format or "Unknown",
        "size_bytes": os.path.getsize(image_path),
        "size_kb": round(os.path.getsize(image_path) / 1024, 2)
    }


def enhance_image(image_path, output_path):
    """Enhance image for better readability."""
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    # Save enhanced version
    img.save(output_path, quality=95)
    return output_path


def create_thumbnail(image_path, output_path, max_size=(600, 600)):
    """Create thumbnail of image."""
    img = Image.open(image_path)
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    img.save(output_path)
    return output_path


def extract_images():
    """Main extraction function."""
    
    print("=" * 60)
    print("SIMPLE IMAGE EXTRACTOR")
    print("THK Cam Follower Specification Sheets")
    print("=" * 60)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if not PIL_AVAILABLE:
        print("\nError: Pillow not installed. Run: pip install pillow")
        return
    
    # Setup folders
    sample_folder = Path("sample_pdfs")
    output_folder = Path("extracted_content")
    ensure_directory(str(output_folder))
    
    # Find images
    if not sample_folder.exists():
        print(f"\nError: Folder not found: {sample_folder}")
        print("Please create the folder and add your images.")
        return []
    
    extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(sample_folder.glob(f"*{ext}"))
        image_files.extend(sample_folder.glob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"\nNo images found in {sample_folder}")
        return []
    
    print(f"\nFound {len(image_files)} images:")
    for f in image_files:
        print(f"  - {f.name}")
    
    # Process each image
    print("\n" + "-" * 60)
    print("PROCESSING IMAGES")
    print("-" * 60)
    
    results = []
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] {image_path.name}")
        
        # Analyze
        info = analyze_image(image_path, idx)
        print(f"  Size: {info['width']}x{info['height']} ({info['size_kb']} KB)")
        
        # Create enhanced version
        enhanced_path = output_folder / f"{image_path.stem}_enhanced.jpg"
        enhance_image(image_path, enhanced_path)
        info['enhanced_path'] = str(enhanced_path)
        print(f"  ✓ Enhanced: {enhanced_path.name}")
        
        # Create thumbnail
        thumb_path = output_folder / f"{image_path.stem}_thumb.jpg"
        create_thumbnail(image_path, thumb_path)
        info['thumbnail_path'] = str(thumb_path)
        print(f"  ✓ Thumbnail: {thumb_path.name}")
        
        results.append(info)
    
    # Save JSON results
    print("\n" + "-" * 60)
    print("SAVING RESULTS")
    print("-" * 60)
    
    json_path = output_folder / "image_analysis.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Analysis saved: {json_path}")
    
    # Create summary text file
    summary_path = output_folder / "extraction_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("PDF/IMAGE EXTRACTION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images: {len(results)}\n\n")
        
        for result in results:
            f.write(f"Page {result['page_number']}: {result['filename']}\n")
            f.write(f"  Dimensions: {result['width']}x{result['height']}\n")
            f.write(f"  Size: {result['size_kb']} KB\n")
            f.write(f"  Enhanced: {result['enhanced_path']}\n")
            f.write(f"  Thumbnail: {result['thumbnail_path']}\n\n")
    
    print(f"✓ Summary saved: {summary_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    
    total_size = sum(r['size_kb'] for r in results)
    print(f"\nImages processed: {len(results)}")
    print(f"Total size: {total_size:.2f} KB")
    print(f"Output folder: {output_folder.absolute()}")
    
    print("\n" + "-" * 60)
    print("NEXT STEPS FOR OCR:")
    print("-" * 60)
    print("""
To extract TEXT from these images, install Tesseract OCR:

Windows:
  1. Download: https://github.com/UB-Mannheim/tesseract/wiki
  2. Install to default location
  3. pip install pytesseract
  4. Run: python extract_samples.py

Or use online OCR tools:
  - Google Drive (upload image, right-click -> Open with Google Docs)
  - https://www.onlineocr.net/
  - https://www.i2ocr.com/
""")
    
    return results


if __name__ == "__main__":
    extract_images()
