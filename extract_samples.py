"""
Extract Content from Sample PDFs/Images

This script extracts text, tables, and data from THK specification sheets.
Uses Tesseract OCR for accurate text recognition.

Prerequisites:
    1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
    2. pip install pytesseract pillow

Usage:
    python extract_samples.py
"""

import os
import json
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from pdf_extractor.extractors.ocr_extractor import OCRExtractor
    from pdf_extractor.utils.logger import setup_logger
    from pdf_extractor.utils.file_utils import ensure_directory
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    OCR_AVAILABLE = False


def check_tesseract():
    """Check if Tesseract is installed."""
    import shutil
    tesseract_path = shutil.which("tesseract")
    
    if tesseract_path:
        print(f"✓ Tesseract found: {tesseract_path}")
        return True
    
    # Check common Windows paths
    windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
    ]
    
    for path in windows_paths:
        if os.path.exists(path):
            print(f"✓ Tesseract found: {path}")
            os.environ['PATH'] += os.pathsep + os.path.dirname(path)
            return True
    
    return False


def extract_with_pillow_fallback():
    """
    Extract basic image info when OCR is not available.
    Uses Pillow to analyze images.
    """
    from PIL import Image
    
    sample_folder = Path("sample_pdfs")
    output_folder = Path("extracted_content")
    ensure_directory(str(output_folder))
    
    print("=" * 60)
    print("IMAGE ANALYSIS (Pillow-based)")
    print("=" * 60)
    
    image_files = list(sample_folder.glob("*.jpg")) + \
                  list(sample_folder.glob("*.png"))
    
    results = []
    
    for idx, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{idx}/{len(image_files)}] Analyzing: {image_path.name}")
        
        img = Image.open(image_path)
        
        result = {
            "page_number": idx,
            "filename": image_path.name,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format
        }
        results.append(result)
        
        print(f"  Size: {img.width}x{img.height}")
        print(f"  Mode: {img.mode}, Format: {img.format}")
        
        # Save thumbnail
        thumb_path = output_folder / f"{image_path.stem}_thumbnail.png"
        img.thumbnail((400, 400))
        img.save(thumb_path)
        print(f"  ✓ Thumbnail saved: {thumb_path}")
    
    # Save results
    json_path = output_folder / "image_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Analysis saved: {json_path}")
    
    return results


def extract_all_samples():
    """Extract content from all sample images using OCR."""
    
    setup_logger("pdf_extractor", level=20)  # INFO level
    
    sample_folder = Path("sample_pdfs")
    output_folder = Path("extracted_content")
    ensure_directory(str(output_folder))
    
    print("=" * 60)
    print("PDF/IMAGE CONTENT EXTRACTOR")
    print("Extracting from THK Cam Follower Specification Sheets")
    print("=" * 60)
    
    # Check samples
    if not sample_folder.exists():
        print(f"\nError: Sample folder not found: {sample_folder}")
        return
    
    image_files = list(sample_folder.glob("*.jpg")) + \
                  list(sample_folder.glob("*.png"))
    
    if not image_files:
        print(f"\nNo images found in {sample_folder}")
        return
    
    print(f"\nFound {len(image_files)} images:")
    for f in sorted(image_files):
        print(f"  - {f.name}")
    
    # Check Tesseract
    print("\n" + "-" * 60)
    if not check_tesseract():
        print("\n⚠ Tesseract not found!")
        print("\nTo install Tesseract on Windows:")
        print("  1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("  2. Run the installer (install to default location)")
        print("  3. Add to PATH or restart terminal")
        print("\nFalling back to image analysis only...")
        return extract_with_pillow_fallback()
    
    # Initialize OCR
    print("-" * 60)
    
    ocr = OCRExtractor(lang='eng')
    
    # Process images
    all_results = []
    
    for idx, image_path in enumerate(sorted(image_files), 1):
        print(f"\n[{idx}/{len(image_files)}] Processing: {image_path.name}")
        print("-" * 40)
        
        # Extract text
        text_result = ocr.extract_from_image(str(image_path), page_number=idx)
        
        # Extract with positions
        pos_result = ocr.extract_with_positions(str(image_path), page_number=idx)
        pos_result['filename'] = image_path.name
        pos_result['full_text'] = text_result.content
        
        # Extract table structure
        table_rows = ocr.extract_tables_ocr(str(image_path), page_number=idx)
        pos_result['table_structure'] = table_rows
        
        all_results.append(pos_result)
        
        # Summary
        print(f"  ✓ Extracted {pos_result['total_blocks']} text blocks")
        print(f"  ✓ Detected {len(table_rows)} lines")
        print(f"  ✓ Word count: {text_result.word_count}")
    
    # Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # JSON
    json_path = output_folder / "extraction_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Complete results: {json_path}")
    
    # Individual files
    for result in all_results:
        filename = Path(result['filename']).stem
        
        # Text file
        text_path = output_folder / f"{filename}_text.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(f"Page {result['page_number']}: {result['filename']}\n")
            f.write("=" * 60 + "\n\n")
            f.write(result.get('full_text', ''))
        print(f"✓ Text: {text_path}")
        
        # CSV table
        csv_path = output_folder / f"{filename}_table.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            for row in result.get('table_structure', []):
                f.write(",".join([f'"{c}"' for c in row['cells']]) + "\n")
        print(f"✓ Table: {csv_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    total_blocks = sum(r['total_blocks'] for r in all_results)
    print(f"\nProcessed: {len(all_results)} pages")
    print(f"Total text blocks: {total_blocks}")
    print(f"Output: {output_folder.absolute()}")
    
    return all_results


if __name__ == "__main__":
    if not OCR_AVAILABLE:
        print("OCR modules not available. Checking dependencies...")
        extract_with_pillow_fallback()
    else:
        extract_all_samples()
