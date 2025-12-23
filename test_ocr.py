"""Test script for the OCR extractor."""
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from pdf_extractor.extractors.ocr_extractor import OCRExtractor

def main():
    print("=" * 60)
    print("Testing OCR Extractor")
    print("=" * 60)
    
    # Initialize the extractor
    extractor = OCRExtractor()
    
    # Extract first 2 pages
    results = extractor.extract_pdf(
        pdf_path='Cam-Followers1.pdf',
        output_dir='output/ocr_results',
        start_page=1,
        end_page=2
    )
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Extracted {len(results)} pages")
    
    for r in results:
        page_num = r.get('page_number', 0)
        text_count = r.get('text_count', 0)
        print(f"  Page {page_num}: {text_count} text items")
        
        # Show first few text items
        text_items = r.get('text_items', [])[:5]
        for item in text_items:
            text = item.get('text', '')[:50]
            conf = item.get('confidence', 0)
            print(f"    - '{text}' (conf: {conf:.2f})")

if __name__ == "__main__":
    main()
