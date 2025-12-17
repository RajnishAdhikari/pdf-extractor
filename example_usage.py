"""
Example Usage - PDF Content Extractor

This script demonstrates how to use the PDF Extractor module
to extract text, images, and tables from PDF files.
"""

from pdf_extractor import PDFExtractor
from pdf_extractor.utils.logger import setup_logger
import logging


def basic_extraction_example(pdf_path: str):
    """
    Basic example: Extract all content from a PDF.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Full Extraction")
    print("=" * 60)
    
    # Initialize extractor with default settings
    extractor = PDFExtractor(
        output_dir="./output",
        extract_text=True,
        extract_images=True,
        extract_tables=True
    )
    
    # Extract all content
    result = extractor.extract(pdf_path)
    
    # Access results
    print(f"\nTotal pages: {result.total_pages}")
    print(f"Extraction time: {result.extraction_time:.2f} seconds")
    
    # Print content per page
    for page in result.pages:
        print(f"\n--- Page {page.page_number} ---")
        
        # Text content
        for text in page.texts:
            preview = text.content[:100] + "..." if len(text.content) > 100 else text.content
            print(f"  Text ({text.word_count} words): {preview}")
        
        # Images
        for img in page.images:
            print(f"  Image: {img.width}x{img.height} {img.format} -> {img.image_path}")
        
        # Tables
        for table in page.tables:
            print(f"  Table: {table.rows}x{table.columns} -> {table.csv_path}")
    
    return result


def text_only_example(pdf_path: str):
    """
    Example: Extract only text content.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Text-Only Extraction")
    print("=" * 60)
    
    extractor = PDFExtractor(
        output_dir="./output",
        extract_text=True,
        extract_images=False,
        extract_tables=False
    )
    
    # Use dedicated method for text-only extraction
    text_results = extractor.extract_text_only(pdf_path)
    
    total_words = sum(t.word_count for t in text_results)
    print(f"\nExtracted text from {len(text_results)} pages")
    print(f"Total words: {total_words}")
    
    # Print text by page
    for text in text_results:
        print(f"\nPage {text.page_number}: {text.word_count} words")
    
    return text_results


def specific_pages_example(pdf_path: str, pages: list):
    """
    Example: Extract content from specific pages only.
    """
    print("\n" + "=" * 60)
    print(f"EXAMPLE 3: Extract Specific Pages {pages}")
    print("=" * 60)
    
    extractor = PDFExtractor(output_dir="./output")
    
    # Extract only specified pages
    result = extractor.extract(pdf_path, pages=pages)
    
    print(f"\nExtracted {len(result.pages)} pages: {[p.page_number for p in result.pages]}")
    
    return result


def analyze_pdf_example(pdf_path: str):
    """
    Example: Analyze PDF structure without extracting.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Analyze PDF Structure")
    print("=" * 60)
    
    extractor = PDFExtractor()
    
    # Analyze without extracting
    analysis = extractor.analyze(pdf_path)
    
    print(f"\nPDF: {analysis['pdf_path']}")
    print(f"Total pages: {analysis['metadata'].get('page_count', 'N/A')}")
    print(f"File size: {analysis['metadata'].get('file_size_bytes', 0) / 1024:.2f} KB")
    
    print("\nPage Analysis:")
    for page in analysis['pages']:
        print(f"  Page {page['page_number']}: "
              f"{page['text_length']} chars, "
              f"{page['image_count']} images, "
              f"{page.get('table_count', 'N/A')} tables")
    
    return analysis


def json_output_example(pdf_path: str):
    """
    Example: Work with JSON output.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: JSON Output")
    print("=" * 60)
    
    extractor = PDFExtractor(output_dir="./output")
    result = extractor.extract(pdf_path, save_json=True)
    
    # Convert to JSON string
    json_output = result.to_json(indent=2)
    
    # Print first 500 characters
    print(f"\nJSON Output Preview:\n{json_output[:500]}...")
    
    # Get as dictionary
    result_dict = result.to_dict()
    print(f"\nSummary from dict: {result_dict['summary']}")
    
    return result


def main():
    """Run all examples."""
    # Setup logging
    setup_logger("pdf_extractor", level=logging.INFO)
    
    # IMPORTANT: Replace with your actual PDF path
    pdf_path = "sample.pdf"
    
    print("\n" + "#" * 60)
    print("# PDF EXTRACTOR - USAGE EXAMPLES")
    print("#" * 60)
    print(f"\nUsing PDF: {pdf_path}")
    print("Make sure to replace 'sample.pdf' with your actual PDF file!")
    
    try:
        # Run examples
        basic_extraction_example(pdf_path)
        text_only_example(pdf_path)
        specific_pages_example(pdf_path, pages=[1, 2, 3])
        analyze_pdf_example(pdf_path)
        json_output_example(pdf_path)
        
    except FileNotFoundError:
        print(f"\nError: PDF file not found: {pdf_path}")
        print("Please provide a valid PDF file path.")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
