"""
PDF Extractor - Main Entry Point

Fast and efficient PDF content extraction using open-source libraries.
Extracts text, images, and tables with page number tracking.

Usage:
    python main.py path/to/document.pdf [--output-dir ./output] [--pages 1,2,3]
    
Example:
    python main.py sample.pdf --output-dir ./extracted
    python main.py sample.pdf --pages 1,5,10 --no-images
"""

import argparse
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pdf_extractor import PDFExtractor
from pdf_extractor.utils.logger import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract text, images, and tables from PDF files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py document.pdf
  python main.py document.pdf --output-dir ./extracted
  python main.py document.pdf --pages 1,2,3,4,5
  python main.py document.pdf --no-images --no-tables
  python main.py document.pdf --analyze
        """
    )
    
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to extract"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="./extracted_content",
        help="Output directory for extracted content (default: ./extracted_content)"
    )
    
    parser.add_argument(
        "--pages", "-p",
        help="Comma-separated list of page numbers to extract (e.g., 1,2,3)"
    )
    
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Skip text extraction"
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip image extraction"
    )
    
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Skip table extraction"
    )
    
    parser.add_argument(
        "--image-format",
        choices=["png", "jpeg", "webp"],
        default="png",
        help="Output format for extracted images (default: png)"
    )
    
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=50,
        help="Minimum image dimension to extract (default: 50)"
    )
    
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze PDF structure without extracting content"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Don't save JSON output file"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger("pdf_extractor", level=log_level)
    
    # Parse pages if provided
    pages = None
    if args.pages:
        try:
            pages = [int(p.strip()) for p in args.pages.split(",")]
        except ValueError:
            logger.error("Invalid page numbers. Use comma-separated integers (e.g., 1,2,3)")
            sys.exit(1)
    
    # Initialize extractor
    extractor = PDFExtractor(
        output_dir=args.output_dir,
        extract_text=not args.no_text,
        extract_images=not args.no_images,
        extract_tables=not args.no_tables,
        image_format=args.image_format,
        min_image_size=args.min_image_size,
        export_tables_csv=True
    )
    
    try:
        if args.analyze:
            # Analyze mode
            logger.info(f"Analyzing: {args.pdf_path}")
            analysis = extractor.analyze(args.pdf_path)
            print(json.dumps(analysis, indent=2))
        else:
            # Full extraction
            logger.info(f"Extracting from: {args.pdf_path}")
            result = extractor.extract(
                args.pdf_path,
                pages=pages,
                save_json=not args.no_json
            )
            
            # Print summary
            summary = result.to_dict()["summary"]
            print("\n" + "=" * 50)
            print("EXTRACTION COMPLETE")
            print("=" * 50)
            print(f"PDF: {args.pdf_path}")
            print(f"Pages processed: {result.total_pages}")
            print(f"Text blocks: {summary['total_text_blocks']}")
            print(f"Images extracted: {summary['total_images']}")
            print(f"Tables extracted: {summary['total_tables']}")
            print(f"Time: {result.extraction_time:.2f} seconds")
            print(f"Output: {args.output_dir}")
            print("=" * 50)
            
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
