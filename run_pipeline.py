#!/usr/bin/env python
"""
PDF Extraction Pipeline Runner
Simple script to run the complete extraction pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pdf_extractor.main import PDFExtractionPipeline
from pdf_extractor.config import Config

logger = Config.setup_logging("run_pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Run the PDF Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run full pipeline:
    python run_pipeline.py --pdf Cam-Followers1.pdf
  
  Skip OCR (use existing results):
    python run_pipeline.py --pdf Cam-Followers1.pdf --skip-ocr
  
  Only run OCR:
    python run_pipeline.py --pdf Cam-Followers1.pdf --only-ocr
  
  Only run table detection:
    python run_pipeline.py --pdf Cam-Followers1.pdf --only-tables
  
  Query after processing:
    python run_pipeline.py --pdf Cam-Followers1.pdf --query "CF12 specifications"
  
  Interactive mode:
    python run_pipeline.py --pdf Cam-Followers1.pdf --interactive
"""
    )
    
    parser.add_argument(
        "--pdf", "-p",
        default="Cam-Followers1.pdf",
        help="Path to the PDF file to process (default: Cam-Followers1.pdf)"
    )
    
    # Skip options
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR extraction")
    parser.add_argument("--skip-tables", action="store_true", help="Skip table detection")
    parser.add_argument("--skip-grouping", action="store_true", help="Skip model grouping")
    parser.add_argument("--skip-rag", action="store_true", help="Skip RAG setup")
    
    # Only options (mutually exclusive with skip)
    parser.add_argument("--only-ocr", action="store_true", help="Only run OCR extraction")
    parser.add_argument("--only-tables", action="store_true", help="Only run table detection")
    parser.add_argument("--only-grouping", action="store_true", help="Only run model grouping")
    parser.add_argument("--only-rag", action="store_true", help="Only run RAG setup")
    
    # Query options
    parser.add_argument("--query", "-q", help="Query to run after pipeline")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive query mode")
    
    args = parser.parse_args()
    
    # Resolve PDF path
    pdf_path = Path(args.pdf)
    if not pdf_path.is_absolute():
        pdf_path = project_root / pdf_path
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return 1
    
    # Handle "only" options
    if args.only_ocr:
        args.skip_tables = True
        args.skip_grouping = True
        args.skip_rag = True
    elif args.only_tables:
        args.skip_ocr = True
        args.skip_grouping = True
        args.skip_rag = True
    elif args.only_grouping:
        args.skip_ocr = True
        args.skip_tables = True
        args.skip_rag = True
    elif args.only_rag:
        args.skip_ocr = True
        args.skip_tables = True
        args.skip_grouping = True
    
    # Create and run pipeline
    logger.info(f"Starting pipeline for: {pdf_path.name}")
    
    pipeline = PDFExtractionPipeline(str(pdf_path))
    
    success = pipeline.run_full_pipeline(
        skip_ocr=args.skip_ocr,
        skip_tables=args.skip_tables,
        skip_grouping=args.skip_grouping,
        skip_rag=args.skip_rag
    )
    
    if not success:
        logger.error("Pipeline completed with errors")
        return 1
    
    # Handle queries
    if args.query:
        print("\n" + "=" * 50)
        print("QUERY RESULTS")
        print("=" * 50)
        
        result = pipeline.query(args.query)
        print(f"Query: {args.query}")
        
        if result.get("success"):
            print(f"Method: {result.get('method')}")
            for match in result.get("matches", []):
                print(f"  - {match['model_name']} (score: {match['score']:.3f})")
            
            if pipeline.rag_retriever:
                print("\nContext for LLM:")
                print("-" * 40)
                context = pipeline.rag_retriever.get_context_for_llm(args.query)
                print(context[:2000] + "..." if len(context) > 2000 else context)
        else:
            print(f"No results: {result.get('error')}")
    
    if args.interactive and pipeline.rag_retriever:
        pipeline.rag_retriever.interactive_query()
    
    logger.info("Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
