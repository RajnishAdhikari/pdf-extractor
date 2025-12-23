"""
Main entry point for the PDF Extraction Pipeline.
Orchestrates OCR extraction, table detection, model grouping, and RAG setup.
"""

import argparse
import sys
from pathlib import Path

from .config import Config
from .extractors.ocr_extractor import OCRExtractor
from .extractors.table_detector import TableDetector
from .processors.model_grouper import ModelGrouper
from .processors.rag_setup import RAGRetriever

logger = Config.setup_logging(__name__)


class PDFExtractionPipeline:
    """
    Main pipeline class that orchestrates the complete extraction process.
    """
    
    def __init__(self, pdf_path: str = None):
        """
        Initialize the pipeline.
        
        Args:
            pdf_path: Path to the PDF file to process
        """
        self.pdf_path = Path(pdf_path) if pdf_path else None
        
        # Initialize components
        self.ocr_extractor = None
        self.table_detector = None
        self.model_grouper = None
        self.rag_retriever = None
        
        # Ensure output directories exist
        Config.ensure_directories()
        
        logger.info("PDF Extraction Pipeline initialized")
    
    def run_ocr_extraction(
        self,
        pdf_path: str = None,
        output_dir: str = None
    ) -> bool:
        """
        Run OCR extraction on the PDF.
        
        Args:
            pdf_path: Path to the PDF file (uses default if not provided)
            output_dir: Output directory for OCR results
            
        Returns:
            True if successful
        """
        pdf_path = pdf_path or self.pdf_path
        if not pdf_path:
            logger.error("No PDF path specified")
            return False
        
        logger.info("=" * 50)
        logger.info("STAGE 1: OCR EXTRACTION")
        logger.info("=" * 50)
        
        self.ocr_extractor = OCRExtractor()
        results = self.ocr_extractor.extract_pdf(
            pdf_path=str(pdf_path),
            output_dir=output_dir
        )
        
        if results:
            total_items = sum(r.get("text_count", 0) for r in results)
            logger.info(f"OCR extraction complete: {len(results)} pages, {total_items} text items")
            return True
        else:
            logger.error("OCR extraction failed")
            return False
    
    def run_table_detection(
        self,
        pdf_path: str = None,
        ocr_input_dir: str = None,
        output_dir: str = None
    ) -> bool:
        """
        Run table detection on the OCR results.
        
        Args:
            pdf_path: Path to the PDF file (for loading images)
            ocr_input_dir: Directory containing OCR JSON files
            output_dir: Output directory for table results
            
        Returns:
            True if successful
        """
        pdf_path = pdf_path or self.pdf_path
        
        logger.info("=" * 50)
        logger.info("STAGE 2: TABLE DETECTION")
        logger.info("=" * 50)
        
        self.table_detector = TableDetector()
        results = self.table_detector.process_all_pages(
            ocr_input_dir=ocr_input_dir,
            output_dir=output_dir,
            pdf_path=str(pdf_path) if pdf_path else None
        )
        
        if results:
            tables_found = sum(len(r.get("tables", [])) for r in results)
            logger.info(f"Table detection complete: {len(results)} pages, {tables_found} tables")
            return True
        else:
            logger.error("Table detection failed")
            return False
    
    def run_model_grouping(
        self,
        input_dir: str = None,
        output_dir: str = None,
        known_models: list = None
    ) -> bool:
        """
        Run model/part number grouping.
        
        Args:
            input_dir: Directory containing page JSON files
            output_dir: Output directory for grouped results
            known_models: Optional list of known model names
            
        Returns:
            True if successful
        """
        logger.info("=" * 50)
        logger.info("STAGE 3: MODEL GROUPING")
        logger.info("=" * 50)
        
        self.model_grouper = ModelGrouper(known_models=known_models)
        results = self.model_grouper.group_by_model(
            input_dir=input_dir,
            output_dir=output_dir
        )
        
        if results:
            total_pages = sum(r.get("page_count", 0) for r in results.values())
            logger.info(f"Model grouping complete: {len(results)} models, {total_pages} page assignments")
            return True
        else:
            logger.error("Model grouping failed")
            return False
    
    def run_rag_setup(
        self,
        data_dir: str = None,
        build_index: bool = True
    ) -> bool:
        """
        Run RAG index setup.
        
        Args:
            data_dir: Directory containing grouped model data
            build_index: Whether to build the semantic search index
            
        Returns:
            True if successful
        """
        logger.info("=" * 50)
        logger.info("STAGE 4: RAG SETUP")
        logger.info("=" * 50)
        
        self.rag_retriever = RAGRetriever()
        
        if not self.rag_retriever.load_data(data_dir):
            logger.error("Failed to load model data for RAG")
            return False
        
        if build_index:
            if self.rag_retriever.build_index():
                self.rag_retriever.save_index()
                logger.info("RAG index built and saved successfully")
            else:
                logger.warning("Semantic search index not built (direct lookup still available)")
        
        logger.info(f"RAG setup complete: {len(self.rag_retriever.model_data)} models indexed")
        return True
    
    def run_full_pipeline(
        self,
        pdf_path: str = None,
        skip_ocr: bool = False,
        skip_tables: bool = False,
        skip_grouping: bool = False,
        skip_rag: bool = False
    ) -> bool:
        """
        Run the complete extraction pipeline.
        
        Args:
            pdf_path: Path to the PDF file
            skip_ocr: Skip OCR extraction (use existing results)
            skip_tables: Skip table detection (use existing results)
            skip_grouping: Skip model grouping (use existing results)
            skip_rag: Skip RAG setup
            
        Returns:
            True if all stages completed successfully
        """
        pdf_path = pdf_path or self.pdf_path
        if not pdf_path:
            logger.error("No PDF path specified")
            return False
        
        self.pdf_path = Path(pdf_path)
        
        logger.info("=" * 60)
        logger.info(f"PDF EXTRACTION PIPELINE - {self.pdf_path.name}")
        logger.info("=" * 60)
        
        success = True
        
        # Stage 1: OCR Extraction
        if not skip_ocr:
            if not self.run_ocr_extraction():
                logger.error("Pipeline failed at OCR extraction stage")
                return False
        else:
            logger.info("Skipping OCR extraction (using existing results)")
        
        # Stage 2: Table Detection
        if not skip_tables:
            if not self.run_table_detection():
                logger.warning("Table detection failed, continuing without tables")
        else:
            logger.info("Skipping table detection (using existing results)")
        
        # Stage 3: Model Grouping
        if not skip_grouping:
            if not self.run_model_grouping():
                logger.error("Pipeline failed at model grouping stage")
                return False
        else:
            logger.info("Skipping model grouping (using existing results)")
        
        # Stage 4: RAG Setup
        if not skip_rag:
            if not self.run_rag_setup():
                logger.warning("RAG setup failed, continuing without RAG")
        else:
            logger.info("Skipping RAG setup")
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return success
    
    def query(self, query_string: str) -> dict:
        """
        Query the RAG system.
        
        Args:
            query_string: Query string
            
        Returns:
            Query results dictionary
        """
        if self.rag_retriever is None:
            logger.warning("RAG not initialized, attempting to load...")
            if not self.run_rag_setup(build_index=False):
                return {"success": False, "error": "RAG not available"}
        
        return self.rag_retriever.query(query_string)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="PDF Extraction Pipeline - Extract and organize data from scanned PDFs"
    )
    parser.add_argument(
        "--pdf", "-p",
        required=True,
        help="Path to the PDF file to process"
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR extraction (use existing results)"
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip table detection"
    )
    parser.add_argument(
        "--skip-grouping",
        action="store_true",
        help="Skip model grouping"
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG setup"
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Query to run after pipeline completion"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive query mode after pipeline"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = PDFExtractionPipeline(args.pdf)
    success = pipeline.run_full_pipeline(
        skip_ocr=args.skip_ocr,
        skip_tables=args.skip_tables,
        skip_grouping=args.skip_grouping,
        skip_rag=args.skip_rag
    )
    
    if not success:
        logger.error("Pipeline completed with errors")
        return 1
    
    # Handle post-pipeline queries
    if args.query:
        result = pipeline.query(args.query)
        print(f"\nQuery: {args.query}")
        if result.get("success"):
            for match in result.get("matches", []):
                print(f"  - {match['model_name']} (score: {match['score']:.3f})")
        else:
            print(f"  No results: {result.get('error')}")
    
    if args.interactive and pipeline.rag_retriever:
        pipeline.rag_retriever.interactive_query()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
