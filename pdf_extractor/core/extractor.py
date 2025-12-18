"""
Core PDF Extractor Module
Main orchestrator for extracting all content from PDF files.
Combines text, image, and table extraction in a unified interface.
"""

import fitz  # PyMuPDF
import pdfplumber
import time
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..extractors.text_extractor import TextExtractor
from ..extractors.image_extractor import ImageExtractor
from ..extractors.table_extractor import TableExtractor
from ..models.extraction_result import (
    ExtractionResult,
    PageContent,
    TextResult,
    ImageResult,
    TableResult
)
from ..utils.logger import get_logger
from ..utils.file_utils import (
    ensure_directory,
    get_pdf_name,
    validate_pdf_path
)


class PDFExtractor:
    """
    Main PDF extraction orchestrator.
    
    Combines text, image, and table extraction into a single,
    easy-to-use interface with comprehensive output.
    
    Example:
        extractor = PDFExtractor(output_dir="./extracted")
        result = extractor.extract("document.pdf")
        result.save_json("output.json")
    """
    
    def __init__(
        self,
        output_dir: str = "./extracted_content",
        extract_text: bool = True,
        extract_images: bool = True,
        extract_tables: bool = True,
        min_image_size: int = 50,
        export_tables_csv: bool = True,
        preserve_text_layout: bool = True
    ):
        """
        Initialize the PDF extractor.
        
        Args:
            output_dir: Directory to save extracted content
            extract_text: Whether to extract text
            extract_images: Whether to extract images
            extract_tables: Whether to extract tables
            min_image_size: Minimum image dimension to extract
            export_tables_csv: Whether to export tables as CSV
            preserve_text_layout: Whether to preserve text layout
        """
        self.output_dir = ensure_directory(output_dir)
        self.extract_text = extract_text
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        
        self.logger = get_logger("pdf_extractor")
        
        # Initialize extractors
        if self.extract_text:
            self.text_extractor = TextExtractor(
                preserve_layout=preserve_text_layout
            )
        
        if self.extract_images:
            self.image_extractor = ImageExtractor(
                min_width=min_image_size,
                min_height=min_image_size
            )
        
        if self.extract_tables:
            self.table_extractor = TableExtractor(
                export_csv=export_tables_csv
            )
    
    def extract(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        save_json: bool = True
    ) -> ExtractionResult:
        """
        Extract all content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract (1-indexed), None for all
            save_json: Whether to save results to JSON
            
        Returns:
            ExtractionResult containing all extracted content
        """
        start_time = time.time()
        
        # Validate PDF
        validate_pdf_path(pdf_path)
        pdf_name = get_pdf_name(pdf_path)
        
        self.logger.info(f"Starting extraction from: {pdf_path}")
        
        # Get total pages
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if pages is None:
            pages = list(range(1, total_pages + 1))
        
        # Extract text using PyMuPDF
        text_results = {}
        if self.extract_text:
            for page_num in pages:
                if 1 <= page_num <= total_pages:
                    page = doc[page_num - 1]
                    text_result = self.text_extractor.extract_from_page(
                        page, page_num
                    )
                    text_results[page_num] = [text_result]
        
        # Extract images using PyMuPDF
        image_results = {}
        if self.extract_images:
            for page_num in pages:
                if 1 <= page_num <= total_pages:
                    page = doc[page_num - 1]
                    images = self.image_extractor.extract_from_page(
                        page, page_num, self.output_dir, pdf_name
                    )
                    image_results[page_num] = images
        
        doc.close()
        
        # Extract tables using pdfplumber
        table_results = {}
        if self.extract_tables:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in pages:
                    if 1 <= page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        tables = self.table_extractor.extract_from_page(
                            page, page_num, self.output_dir, pdf_name
                        )
                        table_results[page_num] = tables
        
        # Combine results by page
        page_contents = []
        for page_num in pages:
            page_content = PageContent(
                page_number=page_num,
                texts=text_results.get(page_num, []),
                images=image_results.get(page_num, []),
                tables=table_results.get(page_num, [])
            )
            page_contents.append(page_content)
        
        # Calculate extraction time
        extraction_time = time.time() - start_time
        
        # Get PDF metadata
        metadata = self._get_pdf_metadata(pdf_path)
        
        # Create result
        result = ExtractionResult(
            pdf_path=pdf_path,
            total_pages=total_pages,
            pages=page_contents,
            extraction_time=extraction_time,
            metadata=metadata
        )
        
        # Log summary
        self._log_summary(result)
        
        # Save JSON if requested
        if save_json:
            json_path = Path(self.output_dir) / pdf_name / "extraction_result.json"
            ensure_directory(json_path.parent)
            result.save_json(str(json_path))
            self.logger.info(f"Saved extraction results to: {json_path}")
        
        return result
    
    def _get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract PDF metadata."""
        metadata = {}
        
        try:
            doc = fitz.open(pdf_path)
            meta = doc.metadata
            
            if meta:
                metadata = {
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                    "subject": meta.get("subject", ""),
                    "keywords": meta.get("keywords", ""),
                    "creator": meta.get("creator", ""),
                    "producer": meta.get("producer", ""),
                    "creation_date": meta.get("creationDate", ""),
                    "modification_date": meta.get("modDate", "")
                }
            
            metadata["file_size_bytes"] = Path(pdf_path).stat().st_size
            metadata["page_count"] = len(doc)
            
            doc.close()
        except Exception as e:
            self.logger.warning(f"Could not extract metadata: {e}")
        
        return metadata
    
    def _log_summary(self, result: ExtractionResult) -> None:
        """Log extraction summary."""
        summary = result.to_dict()["summary"]
        self.logger.info(
            f"Extraction complete in {result.extraction_time:.2f}s - "
            f"Text blocks: {summary['total_text_blocks']}, "
            f"Images: {summary['total_images']}, "
            f"Tables: {summary['total_tables']}"
        )
    
    def extract_text_only(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> List[TextResult]:
        """Extract only text from a PDF."""
        return self.text_extractor.extract_from_file(pdf_path, pages)
    
    def extract_images_only(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> List[ImageResult]:
        """Extract only images from a PDF."""
        return self.image_extractor.extract_from_pdf(
            pdf_path, self.output_dir, pages
        )
    
    def extract_tables_only(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> List[TableResult]:
        """Extract only tables from a PDF."""
        return self.table_extractor.extract_from_file(
            pdf_path, self.output_dir, pages
        )
    
    def analyze(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze a PDF without extracting content."""
        validate_pdf_path(pdf_path)
        
        result = {
            "pdf_path": pdf_path,
            "metadata": self._get_pdf_metadata(pdf_path),
            "pages": []
        }
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_info = {
                "page_number": page_num + 1,
                "text_length": len(page.get_text()),
                "image_count": len(page.get_images()),
                "width": page.rect.width,
                "height": page.rect.height
            }
            result["pages"].append(page_info)
        
        doc.close()
        
        # Add table detection
        if self.extract_tables:
            table_counts = self.table_extractor.detect_tables(pdf_path)
            for page_info in result["pages"]:
                page_num = page_info["page_number"]
                page_info["table_count"] = table_counts.get(page_num, 0)
        
        return result
