"""
Text Extractor Module
Extracts text content from PDF pages using PyMuPDF (fitz).
Fast and efficient text extraction with page number tracking.
"""

import fitz  # PyMuPDF
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models.extraction_result import TextResult
from ..utils.logger import get_logger


class TextExtractor:
    """
    Extracts text content from PDF documents.
    
    Uses PyMuPDF for fast and accurate text extraction.
    Supports both single-page and batch extraction.
    """
    
    def __init__(self, preserve_layout: bool = True):
        """
        Initialize the text extractor.
        
        Args:
            preserve_layout: Whether to preserve text layout (default: True)
        """
        self.preserve_layout = preserve_layout
        self.logger = get_logger("pdf_extractor.text")
    
    def extract_from_page(
        self,
        page: fitz.Page,
        page_number: int
    ) -> TextResult:
        """
        Extract text from a single PDF page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-indexed)
            
        Returns:
            TextResult containing extracted text
        """
        try:
            # Extract text with layout preservation option
            if self.preserve_layout:
                text = page.get_text("text")
            else:
                text = page.get_text("text", sort=True)
            
            # Clean up the text
            text = text.strip()
            
            return TextResult(
                content=text,
                page_number=page_number
            )
        except Exception as e:
            self.logger.error(f"Error extracting text from page {page_number}: {e}")
            return TextResult(content="", page_number=page_number)
    
    def extract_from_document(
        self,
        doc: fitz.Document,
        pages: Optional[List[int]] = None,
        max_workers: int = 4
    ) -> List[TextResult]:
        """
        Extract text from multiple pages in a document.
        
        Args:
            doc: PyMuPDF document object
            pages: List of page numbers to extract (1-indexed), None for all pages
            max_workers: Number of parallel workers
            
        Returns:
            List of TextResult objects
        """
        if pages is None:
            pages = list(range(1, len(doc) + 1))
        
        results = []
        
        # Process pages (PyMuPDF is not thread-safe for page access)
        for page_num in pages:
            if 1 <= page_num <= len(doc):
                page = doc[page_num - 1]  # Convert to 0-indexed
                result = self.extract_from_page(page, page_num)
                results.append(result)
                self.logger.debug(f"Extracted text from page {page_num}: {result.word_count} words")
        
        return results
    
    def extract_from_file(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> List[TextResult]:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract (1-indexed), None for all pages
            
        Returns:
            List of TextResult objects
        """
        self.logger.info(f"Extracting text from: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        try:
            results = self.extract_from_document(doc, pages)
            self.logger.info(f"Extracted text from {len(results)} pages")
            return results
        finally:
            doc.close()
    
    def extract_blocks(
        self,
        page: fitz.Page,
        page_number: int
    ) -> List[TextResult]:
        """
        Extract text blocks from a page (preserves block structure).
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-indexed)
            
        Returns:
            List of TextResult objects for each text block
        """
        results = []
        try:
            blocks = page.get_text("blocks")
            for idx, block in enumerate(blocks):
                # Block format: (x0, y0, x1, y1, "text", block_no, block_type)
                if block[6] == 0:  # Type 0 is text
                    text = block[4].strip()
                    if text:
                        results.append(TextResult(
                            content=text,
                            page_number=page_number
                        ))
        except Exception as e:
            self.logger.error(f"Error extracting text blocks from page {page_number}: {e}")
        
        return results
