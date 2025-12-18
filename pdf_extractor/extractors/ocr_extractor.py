"""
OCR Extractor Module (Tesseract-based)
Extracts text from scanned PDFs using Tesseract OCR.
Open-source, no API required, works offline.
"""

import fitz  # PyMuPDF
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import io

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from ..models.extraction_result import TextResult
from ..utils.logger import get_logger
from ..utils.file_utils import ensure_directory, get_pdf_name


class OCRExtractor:
    """
    Extracts text from scanned PDFs using Tesseract OCR.
    
    Converts PDF pages to images internally, then runs OCR.
    Supports multiple languages and batch processing.
    """
    
    def __init__(
        self,
        lang: str = 'eng',
        config: str = '',
        tesseract_cmd: Optional[str] = None,
        dpi: int = 300
    ):
        """
        Initialize the OCR extractor.
        
        Args:
            lang: Tesseract language code (default: 'eng')
            config: Additional Tesseract config options
            tesseract_cmd: Path to Tesseract executable (if not in PATH)
            dpi: Resolution for PDF to image conversion (default: 300)
        """
        self.lang = lang
        self.config = config
        self.dpi = dpi
        self.logger = get_logger("pdf_extractor.ocr")
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        if not TESSERACT_AVAILABLE:
            self.logger.warning(
                "pytesseract not installed. Install with: pip install pytesseract"
            )
    
    def _pdf_page_to_image(self, page: fitz.Page) -> Image.Image:
        """
        Convert a PDF page to a PIL Image.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            PIL Image object
        """
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    
    def extract_from_page(
        self,
        page: fitz.Page,
        page_number: int
    ) -> TextResult:
        """
        Extract text from a single PDF page using OCR.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number to assign
            
        Returns:
            TextResult containing extracted text
        """
        if not TESSERACT_AVAILABLE:
            return TextResult(content="[OCR not available]", page_number=page_number)
        
        try:
            self.logger.info(f"Running OCR on page {page_number}")
            
            # Convert PDF page to image
            image = self._pdf_page_to_image(page)
            
            # Run OCR
            text = pytesseract.image_to_string(
                image,
                lang=self.lang,
                config=self.config
            )
            
            return TextResult(
                content=text.strip(),
                page_number=page_number
            )
            
        except Exception as e:
            self.logger.error(f"OCR failed for page {page_number}: {e}")
            return TextResult(content="", page_number=page_number)
    
    def extract_with_positions(
        self,
        page: fitz.Page,
        page_number: int
    ) -> Dict[str, Any]:
        """
        Extract text with bounding box positions from a PDF page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number
            
        Returns:
            Dictionary with text blocks and positions
        """
        if not TESSERACT_AVAILABLE:
            return {"page_number": page_number, "blocks": []}
        
        try:
            image = self._pdf_page_to_image(page)
            
            # Get detailed data
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=self.config,
                output_type=pytesseract.Output.DICT
            )
            
            blocks = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                
                if text and conf > 30:  # Filter low confidence
                    blocks.append({
                        "text": text,
                        "confidence": conf / 100.0,
                        "bbox": {
                            "x1": data['left'][i],
                            "y1": data['top'][i],
                            "x2": data['left'][i] + data['width'][i],
                            "y2": data['top'][i] + data['height'][i]
                        },
                        "level": data['level'][i],
                        "line_num": data['line_num'][i]
                    })
            
            return {
                "page_number": page_number,
                "total_blocks": len(blocks),
                "blocks": blocks
            }
            
        except Exception as e:
            self.logger.error(f"OCR with positions failed: {e}")
            return {"page_number": page_number, "blocks": []}
    
    def extract_from_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        output_dir: Optional[str] = None
    ) -> List[TextResult]:
        """
        Extract text from a PDF file using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to extract (1-indexed), None for all
            output_dir: Optional directory to save results
            
        Returns:
            List of TextResult objects
        """
        self.logger.info(f"Running OCR on PDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if pages is None:
            pages = list(range(1, total_pages + 1))
        
        results = []
        
        for page_num in pages:
            if 1 <= page_num <= total_pages:
                self.logger.info(f"Processing page [{page_num}/{total_pages}]")
                page = doc[page_num - 1]
                result = self.extract_from_page(page, page_num)
                results.append(result)
        
        doc.close()
        
        # Save results if output directory specified
        if output_dir:
            self._save_results(results, output_dir, pdf_path)
        
        self.logger.info(f"OCR complete: {len(results)} pages processed")
        return results
    
    def extract_with_positions_from_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text with positions from a PDF file using OCR.
        
        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers (1-indexed), None for all
            output_dir: Optional directory to save results
            
        Returns:
            List of dictionaries with text blocks and positions
        """
        self.logger.info(f"Running OCR with positions on PDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if pages is None:
            pages = list(range(1, total_pages + 1))
        
        results = []
        
        for page_num in pages:
            if 1 <= page_num <= total_pages:
                self.logger.info(f"Processing page [{page_num}/{total_pages}]")
                page = doc[page_num - 1]
                result = self.extract_with_positions(page, page_num)
                results.append(result)
        
        doc.close()
        
        # Save results if output directory specified
        if output_dir:
            ensure_directory(output_dir)
            pdf_name = get_pdf_name(pdf_path)
            output_path = Path(output_dir) / pdf_name / "ocr_positions.json"
            ensure_directory(output_path.parent)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved OCR position results to: {output_path}")
        
        return results
    
    def _save_results(
        self,
        results: List[TextResult],
        output_dir: str,
        pdf_path: str
    ) -> None:
        """
        Save OCR results to files.
        
        Args:
            results: List of TextResult objects
            output_dir: Output directory
            pdf_path: Source PDF path
        """
        ensure_directory(output_dir)
        pdf_name = get_pdf_name(pdf_path)
        output_folder = Path(output_dir) / pdf_name
        ensure_directory(output_folder)
        
        # Save combined text
        combined_path = output_folder / "ocr_text.txt"
        with open(combined_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"=== Page {result.page_number} ===\n")
                f.write(result.content)
                f.write("\n\n")
        
        # Save JSON
        json_path = output_folder / "ocr_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved OCR results to: {output_folder}")
    
    def extract_tables_ocr(
        self,
        page: fitz.Page,
        page_number: int
    ) -> List[Dict[str, Any]]:
        """
        Attempt to extract table-like structures from OCR results.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number
            
        Returns:
            List of potential table rows
        """
        result = self.extract_with_positions(page, page_number)
        
        if not result['blocks']:
            return []
        
        # Group blocks by line number
        lines = {}
        for block in result['blocks']:
            line_num = block.get('line_num', 0)
            if line_num not in lines:
                lines[line_num] = []
            lines[line_num].append(block)
        
        # Sort each line by X position
        sorted_rows = []
        for line_num in sorted(lines.keys()):
            cells = sorted(lines[line_num], key=lambda b: b['bbox']['x1'])
            row_text = [c['text'] for c in cells]
            sorted_rows.append({
                "line_num": line_num,
                "cells": row_text
            })
        
        return sorted_rows
