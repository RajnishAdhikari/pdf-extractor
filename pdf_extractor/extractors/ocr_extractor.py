"""
OCR Extractor Module
Extracts text from scanned PDF pages using PyMuPDF and PaddleOCR.

This module converts PDF pages to high-resolution images and applies OCR
to extract text with bounding box coordinates. Results are saved as
structured JSON files.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

# Import configuration
try:
    from ..config import Config
    from ..utils.file_utils import FileUtils
except ImportError:
    # Handle direct script execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pdf_extractor.config import Config
    from pdf_extractor.utils.file_utils import FileUtils

logger = Config.setup_logging(__name__)


class OCRExtractor:
    """
    Extracts text from scanned PDF pages using OCR.
    
    Uses PyMuPDF to convert PDF pages to images and PaddleOCR
    for text detection and recognition.
    """
    
    def __init__(self, dpi: int = None, language: str = None):
        """
        Initialize the OCR extractor.
        
        Args:
            dpi: Resolution for PDF to image conversion (default from Config)
            language: OCR language (default from Config)
        """
        self.dpi = dpi or Config.OCR_DPI
        self.language = language or Config.OCR_LANGUAGE
        self.ocr_engine = None
        
        logger.info(f"Initializing OCRExtractor with DPI={self.dpi}, language={self.language}")
    
    def _initialize_ocr(self) -> bool:
        """
        Initialize the PaddleOCR engine.
        
        Returns:
            True if successful, False otherwise
        """
        if self.ocr_engine is not None:
            return True
        
        try:
            from paddleocr import PaddleOCR
            
            logger.info("Initializing PaddleOCR engine...")
            self.ocr_engine = PaddleOCR(
                use_angle_cls=True,
                lang=self.language
            )
            logger.info("PaddleOCR engine initialized successfully")
            return True
        except ImportError:
            logger.error("PaddleOCR is not installed. Please install with: pip install paddleocr paddlepaddle")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            return False
    
    def _pdf_page_to_image(self, page: fitz.Page) -> Optional[np.ndarray]:
        """
        Convert a PDF page to a numpy array image.
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Numpy array of the page image (RGB), or None on failure
        """
        try:
            # Calculate the zoom factor from DPI (72 is the base DPI for PDFs)
            zoom = self.dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            
            # Render page to pixmap
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PIL Image then to numpy array
            img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            img_array = np.array(img)
            
            logger.debug(f"Converted page to image: {pixmap.width}x{pixmap.height} pixels")
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to convert page to image: {e}")
            return None
    
    def _extract_text_from_image(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from an image using PaddleOCR.
        
        Args:
            image: Numpy array of the image (RGB)
            
        Returns:
            List of text items with bounding boxes and confidence scores
        """
        if self.ocr_engine is None:
            logger.error("OCR engine not initialized")
            return []
        
        try:
            # Run OCR (cls parameter already set during initialization)
            result = self.ocr_engine.ocr(image)
            
            text_items = []
            
            # Handle different result formats
            if result is None or len(result) == 0:
                logger.debug("No text detected in image")
                return []
            
            # PaddleOCR returns a list of results per page
            for line in result:
                if line is None:
                    continue
                    
                for item in line:
                    try:
                        if item is None or len(item) < 2:
                            continue
                        
                        # Extract bounding box and text info
                        bbox = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                        text_info = item[1]  # (text, confidence) or dict
                        
                        # Handle different text_info formats from newer PaddleOCR
                        if isinstance(text_info, dict):
                            text = text_info.get('text', str(text_info))
                            confidence = float(text_info.get('score', text_info.get('confidence', 0.0)))
                        elif isinstance(text_info, tuple) and len(text_info) >= 2:
                            text = str(text_info[0])
                            try:
                                confidence = float(text_info[1])
                            except (ValueError, TypeError):
                                confidence = 0.0
                        elif isinstance(text_info, list) and len(text_info) >= 2:
                            text = str(text_info[0])
                            try:
                                confidence = float(text_info[1])
                            except (ValueError, TypeError):
                                confidence = 0.0
                        else:
                            text = str(text_info)
                            confidence = 0.0
                        
                        # Convert bbox to list of lists for JSON serialization
                        # Handle potential non-numeric values in bbox
                        bbox_list = []
                        for point in bbox:
                            try:
                                coords = [float(coord) for coord in point]
                                bbox_list.append(coords)
                            except (ValueError, TypeError):
                                # If conversion fails, use zeros
                                bbox_list.append([0.0, 0.0])
                        
                        if text.strip():  # Only add non-empty text
                            text_items.append({
                                "text": text,
                                "bounding_box": bbox_list,
                                "confidence": round(confidence, 4)
                            })
                    except Exception as item_error:
                        logger.debug(f"Skipping malformed OCR item: {item_error}")
                        continue
            
            logger.debug(f"Extracted {len(text_items)} text items from image")
            return text_items
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return []
    
    def extract_page(self, pdf_path: str, page_number: int) -> Optional[Dict[str, Any]]:
        """
        Extract text from a single PDF page.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number (0-indexed)
            
        Returns:
            Dictionary with page data, or None on failure
        """
        try:
            doc = fitz.open(pdf_path)
            
            if page_number < 0 or page_number >= len(doc):
                logger.error(f"Invalid page number: {page_number} (PDF has {len(doc)} pages)")
                doc.close()
                return None
            
            page = doc[page_number]
            
            # Convert page to image
            logger.info(f"Rendering page {page_number + 1} to image...")
            image = self._pdf_page_to_image(page)
            
            if image is None:
                logger.error(f"Failed to render page {page_number + 1}")
                doc.close()
                return None
            
            # Extract text using OCR
            logger.info(f"Running OCR on page {page_number + 1}...")
            text_items = self._extract_text_from_image(image)
            
            # Build result
            result = {
                "page_number": page_number + 1,  # 1-indexed for output
                "width": image.shape[1],
                "height": image.shape[0],
                "dpi": self.dpi,
                "text_items": text_items,
                "text_count": len(text_items)
            }
            
            logger.info(f"Page {page_number + 1}: Extracted {len(text_items)} text items")
            doc.close()
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract page {page_number + 1}: {e}")
            return None
    
    def extract_pdf(
        self,
        pdf_path: str,
        output_dir: str = None,
        start_page: int = None,
        end_page: int = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text from all pages of a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save JSON outputs (default from Config)
            start_page: Starting page number (1-indexed, default: 1)
            end_page: Ending page number (1-indexed, default: last page)
            
        Returns:
            List of page data dictionaries
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir) if output_dir else Config.OCR_OUTPUT_DIR
        
        # Validate PDF path
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        # Initialize OCR engine
        if not self._initialize_ocr():
            logger.error("Cannot proceed without OCR engine")
            return []
        
        # Ensure output directory exists
        FileUtils.ensure_directory(output_dir)
        
        try:
            # Open PDF
            logger.info(f"Opening PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF has {total_pages} pages")
            
            # Determine page range
            start_idx = (start_page - 1) if start_page else 0
            end_idx = end_page if end_page else total_pages
            
            # Validate page range
            start_idx = max(0, min(start_idx, total_pages - 1))
            end_idx = max(start_idx + 1, min(end_idx, total_pages))
            
            logger.info(f"Processing pages {start_idx + 1} to {end_idx}")
            
            results = []
            
            for page_idx in range(start_idx, end_idx):
                page = doc[page_idx]
                
                # Convert page to image
                logger.info(f"Processing page {page_idx + 1}/{total_pages}...")
                image = self._pdf_page_to_image(page)
                
                if image is None:
                    logger.error(f"Failed to render page {page_idx + 1}, skipping...")
                    continue
                
                # Extract text using OCR
                text_items = self._extract_text_from_image(image)
                
                # Build result
                page_data = {
                    "page_number": page_idx + 1,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                    "dpi": self.dpi,
                    "text_items": text_items,
                    "text_count": len(text_items),
                    "source_file": str(pdf_path.name)
                }
                
                # Save to JSON file
                json_path = FileUtils.get_page_json_path(output_dir, page_idx + 1)
                if FileUtils.write_json(page_data, json_path):
                    logger.info(f"Page {page_idx + 1}: Saved {len(text_items)} text items to {json_path.name}")
                else:
                    logger.error(f"Failed to save page {page_idx + 1} data")
                
                results.append(page_data)
            
            doc.close()
            logger.info(f"OCR extraction complete. Processed {len(results)} pages.")
            return results
            
        except fitz.FileDataError as e:
            logger.error(f"Invalid or corrupted PDF file: {e}")
            return []
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return []
    
    def get_all_text(self, page_data: Dict[str, Any]) -> str:
        """
        Get all text from a page data dictionary as a single string.
        
        Args:
            page_data: Page data dictionary from extraction
            
        Returns:
            Concatenated text string
        """
        text_items = page_data.get("text_items", [])
        return " ".join(item.get("text", "") for item in text_items)


def main():
    """Command-line entry point for OCR extraction."""
    parser = argparse.ArgumentParser(
        description="Extract text from scanned PDF pages using OCR"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input PDF file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for JSON files (default: output/ocr_results)"
    )
    parser.add_argument(
        "--start-page",
        type=int,
        default=None,
        help="Starting page number (1-indexed)"
    )
    parser.add_argument(
        "--end-page",
        type=int,
        default=None,
        help="Ending page number (1-indexed)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help=f"DPI for image rendering (default: {Config.OCR_DPI})"
    )
    
    args = parser.parse_args()
    
    # Create extractor and run
    extractor = OCRExtractor(dpi=args.dpi)
    results = extractor.extract_pdf(
        pdf_path=args.input,
        output_dir=args.output,
        start_page=args.start_page,
        end_page=args.end_page
    )
    
    if results:
        total_text_items = sum(r.get("text_count", 0) for r in results)
        logger.info(f"Extraction complete: {len(results)} pages, {total_text_items} text items total")
        return 0
    else:
        logger.error("Extraction failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
