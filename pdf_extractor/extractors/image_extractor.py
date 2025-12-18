"""
Image Extractor Module
Extracts embedded images from PDF pages using PyMuPDF (fitz).
Fast image extraction with PNG format output.
"""

import fitz  # PyMuPDF
import os
from typing import List, Optional
from PIL import Image
import io

from ..models.extraction_result import ImageResult
from ..utils.logger import get_logger
from ..utils.file_utils import ensure_directory, get_output_path, get_pdf_name


class ImageExtractor:
    """
    Extracts embedded images from PDF documents.
    
    Uses PyMuPDF for fast image extraction.
    Saves images in PNG format.
    """
    
    def __init__(
        self,
        min_width: int = 50,
        min_height: int = 50
    ):
        """
        Initialize the image extractor.
        
        Args:
            min_width: Minimum image width to extract (filters out small images)
            min_height: Minimum image height to extract
        """
        self.min_width = min_width
        self.min_height = min_height
        self.logger = get_logger("pdf_extractor.image")
    
    def extract_from_page(
        self,
        page: fitz.Page,
        page_number: int,
        output_dir: str,
        pdf_name: str
    ) -> List[ImageResult]:
        """
        Extract all images from a single PDF page.
        
        Args:
            page: PyMuPDF page object
            page_number: Page number (1-indexed)
            output_dir: Directory to save extracted images
            pdf_name: Name of the source PDF
            
        Returns:
            List of ImageResult objects
        """
        results = []
        
        try:
            # Get list of images on the page
            image_list = page.get_images(full=True)
            
            for img_idx, img_info in enumerate(image_list):
                try:
                    result = self._extract_single_image(
                        page.parent,  # Document
                        img_info,
                        page_number,
                        img_idx,
                        output_dir,
                        pdf_name
                    )
                    if result:
                        results.append(result)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to extract image {img_idx} from page {page_number}: {e}"
                    )
        except Exception as e:
            self.logger.error(f"Error extracting images from page {page_number}: {e}")
        
        return results
    
    def _extract_single_image(
        self,
        doc: fitz.Document,
        img_info: tuple,
        page_number: int,
        img_idx: int,
        output_dir: str,
        pdf_name: str
    ) -> Optional[ImageResult]:
        """
        Extract a single image from the document.
        
        Args:
            doc: PyMuPDF document object
            img_info: Image info tuple from get_images()
            page_number: Page number
            img_idx: Image index on the page
            output_dir: Output directory
            pdf_name: PDF filename
            
        Returns:
            ImageResult or None if extraction failed
        """
        xref = img_info[0]
        
        # Extract image data
        base_image = doc.extract_image(xref)
        
        if not base_image:
            return None
        
        image_bytes = base_image["image"]
        width = base_image.get("width", 0)
        height = base_image.get("height", 0)
        
        # Filter small images
        if width < self.min_width or height < self.min_height:
            self.logger.debug(
                f"Skipping small image ({width}x{height}) on page {page_number}"
            )
            return None
        
        # Convert to PNG format
        try:
            img = Image.open(io.BytesIO(image_bytes))
            buffer = io.BytesIO()
            
            # Handle RGBA
            if img.mode == "RGBA":
                img = img.convert("RGB")
            
            img.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        except Exception as e:
            self.logger.warning(f"Could not convert image to PNG: {e}")
        
        # Generate output path
        output_path = get_output_path(
            output_dir,
            pdf_name,
            "image",
            page_number,
            img_idx,
            "png"
        )
        
        # Save image
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        
        return ImageResult(
            image_path=output_path,
            page_number=page_number,
            image_index=img_idx,
            width=width,
            height=height,
            format="png",
            size_bytes=len(image_bytes)
        )
    
    def extract_from_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        pages: Optional[List[int]] = None
    ) -> List[ImageResult]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            pages: List of page numbers (1-indexed), None for all
            
        Returns:
            List of ImageResult objects
        """
        self.logger.info(f"Extracting images from: {pdf_path}")
        
        pdf_name = get_pdf_name(pdf_path)
        doc = fitz.open(pdf_path)
        
        if pages is None:
            pages = list(range(1, len(doc) + 1))
        
        results = []
        
        for page_num in pages:
            if 1 <= page_num <= len(doc):
                page = doc[page_num - 1]
                page_results = self.extract_from_page(
                    page, page_num, output_dir, pdf_name
                )
                results.extend(page_results)
                
                if page_results:
                    self.logger.debug(
                        f"Extracted {len(page_results)} images from page {page_num}"
                    )
        
        doc.close()
        self.logger.info(f"Extracted {len(results)} images total")
        return results
