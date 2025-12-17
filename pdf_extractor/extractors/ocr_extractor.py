"""
OCR Extractor Module (Tesseract-based)
Extracts text from scanned PDFs and images using Tesseract OCR.
Open-source, no API required, works offline.
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import re

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
    Extracts text from images and scanned PDFs using Tesseract OCR.
    
    Uses pytesseract for accurate text recognition.
    Supports multiple languages and batch processing.
    """
    
    def __init__(
        self,
        lang: str = 'eng',
        config: str = '',
        tesseract_cmd: Optional[str] = None
    ):
        """
        Initialize the OCR extractor.
        
        Args:
            lang: Tesseract language code (default: 'eng')
            config: Additional Tesseract config options
            tesseract_cmd: Path to Tesseract executable (if not in PATH)
        """
        self.lang = lang
        self.config = config
        self.logger = get_logger("pdf_extractor.ocr")
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        if not TESSERACT_AVAILABLE:
            self.logger.warning(
                "pytesseract not installed. Install with: pip install pytesseract"
            )
    
    def extract_from_image(
        self,
        image_path: str,
        page_number: int = 1
    ) -> TextResult:
        """
        Extract text from a single image.
        
        Args:
            image_path: Path to the image file
            page_number: Page number to assign
            
        Returns:
            TextResult containing extracted text
        """
        if not TESSERACT_AVAILABLE:
            return TextResult(content="[OCR not available]", page_number=page_number)
        
        try:
            self.logger.info(f"Running OCR on: {image_path}")
            
            # Open image
            image = Image.open(image_path)
            
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
            self.logger.error(f"OCR failed for {image_path}: {e}")
            return TextResult(content="", page_number=page_number)
    
    def extract_with_positions(
        self,
        image_path: str,
        page_number: int = 1
    ) -> Dict[str, Any]:
        """
        Extract text with bounding box positions.
        
        Args:
            image_path: Path to image file
            page_number: Page number
            
        Returns:
            Dictionary with text blocks and positions
        """
        if not TESSERACT_AVAILABLE:
            return {"page_number": page_number, "blocks": []}
        
        try:
            image = Image.open(image_path)
            
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
                "image_path": image_path,
                "total_blocks": len(blocks),
                "blocks": blocks
            }
            
        except Exception as e:
            self.logger.error(f"OCR with positions failed: {e}")
            return {"page_number": page_number, "blocks": []}
    
    def extract_from_folder(
        self,
        folder_path: str,
        output_dir: Optional[str] = None,
        extensions: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract text from all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            output_dir: Optional directory to save results
            extensions: Image extensions to process
            
        Returns:
            List of extraction results per image
        """
        extensions = extensions or ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        folder = Path(folder_path)
        
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        image_files = sorted(set(image_files))
        self.logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        
        for idx, image_path in enumerate(image_files, 1):
            self.logger.info(f"Processing [{idx}/{len(image_files)}]: {image_path.name}")
            
            result = self.extract_with_positions(str(image_path), page_number=idx)
            result["filename"] = image_path.name
            results.append(result)
        
        # Save results
        if output_dir:
            ensure_directory(output_dir)
            output_path = Path(output_dir) / "ocr_results.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved OCR results to: {output_path}")
            
            # Save text files
            for result in results:
                text_file = Path(output_dir) / f"{Path(result['filename']).stem}_text.txt"
                all_text = "\n".join([b['text'] for b in result['blocks']])
                
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"Page {result['page_number']}: {result['filename']}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(all_text)
        
        return results
    
    def extract_tables_ocr(
        self,
        image_path: str,
        page_number: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Attempt to extract table-like structures from OCR results.
        
        Args:
            image_path: Path to image
            page_number: Page number
            
        Returns:
            List of potential table rows
        """
        result = self.extract_with_positions(image_path, page_number)
        
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
