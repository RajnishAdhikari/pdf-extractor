"""
Extractors package for OCR and table detection.
"""

from .ocr_extractor import OCRExtractor
from .table_detector import TableDetector

__all__ = ["OCRExtractor", "TableDetector"]
