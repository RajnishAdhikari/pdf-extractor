"""
PDF Extractor Package
A modular, open-source PDF extraction toolkit for extracting text, images, and tables.
"""

from .core.extractor import PDFExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.image_extractor import ImageExtractor
from .extractors.table_extractor import TableExtractor

__version__ = "1.0.0"
__all__ = ["PDFExtractor", "TextExtractor", "ImageExtractor", "TableExtractor"]
