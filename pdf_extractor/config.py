"""
Configuration settings for the PDF Extraction Pipeline.
"""

import os
import logging
from pathlib import Path


class Config:
    """Configuration class for PDF extraction pipeline."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / "output"
    
    # Output subdirectories
    OCR_OUTPUT_DIR = OUTPUT_DIR / "ocr_results"
    TABLE_OUTPUT_DIR = OUTPUT_DIR / "table_results"
    GROUPED_OUTPUT_DIR = OUTPUT_DIR / "grouped_by_model"
    RAG_INDEX_DIR = OUTPUT_DIR / "rag_index"
    
    # OCR Settings
    OCR_DPI = 300  # DPI for PDF to image conversion
    OCR_LANGUAGE = "en"  # OCR language
    
    # Table Detection Settings
    TABLE_DETECTION_MODEL = "microsoft/table-transformer-detection"
    TABLE_STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition"
    TABLE_DETECTION_THRESHOLD = 0.7
    
    # RAG Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    TOP_K_RESULTS = 5
    
    # Model identifier patterns (regex patterns for part numbers)
    MODEL_PATTERNS = [
        r'\b[A-Z]{2,4}\d{1,4}[A-Z]?\b',  # e.g., CF12, ABC123, CFW24B
        r'\b[A-Z]{1,3}-\d{1,4}[A-Z]?\b',  # e.g., CF-12, A-123
        r'\b\d{1,4}[A-Z]{2,4}\d{0,4}\b',  # e.g., 12CF, 123ABC45
    ]
    
    # Logging configuration
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    @classmethod
    def setup_logging(cls, name: str = None) -> logging.Logger:
        """
        Setup and return a configured logger.
        
        Args:
            name: Logger name (defaults to root logger if None)
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                cls.LOG_FORMAT,
                datefmt=cls.LOG_DATE_FORMAT
            ))
            logger.addHandler(handler)
            logger.setLevel(cls.LOG_LEVEL)
        
        return logger
    
    @classmethod
    def ensure_directories(cls):
        """Create all necessary output directories."""
        for dir_path in [
            cls.OUTPUT_DIR,
            cls.OCR_OUTPUT_DIR,
            cls.TABLE_OUTPUT_DIR,
            cls.GROUPED_OUTPUT_DIR,
            cls.RAG_INDEX_DIR
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
