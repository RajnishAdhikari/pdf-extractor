"""
File utility functions for PDF extraction.
Handles file and directory operations.
"""

import os
import re
from pathlib import Path
from typing import Optional


def ensure_directory(directory_path: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        The absolute path of the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path.absolute())


def get_output_path(
    base_dir: str,
    pdf_name: str,
    content_type: str,
    page_number: int,
    index: int = 0,
    extension: str = ""
) -> str:
    """
    Generate a standardized output path for extracted content.
    
    Args:
        base_dir: Base output directory
        pdf_name: Name of the source PDF (without extension)
        content_type: Type of content (text, image, table)
        page_number: Page number in the PDF
        index: Index of the item on the page
        extension: File extension (with or without dot)
        
    Returns:
        Full path to the output file
    """
    # Ensure extension has a dot
    if extension and not extension.startswith('.'):
        extension = f'.{extension}'
    
    # Create subdirectory for content type
    content_dir = os.path.join(base_dir, pdf_name, content_type + 's')
    ensure_directory(content_dir)
    
    # Generate filename
    filename = f"page_{page_number:04d}_{content_type}_{index:03d}{extension}"
    
    return os.path.join(content_dir, filename)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename by removing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized or 'unnamed'


def get_pdf_name(pdf_path: str) -> str:
    """
    Extract the PDF filename without extension.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        PDF filename without extension
    """
    return sanitize_filename(Path(pdf_path).stem)


def validate_pdf_path(pdf_path: str) -> bool:
    """
    Validate that a PDF file exists and is readable.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if valid, raises exception otherwise
    """
    path = Path(pdf_path)
    
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if not path.is_file():
        raise ValueError(f"Path is not a file: {pdf_path}")
    
    if path.suffix.lower() != '.pdf':
        raise ValueError(f"File is not a PDF: {pdf_path}")
    
    return True
