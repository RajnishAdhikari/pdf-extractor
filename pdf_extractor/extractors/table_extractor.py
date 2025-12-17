"""
Table Extractor Module
Extracts tables from PDF pages using multiple methods.
Uses pdfplumber for accurate table detection and extraction.
"""

import pdfplumber
import csv
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..models.extraction_result import TableResult
from ..utils.logger import get_logger
from ..utils.file_utils import ensure_directory, get_output_path, get_pdf_name


class TableExtractor:
    """
    Extracts tables from PDF documents.
    
    Uses pdfplumber for accurate table detection.
    Supports exporting to CSV format.
    """
    
    def __init__(
        self,
        export_csv: bool = True,
        table_settings: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the table extractor.
        
        Args:
            export_csv: Whether to export tables as CSV files
            table_settings: Custom pdfplumber table detection settings
        """
        self.export_csv = export_csv
        self.table_settings = table_settings or {}
        self.logger = get_logger("pdf_extractor.table")
    
    def extract_from_page(
        self,
        page: pdfplumber.page.Page,
        page_number: int,
        output_dir: Optional[str] = None,
        pdf_name: Optional[str] = None
    ) -> List[TableResult]:
        """
        Extract all tables from a single PDF page.
        
        Args:
            page: pdfplumber page object
            page_number: Page number (1-indexed)
            output_dir: Directory to save CSV files (optional)
            pdf_name: Name of the source PDF (optional)
            
        Returns:
            List of TableResult objects
        """
        results = []
        
        try:
            # Find tables on the page
            tables = page.extract_tables(self.table_settings)
            
            for table_idx, table_data in enumerate(tables):
                if not table_data or not any(table_data):
                    continue
                
                # Clean table data
                cleaned_data = self._clean_table_data(table_data)
                
                if not cleaned_data:
                    continue
                
                # Create result
                result = TableResult(
                    data=cleaned_data,
                    page_number=page_number,
                    table_index=table_idx
                )
                
                # Export to CSV if enabled
                if self.export_csv and output_dir and pdf_name:
                    csv_path = self._save_table_csv(
                        cleaned_data,
                        output_dir,
                        pdf_name,
                        page_number,
                        table_idx
                    )
                    result.csv_path = csv_path
                
                results.append(result)
                
        except Exception as e:
            self.logger.error(f"Error extracting tables from page {page_number}: {e}")
        
        return results
    
    def _clean_table_data(
        self,
        table_data: List[List[Optional[str]]]
    ) -> List[List[str]]:
        """
        Clean table data by handling None values and empty cells.
        
        Args:
            table_data: Raw table data from pdfplumber
            
        Returns:
            Cleaned table data
        """
        cleaned = []
        
        for row in table_data:
            if row is None:
                continue
            
            cleaned_row = []
            for cell in row:
                if cell is None:
                    cleaned_row.append("")
                else:
                    # Clean whitespace and newlines
                    cell_text = str(cell).strip()
                    cell_text = " ".join(cell_text.split())
                    cleaned_row.append(cell_text)
            
            # Skip completely empty rows
            if any(cell for cell in cleaned_row):
                cleaned.append(cleaned_row)
        
        return cleaned
    
    def _save_table_csv(
        self,
        table_data: List[List[str]],
        output_dir: str,
        pdf_name: str,
        page_number: int,
        table_idx: int
    ) -> str:
        """
        Save table data to CSV file.
        
        Args:
            table_data: Cleaned table data
            output_dir: Output directory
            pdf_name: PDF filename
            page_number: Page number
            table_idx: Table index
            
        Returns:
            Path to saved CSV file
        """
        csv_path = get_output_path(
            output_dir,
            pdf_name,
            "table",
            page_number,
            table_idx,
            "csv"
        )
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(table_data)
        
        return csv_path
    
    def extract_from_file(
        self,
        pdf_path: str,
        output_dir: Optional[str] = None,
        pages: Optional[List[int]] = None
    ) -> List[TableResult]:
        """
        Extract tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save CSV files
            pages: List of page numbers (1-indexed), None for all
            
        Returns:
            List of TableResult objects
        """
        self.logger.info(f"Extracting tables from: {pdf_path}")
        
        pdf_name = get_pdf_name(pdf_path)
        results = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            if pages is None:
                pages = list(range(1, total_pages + 1))
            
            for page_num in pages:
                if 1 <= page_num <= total_pages:
                    page = pdf.pages[page_num - 1]  # 0-indexed
                    page_results = self.extract_from_page(
                        page, page_num, output_dir, pdf_name
                    )
                    results.extend(page_results)
                    
                    if page_results:
                        self.logger.debug(
                            f"Extracted {len(page_results)} tables from page {page_num}"
                        )
        
        self.logger.info(f"Extracted {len(results)} tables total")
        return results
    
    def detect_tables(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None
    ) -> Dict[int, int]:
        """
        Detect tables in a PDF without extracting data.
        
        Useful for quick analysis of PDF structure.
        
        Args:
            pdf_path: Path to the PDF file
            pages: List of page numbers to check
            
        Returns:
            Dictionary mapping page numbers to table counts
        """
        table_counts = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            
            if pages is None:
                pages = list(range(1, total_pages + 1))
            
            for page_num in pages:
                if 1 <= page_num <= total_pages:
                    page = pdf.pages[page_num - 1]
                    tables = page.find_tables(self.table_settings)
                    table_counts[page_num] = len(tables)
        
        return table_counts
