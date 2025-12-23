"""
Table Detector Module
Detects and extracts structured tables from PDF pages using Table Transformer.

This module uses the Microsoft Table Transformer model to identify table regions
and their structure, then populates cells with OCR-extracted text.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Import configuration
try:
    from ..config import Config
    from ..utils.file_utils import FileUtils
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pdf_extractor.config import Config
    from pdf_extractor.utils.file_utils import FileUtils

logger = Config.setup_logging(__name__)


class TableDetector:
    """
    Detects and extracts tables from PDF page images.
    
    Uses Microsoft Table Transformer for table detection and structure
    recognition, then populates cells with OCR text based on coordinates.
    """
    
    def __init__(
        self,
        detection_threshold: float = None,
        detection_model: str = None,
        structure_model: str = None
    ):
        """
        Initialize the table detector.
        
        Args:
            detection_threshold: Confidence threshold for table detection
            detection_model: Hugging Face model ID for table detection
            structure_model: Hugging Face model ID for structure recognition
        """
        self.detection_threshold = detection_threshold or Config.TABLE_DETECTION_THRESHOLD
        self.detection_model_name = detection_model or Config.TABLE_DETECTION_MODEL
        self.structure_model_name = structure_model or Config.TABLE_STRUCTURE_MODEL
        
        self.detection_model = None
        self.detection_processor = None
        self.structure_model = None
        self.structure_processor = None
        self.device = None
        
        logger.info(f"Initializing TableDetector with threshold={self.detection_threshold}")
    
    def _initialize_models(self) -> bool:
        """
        Initialize the Table Transformer models.
        
        Returns:
            True if successful, False otherwise
        """
        if self.detection_model is not None:
            return True
        
        try:
            import torch
            from transformers import (
                AutoModelForObjectDetection,
                AutoImageProcessor
            )
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load detection model
            logger.info(f"Loading table detection model: {self.detection_model_name}")
            self.detection_processor = AutoImageProcessor.from_pretrained(
                self.detection_model_name
            )
            self.detection_model = AutoModelForObjectDetection.from_pretrained(
                self.detection_model_name
            ).to(self.device)
            
            # Load structure recognition model
            logger.info(f"Loading table structure model: {self.structure_model_name}")
            self.structure_processor = AutoImageProcessor.from_pretrained(
                self.structure_model_name
            )
            self.structure_model = AutoModelForObjectDetection.from_pretrained(
                self.structure_model_name
            ).to(self.device)
            
            logger.info("Table Transformer models loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Required libraries not installed: {e}")
            logger.error("Please install: pip install transformers torch torchvision timm")
            return False
        except Exception as e:
            logger.error(f"Failed to load Table Transformer models: {e}")
            return False
    
    def _detect_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect table regions in an image.
        
        Args:
            image: PIL Image of the page
            
        Returns:
            List of detected tables with bounding boxes
        """
        import torch
        
        try:
            # Prepare image for model
            inputs = self.detection_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run detection
            with torch.no_grad():
                outputs = self.detection_model(**inputs)
            
            # Post-process results
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detection_processor.post_process_object_detection(
                outputs,
                threshold=self.detection_threshold,
                target_sizes=target_sizes
            )[0]
            
            tables = []
            for score, label, box in zip(
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy(),
                results["boxes"].cpu().numpy()
            ):
                # Only keep table detections (label 0 is typically "table")
                tables.append({
                    "bounding_box": [float(x) for x in box],  # [x1, y1, x2, y2]
                    "confidence": float(score),
                    "label": int(label)
                })
            
            logger.debug(f"Detected {len(tables)} tables in image")
            return tables
            
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []
    
    def _recognize_structure(
        self,
        image: Image.Image,
        table_box: List[float]
    ) -> Dict[str, Any]:
        """
        Recognize the structure (rows/columns) of a table.
        
        Args:
            image: PIL Image of the full page
            table_box: Bounding box of the table [x1, y1, x2, y2]
            
        Returns:
            Dictionary with structure information (rows, columns, cells)
        """
        import torch
        
        try:
            # Crop table region from image
            x1, y1, x2, y2 = [int(x) for x in table_box]
            table_image = image.crop((x1, y1, x2, y2))
            
            # Prepare for model
            inputs = self.structure_processor(images=table_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run structure recognition
            with torch.no_grad():
                outputs = self.structure_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([table_image.size[::-1]]).to(self.device)
            results = self.structure_processor.post_process_object_detection(
                outputs,
                threshold=0.5,
                target_sizes=target_sizes
            )[0]
            
            # Categorize detected elements
            rows = []
            columns = []
            cells = []
            
            # Common label mappings for structure model
            label_names = {
                0: "table",
                1: "table column",
                2: "table row",
                3: "table column header",
                4: "table projected row header",
                5: "table spanning cell"
            }
            
            for score, label, box in zip(
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy(),
                results["boxes"].cpu().numpy()
            ):
                item = {
                    "box": [float(x) for x in box],
                    "confidence": float(score),
                    "type": label_names.get(int(label), f"unknown_{label}")
                }
                
                if label == 2:  # row
                    rows.append(item)
                elif label == 1:  # column
                    columns.append(item)
                else:
                    cells.append(item)
            
            # Sort rows by y-coordinate (top to bottom)
            rows.sort(key=lambda r: r["box"][1])
            
            # Sort columns by x-coordinate (left to right)
            columns.sort(key=lambda c: c["box"][0])
            
            return {
                "rows": rows,
                "columns": columns,
                "cells": cells,
                "row_count": len(rows),
                "column_count": len(columns)
            }
            
        except Exception as e:
            logger.error(f"Structure recognition failed: {e}")
            return {"rows": [], "columns": [], "cells": [], "row_count": 0, "column_count": 0}
    
    def _assign_text_to_cells(
        self,
        text_items: List[Dict[str, Any]],
        table_box: List[float],
        structure: Dict[str, Any]
    ) -> List[List[str]]:
        """
        Assign OCR text items to table cells based on coordinates.
        
        Args:
            text_items: List of OCR text items with bounding boxes
            table_box: Bounding box of the table [x1, y1, x2, y2]
            structure: Structure information from recognition
            
        Returns:
            2D list representing table content (rows x columns)
        """
        rows = structure.get("rows", [])
        columns = structure.get("columns", [])
        
        if not rows or not columns:
            # Fallback: try to extract text that falls within the table
            logger.warning("No row/column structure detected, using fallback extraction")
            return self._fallback_text_extraction(text_items, table_box)
        
        tx1, ty1, tx2, ty2 = table_box
        
        # Create cell grid
        num_rows = len(rows)
        num_cols = len(columns)
        cell_grid = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
        
        # Assign text to cells
        for text_item in text_items:
            bbox = text_item.get("bounding_box", [])
            if not bbox or len(bbox) < 4:
                continue
            
            # Get center of text bounding box
            text_center_x = sum(p[0] for p in bbox) / 4
            text_center_y = sum(p[1] for p in bbox) / 4
            
            # Check if text is within table bounds
            if not (tx1 <= text_center_x <= tx2 and ty1 <= text_center_y <= ty2):
                continue
            
            # Adjust coordinates relative to table
            rel_x = text_center_x - tx1
            rel_y = text_center_y - ty1
            
            # Find which row and column this text belongs to
            row_idx = None
            col_idx = None
            
            for i, row in enumerate(rows):
                ry1, ry2 = row["box"][1], row["box"][3]
                if ry1 <= rel_y <= ry2:
                    row_idx = i
                    break
            
            for j, col in enumerate(columns):
                cx1, cx2 = col["box"][0], col["box"][2]
                if cx1 <= rel_x <= cx2:
                    col_idx = j
                    break
            
            if row_idx is not None and col_idx is not None:
                cell_grid[row_idx][col_idx].append(text_item.get("text", ""))
        
        # Convert cell lists to strings
        result = []
        for row in cell_grid:
            row_texts = []
            for cell_texts in row:
                # Sort by x-coordinate if available, then join
                cell_content = " ".join(cell_texts)
                row_texts.append(cell_content)
            result.append(row_texts)
        
        return result
    
    def _fallback_text_extraction(
        self,
        text_items: List[Dict[str, Any]],
        table_box: List[float]
    ) -> List[List[str]]:
        """
        Fallback method to extract table text without structure recognition.
        
        Groups text by y-coordinate to form rows.
        
        Args:
            text_items: List of OCR text items
            table_box: Bounding box of the table
            
        Returns:
            2D list of text (approximate rows)
        """
        tx1, ty1, tx2, ty2 = table_box
        
        # Filter text within table bounds
        table_texts = []
        for item in text_items:
            bbox = item.get("bounding_box", [])
            if not bbox or len(bbox) < 4:
                continue
            
            center_x = sum(p[0] for p in bbox) / 4
            center_y = sum(p[1] for p in bbox) / 4
            
            if tx1 <= center_x <= tx2 and ty1 <= center_y <= ty2:
                table_texts.append({
                    "text": item.get("text", ""),
                    "x": center_x,
                    "y": center_y
                })
        
        if not table_texts:
            return []
        
        # Sort by y, then x
        table_texts.sort(key=lambda t: (t["y"], t["x"]))
        
        # Group into rows (texts within similar y-coordinates)
        rows = []
        current_row = []
        last_y = None
        row_threshold = 20  # pixels
        
        for item in table_texts:
            if last_y is None or abs(item["y"] - last_y) < row_threshold:
                current_row.append(item)
            else:
                if current_row:
                    # Sort row by x-coordinate
                    current_row.sort(key=lambda t: t["x"])
                    rows.append([t["text"] for t in current_row])
                current_row = [item]
            last_y = item["y"]
        
        if current_row:
            current_row.sort(key=lambda t: t["x"])
            rows.append([t["text"] for t in current_row])
        
        return rows
    
    def _check_table_continuation(
        self,
        current_table: List[List[str]],
        previous_table: Optional[List[List[str]]]
    ) -> bool:
        """
        Check if the current table is a continuation of the previous table.
        
        Args:
            current_table: Current table content
            previous_table: Previous page's last table content
            
        Returns:
            True if tables should be merged
        """
        if not current_table or not previous_table:
            return False
        
        # Check if headers match (first row similarity)
        if len(current_table) > 0 and len(previous_table) > 0:
            current_header = current_table[0] if current_table else []
            prev_header = previous_table[0] if previous_table else []
            
            # If headers are very similar, it might be a continuation
            if current_header and prev_header:
                # Simple comparison - check if at least 50% of cells match
                matches = sum(
                    1 for a, b in zip(current_header, prev_header)
                    if a.strip().lower() == b.strip().lower()
                )
                min_len = min(len(current_header), len(prev_header))
                if min_len > 0 and matches / min_len >= 0.5:
                    return True
        
        return False
    
    def detect_tables_in_page(
        self,
        ocr_data: Dict[str, Any],
        page_image: Image.Image = None,
        pdf_path: str = None
    ) -> Dict[str, Any]:
        """
        Detect and extract tables from a single page.
        
        Args:
            ocr_data: OCR results for the page (with text_items)
            page_image: PIL Image of the page (optional, will load from PDF if not provided)
            pdf_path: Path to PDF file (used if page_image not provided)
            
        Returns:
            Updated page data with table information
        """
        page_number = ocr_data.get("page_number", 1)
        text_items = ocr_data.get("text_items", [])
        
        # Load page image if not provided
        if page_image is None and pdf_path:
            try:
                import fitz
                doc = fitz.open(pdf_path)
                page = doc[page_number - 1]
                zoom = Config.OCR_DPI / 72.0
                matrix = fitz.Matrix(zoom, zoom)
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                page_image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                doc.close()
            except Exception as e:
                logger.error(f"Failed to load page image: {e}")
                ocr_data["tables"] = []
                return ocr_data
        
        if page_image is None:
            logger.error("No page image available for table detection")
            ocr_data["tables"] = []
            return ocr_data
        
        # Detect tables
        logger.info(f"Detecting tables on page {page_number}...")
        detected_tables = self._detect_tables(page_image)
        
        if not detected_tables:
            logger.info(f"No tables detected on page {page_number}")
            ocr_data["tables"] = []
            return ocr_data
        
        logger.info(f"Found {len(detected_tables)} table(s) on page {page_number}")
        
        # Process each detected table
        tables = []
        for idx, table in enumerate(detected_tables):
            table_box = table["bounding_box"]
            
            # Recognize table structure
            logger.debug(f"Recognizing structure for table {idx + 1}...")
            structure = self._recognize_structure(page_image, table_box)
            
            # Assign text to cells
            logger.debug(f"Populating cells for table {idx + 1}...")
            table_content = self._assign_text_to_cells(text_items, table_box, structure)
            
            table_data = {
                "table_id": idx + 1,
                "bounding_box": table_box,
                "confidence": table["confidence"],
                "row_count": len(table_content),
                "column_count": max(len(row) for row in table_content) if table_content else 0,
                "rows": table_content,
                "continued_from_previous": False
            }
            
            tables.append(table_data)
            logger.info(f"Table {idx + 1}: {table_data['row_count']} rows x {table_data['column_count']} columns")
        
        ocr_data["tables"] = tables
        return ocr_data
    
    def process_all_pages(
        self,
        ocr_input_dir: str = None,
        output_dir: str = None,
        pdf_path: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process all pages to detect and extract tables.
        
        Args:
            ocr_input_dir: Directory containing OCR JSON files
            output_dir: Directory to save table-enriched JSON files
            pdf_path: Path to the original PDF file
            
        Returns:
            List of page data with table information
        """
        ocr_input_dir = Path(ocr_input_dir) if ocr_input_dir else Config.OCR_OUTPUT_DIR
        output_dir = Path(output_dir) if output_dir else Config.TABLE_OUTPUT_DIR
        
        # Initialize models
        if not self._initialize_models():
            logger.error("Cannot proceed without Table Transformer models")
            return []
        
        # Ensure output directory exists
        FileUtils.ensure_directory(output_dir)
        
        # Get all OCR JSON files
        json_files = FileUtils.list_json_files(ocr_input_dir)
        if not json_files:
            logger.error(f"No OCR JSON files found in {ocr_input_dir}")
            return []
        
        logger.info(f"Processing {len(json_files)} pages for table detection...")
        
        results = []
        previous_table = None
        
        for json_file in json_files:
            # Load OCR data
            ocr_data = FileUtils.read_json(json_file)
            if not ocr_data:
                logger.warning(f"Failed to read {json_file}, skipping...")
                continue
            
            page_number = ocr_data.get("page_number", 0)
            
            # Detect tables
            logger.info(f"Processing page {page_number} for tables...")
            enriched_data = self.detect_tables_in_page(
                ocr_data,
                pdf_path=pdf_path
            )
            
            # Check for table continuation
            if enriched_data.get("tables") and previous_table:
                first_table = enriched_data["tables"][0]
                if self._check_table_continuation(
                    first_table.get("rows", []),
                    previous_table
                ):
                    first_table["continued_from_previous"] = True
                    logger.info(f"Table on page {page_number} continues from previous page")
            
            # Save last table for continuation checking
            if enriched_data.get("tables"):
                previous_table = enriched_data["tables"][-1].get("rows", [])
            else:
                previous_table = None
            
            # Save enriched data
            output_path = FileUtils.get_page_json_path(output_dir, page_number)
            if FileUtils.write_json(enriched_data, output_path):
                logger.info(f"Saved table results for page {page_number}")
            
            results.append(enriched_data)
        
        tables_found = sum(len(r.get("tables", [])) for r in results)
        logger.info(f"Table detection complete. Found {tables_found} tables across {len(results)} pages.")
        
        return results


def main():
    """Command-line entry point for table detection."""
    parser = argparse.ArgumentParser(
        description="Detect and extract tables from PDF pages"
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Input directory containing OCR JSON files (default: output/ocr_results)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for table-enriched JSON files (default: output/table_results)"
    )
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the original PDF file (for loading page images)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"Detection confidence threshold (default: {Config.TABLE_DETECTION_THRESHOLD})"
    )
    
    args = parser.parse_args()
    
    # Create detector and run
    detector = TableDetector(detection_threshold=args.threshold)
    results = detector.process_all_pages(
        ocr_input_dir=args.input,
        output_dir=args.output,
        pdf_path=args.pdf
    )
    
    if results:
        tables_found = sum(len(r.get("tables", [])) for r in results)
        logger.info(f"Detection complete: {len(results)} pages, {tables_found} tables found")
        return 0
    else:
        logger.error("Table detection failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
