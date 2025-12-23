"""
Model Grouper Module
Organizes extracted PDF data by model/part number.

This module reads page-level JSON files and groups them by model identifiers
found in the text, creating consolidated JSON files for each model.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import configuration
try:
    from ..config import Config
    from ..utils.file_utils import FileUtils
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pdf_extractor.config import Config
    from pdf_extractor.utils.file_utils import FileUtils

logger = Config.setup_logging(__name__)


class ModelGrouper:
    """
    Groups extracted PDF data by model/part number.
    
    Reads page-level JSON files with OCR text and tables,
    identifies model references, and consolidates data by model.
    """
    
    def __init__(
        self,
        model_patterns: List[str] = None,
        known_models: List[str] = None
    ):
        """
        Initialize the model grouper.
        
        Args:
            model_patterns: Regex patterns to identify model codes
            known_models: List of known model names/codes to look for
        """
        self.model_patterns = model_patterns or Config.MODEL_PATTERNS
        self.known_models = set(known_models) if known_models else set()
        
        # Compile regex patterns
        self.compiled_patterns = []
        for pattern in self.model_patterns:
            try:
                self.compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                logger.warning(f"Invalid regex pattern '{pattern}': {e}")
        
        logger.info(f"Initialized ModelGrouper with {len(self.compiled_patterns)} patterns")
        if self.known_models:
            logger.info(f"Known models: {len(self.known_models)}")
    
    def _extract_text_from_page(self, page_data: Dict[str, Any]) -> str:
        """
        Extract all text from a page data dictionary.
        
        Args:
            page_data: Page data with text_items and/or tables
            
        Returns:
            Combined text string from the page
        """
        texts = []
        
        # Get text from OCR items
        for item in page_data.get("text_items", []):
            text = item.get("text", "")
            if text:
                texts.append(text)
        
        # Get text from tables
        for table in page_data.get("tables", []):
            for row in table.get("rows", []):
                for cell in row:
                    if cell:
                        texts.append(str(cell))
        
        return " ".join(texts)
    
    def _find_model_references(self, text: str) -> Set[str]:
        """
        Find model/part number references in text.
        
        Args:
            text: Text to search for model references
            
        Returns:
            Set of found model identifiers
        """
        found_models = set()
        
        # Check for known models (exact match, case-insensitive)
        text_upper = text.upper()
        for model in self.known_models:
            if model.upper() in text_upper:
                found_models.add(model)
        
        # Search using regex patterns
        for pattern in self.compiled_patterns:
            matches = pattern.findall(text)
            for match in matches:
                # Clean up the match
                model_code = match.strip()
                if len(model_code) >= 2:  # Minimum length check
                    found_models.add(model_code)
        
        return found_models
    
    def _filter_model_codes(self, model_codes: Set[str]) -> Set[str]:
        """
        Filter out likely false positives from detected model codes.
        
        Args:
            model_codes: Set of detected model codes
            
        Returns:
            Filtered set of model codes
        """
        # Common words that might match patterns but aren't model codes
        exclusions = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT',
            'PAGE', 'TABLE', 'FIGURE', 'NOTE', 'ITEM', 'TYPE',
            'MIN', 'MAX', 'STD', 'REF', 'QTY', 'PCS', 'CAD',
            'PDF', 'USA', 'ISO', 'DIN', 'JIS', 'ANSI'
        }
        
        filtered = set()
        for code in model_codes:
            code_upper = code.upper()
            if code_upper not in exclusions:
                filtered.add(code)
        
        return filtered
    
    def _create_consolidated_data(
        self,
        model_name: str,
        pages_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create a consolidated JSON structure for a model.
        
        Args:
            model_name: The model identifier
            pages_data: List of page data dictionaries for this model
            
        Returns:
            Consolidated data dictionary
        """
        consolidated = {
            "model_name": model_name,
            "page_count": len(pages_data),
            "source_pages": [],
            "all_text": [],
            "all_tables": []
        }
        
        for page_data in pages_data:
            page_number = page_data.get("page_number", 0)
            source_file = page_data.get("source_file", "")
            
            # Record source page reference
            consolidated["source_pages"].append({
                "page_number": page_number,
                "source_file": source_file
            })
            
            # Collect text items with page reference
            for item in page_data.get("text_items", []):
                text_entry = {
                    "text": item.get("text", ""),
                    "confidence": item.get("confidence", 0),
                    "page": page_number
                }
                consolidated["all_text"].append(text_entry)
            
            # Collect tables with page reference
            for table in page_data.get("tables", []):
                table_entry = {
                    "table_id": table.get("table_id", 0),
                    "rows": table.get("rows", []),
                    "row_count": table.get("row_count", 0),
                    "column_count": table.get("column_count", 0),
                    "page": page_number,
                    "continued_from_previous": table.get("continued_from_previous", False)
                }
                consolidated["all_tables"].append(table_entry)
        
        # Add summary statistics
        consolidated["text_item_count"] = len(consolidated["all_text"])
        consolidated["table_count"] = len(consolidated["all_tables"])
        
        return consolidated
    
    def group_by_model(
        self,
        input_dir: str = None,
        output_dir: str = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group all page data by model identifier.
        
        Args:
            input_dir: Directory containing page JSON files
            output_dir: Directory to save grouped results
            
        Returns:
            Dictionary mapping model names to their consolidated data
        """
        input_dir = Path(input_dir) if input_dir else Config.TABLE_OUTPUT_DIR
        output_dir = Path(output_dir) if output_dir else Config.GROUPED_OUTPUT_DIR
        
        # Ensure output directory exists
        FileUtils.ensure_directory(output_dir)
        
        # Get all JSON files
        json_files = FileUtils.list_json_files(input_dir)
        if not json_files:
            logger.error(f"No JSON files found in {input_dir}")
            return {}
        
        logger.info(f"Processing {len(json_files)} page files for model grouping...")
        
        # Track model -> pages mapping
        model_pages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        unmatched_pages = []
        multi_model_pages = []
        
        # Process each page
        for json_file in json_files:
            page_data = FileUtils.read_json(json_file)
            if not page_data:
                logger.warning(f"Failed to read {json_file}, skipping...")
                continue
            
            page_number = page_data.get("page_number", 0)
            
            # Extract text and find model references
            page_text = self._extract_text_from_page(page_data)
            model_refs = self._find_model_references(page_text)
            model_refs = self._filter_model_codes(model_refs)
            
            if not model_refs:
                logger.warning(f"Page {page_number}: No model references found")
                unmatched_pages.append(page_number)
            elif len(model_refs) > 1:
                logger.info(f"Page {page_number}: Multiple models found: {model_refs}")
                multi_model_pages.append((page_number, list(model_refs)))
            else:
                logger.debug(f"Page {page_number}: Found model(s): {model_refs}")
            
            # Add page to each matching model
            for model in model_refs:
                model_pages[model].append(page_data)
        
        # Log summary
        logger.info(f"Found {len(model_pages)} unique models across all pages")
        if unmatched_pages:
            logger.warning(f"Pages without model references: {unmatched_pages}")
        if multi_model_pages:
            logger.info(f"Pages with multiple models: {len(multi_model_pages)}")
        
        # Create output for each model
        results = {}
        
        for model_name, pages in model_pages.items():
            # Create model-specific directory
            safe_name = FileUtils.get_safe_filename(model_name)
            model_dir = output_dir / safe_name
            FileUtils.ensure_directory(model_dir)
            
            # Create consolidated data
            consolidated = self._create_consolidated_data(model_name, pages)
            
            # Save consolidated JSON
            output_path = model_dir / f"{safe_name}.json"
            if FileUtils.write_json(consolidated, output_path):
                logger.info(f"Model '{model_name}': {len(pages)} pages -> {output_path.name}")
            
            results[model_name] = consolidated
        
        # Save unmatched pages summary
        if unmatched_pages:
            summary = {
                "unmatched_pages": unmatched_pages,
                "multi_model_pages": multi_model_pages
            }
            summary_path = output_dir / "_grouping_summary.json"
            FileUtils.write_json(summary, summary_path)
            logger.info(f"Grouping summary saved to {summary_path.name}")
        
        logger.info(f"Model grouping complete. {len(results)} models processed.")
        return results
    
    def add_known_model(self, model_name: str):
        """
        Add a known model name to search for.
        
        Args:
            model_name: Model name/code to add
        """
        self.known_models.add(model_name)
        logger.debug(f"Added known model: {model_name}")
    
    def add_known_models_from_file(self, file_path: str) -> int:
        """
        Load known model names from a text file.
        
        Args:
            file_path: Path to file with model names (one per line)
            
        Returns:
            Number of models loaded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"Model list file not found: {file_path}")
            return 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                count = 0
                for line in f:
                    model = line.strip()
                    if model and not model.startswith('#'):
                        self.known_models.add(model)
                        count += 1
                
                logger.info(f"Loaded {count} known models from {file_path}")
                return count
        except Exception as e:
            logger.error(f"Error reading model list file: {e}")
            return 0
    
    def get_model_list(self) -> List[str]:
        """
        Get list of all known models.
        
        Returns:
            Sorted list of known model names
        """
        return sorted(self.known_models)


def main():
    """Command-line entry point for model grouping."""
    parser = argparse.ArgumentParser(
        description="Group extracted PDF data by model/part number"
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Input directory containing page JSON files (default: output/table_results)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for grouped results (default: output/grouped_by_model)"
    )
    parser.add_argument(
        "--models-file",
        default=None,
        help="Path to file containing known model names (one per line)"
    )
    parser.add_argument(
        "--pattern",
        action="append",
        help="Additional regex pattern for model detection (can be used multiple times)"
    )
    
    args = parser.parse_args()
    
    # Create grouper
    patterns = list(Config.MODEL_PATTERNS)
    if args.pattern:
        patterns.extend(args.pattern)
    
    grouper = ModelGrouper(model_patterns=patterns)
    
    # Load known models if provided
    if args.models_file:
        grouper.add_known_models_from_file(args.models_file)
    
    # Run grouping
    results = grouper.group_by_model(
        input_dir=args.input,
        output_dir=args.output
    )
    
    if results:
        total_pages = sum(r.get("page_count", 0) for r in results.values())
        logger.info(f"Grouping complete: {len(results)} models, {total_pages} page assignments")
        return 0
    else:
        logger.error("Model grouping failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
