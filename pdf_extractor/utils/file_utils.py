"""
File utilities for the PDF extraction pipeline.
Provides helper functions for file I/O, JSON handling, and path operations.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import Config

logger = Config.setup_logging(__name__)


class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def read_json(file_path: Union[str, Path]) -> Optional[Dict]:
        """
        Read and parse a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data as dictionary, or None if error
        """
        file_path = Path(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"Successfully read JSON file: {file_path}")
                return data
        except FileNotFoundError:
            logger.error(f"JSON file not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {file_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return None
    
    @staticmethod
    def write_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> bool:
        """
        Write data to a JSON file.
        
        Args:
            data: Data to write (must be JSON serializable)
            file_path: Path to the output file
            indent: JSON indentation level
            
        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)
                logger.debug(f"Successfully wrote JSON file: {file_path}")
                return True
        except Exception as e:
            logger.error(f"Error writing JSON file {file_path}: {e}")
            return False
    
    @staticmethod
    def list_json_files(directory: Union[str, Path]) -> List[Path]:
        """
        List all JSON files in a directory.
        
        Args:
            directory: Path to the directory
            
        Returns:
            List of Path objects for JSON files
        """
        directory = Path(directory)
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return []
        
        json_files = sorted(directory.glob("*.json"))
        logger.debug(f"Found {len(json_files)} JSON files in {directory}")
        return json_files
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> bool:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory: Path to the directory
            
        Returns:
            True if directory exists or was created, False on error
        """
        directory = Path(directory)
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False
    
    @staticmethod
    def copy_file(source: Union[str, Path], destination: Union[str, Path]) -> bool:
        """
        Copy a file to a new location.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Returns:
            True if successful, False otherwise
        """
        source = Path(source)
        destination = Path(destination)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            logger.debug(f"Copied file from {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error copying file from {source} to {destination}: {e}")
            return False
    
    @staticmethod
    def get_safe_filename(name: str) -> str:
        """
        Convert a string to a safe filename.
        
        Args:
            name: Original name
            
        Returns:
            Safe filename string
        """
        # Replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        safe_name = name
        for char in unsafe_chars:
            safe_name = safe_name.replace(char, '_')
        
        # Remove leading/trailing whitespace and dots
        safe_name = safe_name.strip('. ')
        
        # Limit length
        if len(safe_name) > 200:
            safe_name = safe_name[:200]
        
        return safe_name if safe_name else "unnamed"
    
    @staticmethod
    def get_page_json_path(output_dir: Union[str, Path], page_number: int) -> Path:
        """
        Get the standard path for a page's JSON file.
        
        Args:
            output_dir: Output directory
            page_number: Page number (1-indexed)
            
        Returns:
            Path to the page JSON file
        """
        return Path(output_dir) / f"page_{page_number:04d}.json"
