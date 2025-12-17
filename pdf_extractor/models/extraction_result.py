"""
Data models for extracted PDF content.
Provides structured representation of extraction results with page numbers.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json


@dataclass
class TextResult:
    """Represents extracted text from a PDF page."""
    content: str
    page_number: int
    char_count: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        if self.content:
            self.char_count = len(self.content)
            self.word_count = len(self.content.split())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "text",
            "page_number": self.page_number,
            "content": self.content,
            "char_count": self.char_count,
            "word_count": self.word_count
        }


@dataclass
class ImageResult:
    """Represents an extracted image from a PDF page."""
    image_path: str
    page_number: int
    image_index: int
    width: int = 0
    height: int = 0
    format: str = ""
    size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "image",
            "page_number": self.page_number,
            "image_index": self.image_index,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "size_bytes": self.size_bytes
        }


@dataclass
class TableResult:
    """Represents an extracted table from a PDF page."""
    data: List[List[str]]
    page_number: int
    table_index: int
    rows: int = 0
    columns: int = 0
    csv_path: Optional[str] = None
    
    def __post_init__(self):
        if self.data:
            self.rows = len(self.data)
            self.columns = max(len(row) for row in self.data) if self.data else 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "table",
            "page_number": self.page_number,
            "table_index": self.table_index,
            "rows": self.rows,
            "columns": self.columns,
            "data": self.data,
            "csv_path": self.csv_path
        }


@dataclass
class PageContent:
    """Represents all extracted content from a single PDF page."""
    page_number: int
    texts: List[TextResult] = field(default_factory=list)
    images: List[ImageResult] = field(default_factory=list)
    tables: List[TableResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "texts": [t.to_dict() for t in self.texts],
            "images": [i.to_dict() for i in self.images],
            "tables": [t.to_dict() for t in self.tables],
            "summary": {
                "text_blocks": len(self.texts),
                "images_count": len(self.images),
                "tables_count": len(self.tables)
            }
        }


@dataclass
class ExtractionResult:
    """Complete extraction result for a PDF document."""
    pdf_path: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)
    extraction_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "extraction_time_seconds": self.extraction_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "pages": [p.to_dict() for p in self.pages],
            "summary": {
                "total_text_blocks": sum(len(p.texts) for p in self.pages),
                "total_images": sum(len(p.images) for p in self.pages),
                "total_tables": sum(len(p.tables) for p in self.pages)
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert extraction result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_json(self, output_path: str) -> None:
        """Save extraction result to a JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
