# PDF Content Extractor

A fast, efficient, and modular Python library for extracting text, images, and tables from PDF files using **open-source tools only**.

## Features

- ✅ **Text Extraction** - Fast text extraction with layout preservation
- ✅ **Image Extraction** - Extract embedded images in PNG/JPEG/WebP formats
- ✅ **Table Extraction** - Accurate table detection with CSV export
- ✅ **Page Tracking** - All content tagged with page numbers
- ✅ **JSON Output** - Structured output with metadata
- ✅ **Modular Design** - Use individual extractors or combined
- ✅ **100% Open Source** - No paid LLM or API dependencies

## Open-Source Libraries Used

| Library | Purpose | Speed |
|---------|---------|-------|
| **PyMuPDF (fitz)** | Text & Image extraction | ⚡ Very Fast |
| **pdfplumber** | Table detection & extraction | ⚡ Fast |
| **Pillow** | Image format conversion | ⚡ Fast |

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Command Line

```bash
# Extract all content
python main.py document.pdf

# Extract to specific directory
python main.py document.pdf --output-dir ./output

# Extract specific pages
python main.py document.pdf --pages 1,2,3,4,5

# Skip images
python main.py document.pdf --no-images

# Analyze PDF structure only
python main.py document.pdf --analyze
```

### Python API

```python
from pdf_extractor import PDFExtractor

# Initialize extractor
extractor = PDFExtractor(
    output_dir="./extracted",
    extract_text=True,
    extract_images=True,
    extract_tables=True
)

# Extract all content
result = extractor.extract("document.pdf")

# Access results by page
for page in result.pages:
    print(f"Page {page.page_number}:")
    
    # Text
    for text in page.texts:
        print(f"  Text: {text.word_count} words")
    
    # Images
    for img in page.images:
        print(f"  Image: {img.image_path}")
    
    # Tables
    for table in page.tables:
        print(f"  Table: {table.rows}x{table.columns}")

# Save as JSON
result.save_json("output.json")
```

## Project Structure

```
misumi project/
├── main.py                     # CLI entry point
├── requirements.txt            # Dependencies
├── example_usage.py            # Usage examples
├── pdf_extractor/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── extractor.py        # Main PDFExtractor class
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── text_extractor.py   # Text extraction module
│   │   ├── image_extractor.py  # Image extraction module
│   │   └── table_extractor.py  # Table extraction module
│   ├── models/
│   │   ├── __init__.py
│   │   └── extraction_result.py # Data models
│   └── utils/
│       ├── __init__.py
│       ├── file_utils.py       # File utilities
│       └── logger.py           # Logging setup
```

## Output Format

Extracted content is saved in a structured format:

```
extracted_content/
└── document_name/
    ├── extraction_result.json   # Complete extraction data
    ├── texts/
    │   ├── page_0001_text_000.txt
    │   └── ...
    ├── images/
    │   ├── page_0001_image_000.png
    │   └── ...
    └── tables/
        ├── page_0002_table_000.csv
        └── ...
```

### JSON Output Structure

```json
{
  "pdf_path": "document.pdf",
  "total_pages": 10,
  "extraction_time_seconds": 2.5,
  "timestamp": "2024-12-17T21:30:00",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "page_count": 10
  },
  "pages": [
    {
      "page_number": 1,
      "texts": [
        {
          "type": "text",
          "page_number": 1,
          "content": "...",
          "word_count": 150
        }
      ],
      "images": [...],
      "tables": [...]
    }
  ],
  "summary": {
    "total_text_blocks": 10,
    "total_images": 5,
    "total_tables": 3
  }
}
```

## Advanced Usage

### Extract Only Text

```python
extractor = PDFExtractor(output_dir="./output")
texts = extractor.extract_text_only("document.pdf")

for text in texts:
    print(f"Page {text.page_number}: {text.word_count} words")
```

### Extract Only Images

```python
extractor = PDFExtractor(output_dir="./output", image_format="jpeg")
images = extractor.extract_images_only("document.pdf")

for img in images:
    print(f"Page {img.page_number}: {img.width}x{img.height}")
```

### Extract Only Tables

```python
extractor = PDFExtractor(output_dir="./output")
tables = extractor.extract_tables_only("document.pdf")

for table in tables:
    print(f"Page {table.page_number}: {table.rows}x{table.columns}")
    print(f"  CSV: {table.csv_path}")
```

### Analyze PDF Structure

```python
extractor = PDFExtractor()
analysis = extractor.analyze("document.pdf")

for page in analysis['pages']:
    print(f"Page {page['page_number']}: "
          f"{page['text_length']} chars, "
          f"{page['image_count']} images, "
          f"{page['table_count']} tables")
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `output_dir` | `./extracted_content` | Output directory |
| `extract_text` | `True` | Extract text content |
| `extract_images` | `True` | Extract images |
| `extract_tables` | `True` | Extract tables |
| `image_format` | `png` | Image output format |
| `min_image_size` | `50` | Minimum image dimension |
| `export_tables_csv` | `True` | Save tables as CSV |
| `preserve_text_layout` | `True` | Keep text layout |

## Performance

- **Text extraction**: ~100+ pages/second
- **Image extraction**: Depends on image count/size
- **Table extraction**: ~10-50 pages/second

## License

Open source - MIT License
