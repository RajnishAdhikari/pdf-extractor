"""
Extract all content from Cam-Followers1.pdf
Extracts: Text, Images, and Tables with page numbers
"""

import fitz  # PyMuPDF
import os
import json
from pathlib import Path
from datetime import datetime

def extract_pdf_content(pdf_path, output_dir):
    """Extract all content from PDF and save to output directory."""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sub-directories
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("CAM-FOLLOWERS PDF EXTRACTION")
    print("=" * 70)
    print(f"Source: {pdf_path}")
    print(f"Output: {output_dir}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Open PDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"\nTotal Pages: {total_pages}")
    
    # Results storage
    all_content = {
        "source_file": str(pdf_path),
        "extraction_time": datetime.now().isoformat(),
        "total_pages": total_pages,
        "pages": []
    }
    
    total_images = 0
    total_chars = 0
    
    # Process each page
    for page_num in range(total_pages):
        page = doc[page_num]
        page_data = {
            "page_number": page_num + 1,
            "text": "",
            "images": [],
            "links": []
        }
        
        print(f"\n--- Page {page_num + 1}/{total_pages} ---")
        
        # Extract text
        text = page.get_text("text")
        page_data["text"] = text
        total_chars += len(text)
        print(f"  Text: {len(text)} characters")
        
        # Extract images
        images = page.get_images(full=True)
        print(f"  Images: {len(images)}")
        
        for img_idx, img in enumerate(images):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Save image
                image_filename = f"page_{page_num + 1}_img_{img_idx + 1}.{image_ext}"
                image_path = images_dir / image_filename
                
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                page_data["images"].append({
                    "filename": image_filename,
                    "path": str(image_path),
                    "size_bytes": len(image_bytes),
                    "format": image_ext
                })
                total_images += 1
                print(f"    ✓ Saved: {image_filename}")
                
            except Exception as e:
                print(f"    ✗ Image error: {e}")
        
        # Extract links
        links = page.get_links()
        for link in links:
            if link.get("uri"):
                page_data["links"].append(link["uri"])
        
        all_content["pages"].append(page_data)
    
    doc.close()
    
    # Save combined text
    text_file = output_dir / "full_text.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(f"CAM-FOLLOWERS PDF - EXTRACTED TEXT\n")
        f.write(f"Source: {pdf_path}\n")
        f.write(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        for page_data in all_content["pages"]:
            f.write(f"\n{'=' * 40}\n")
            f.write(f"PAGE {page_data['page_number']}\n")
            f.write(f"{'=' * 40}\n\n")
            f.write(page_data["text"])
            f.write("\n")
    
    print(f"\n✓ Text saved: {text_file}")
    
    # Save JSON structure
    json_file = output_dir / "extraction_data.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(all_content, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON saved: {json_file}")
    
    # Save markdown summary
    md_file = output_dir / "content_summary.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Cam-Followers PDF Extraction Summary\n\n")
        f.write(f"- **Source**: `{pdf_path}`\n")
        f.write(f"- **Extracted**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Pages**: {total_pages}\n")
        f.write(f"- **Total Characters**: {total_chars:,}\n")
        f.write(f"- **Total Images**: {total_images}\n\n")
        
        f.write("## Pages Overview\n\n")
        f.write("| Page | Characters | Images |\n")
        f.write("|------|------------|--------|\n")
        
        for page_data in all_content["pages"]:
            text_len = len(page_data["text"])
            img_count = len(page_data["images"])
            f.write(f"| {page_data['page_number']} | {text_len:,} | {img_count} |\n")
        
        f.write("\n## Extracted Files\n\n")
        f.write("- `full_text.txt` - All text content\n")
        f.write("- `extraction_data.json` - Structured JSON data\n")
        f.write(f"- `images/` - {total_images} extracted images\n")
    
    print(f"✓ Summary saved: {md_file}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"Pages: {total_pages}")
    print(f"Total characters: {total_chars:,}")
    print(f"Images extracted: {total_images}")
    print(f"Output folder: {output_dir.absolute()}")
    print("=" * 70)
    
    return all_content


if __name__ == "__main__":
    pdf_path = "Cam-Followers1.pdf"
    output_dir = "extracted_content/cam_followers"
    
    extract_pdf_content(pdf_path, output_dir)
