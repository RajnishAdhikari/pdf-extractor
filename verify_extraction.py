import json

with open('extracted_content/cam_followers/extraction_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=" * 60)
print("EXTRACTION VERIFICATION")
print("=" * 60)
print(f"Source: {data['source_file']}")
print(f"Extraction Time: {data['extraction_time']}")
print(f"Total Pages: {data['total_pages']}")
print()
print("Page Summary:")
print("-" * 40)
total_chars = 0
total_imgs = 0
for p in data['pages']:
    chars = len(p['text'])
    imgs = len(p['images'])
    total_chars += chars
    total_imgs += imgs
    print(f"  Page {p['page_number']:2d}: {chars:5,} chars, {imgs} image(s)")

print("-" * 40)
print(f"TOTAL: {total_chars:,} characters, {total_imgs} images")
print()
print("Sample text from Page 1:")
print("-" * 40)
print(data['pages'][0]['text'][:500])
