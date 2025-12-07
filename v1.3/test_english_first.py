# test_english_only.py
"""Test with English language specifically"""

from image_translator import ImageTranslator
from PIL import Image, ImageDraw
import os

# Create English test image
img = Image.new('RGB', (600, 200), color='white')
draw = ImageDraw.Draw(img)
draw.text((50, 50), "Hello World", fill='black')
draw.text((50, 100), "Test Translation", fill='black')
img.save('test_images/english_test2.jpg')

# Test with ENGLISH specified (not auto)
translator = ImageTranslator()
result = translator.translate_image(
    image_path='test_images/english_test2.jpg',
    source_language='english',  # ← Specify English, not auto
    output_path='test_images/english_result.jpg'
)

print(f"Result: {result['status']}")
if result['status'] == 'success':
    print(f"✅ Success! Detected {result['regions_detected']} regions")
else:
    print(f"Message: {result.get('message', 'Unknown error')}")
