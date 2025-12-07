from image_translator import ImageTranslator
from PIL import Image, ImageDraw

# Create image with numbers (universal)
img = Image.new('RGB', (400, 200), color='white')
draw = ImageDraw.Draw(img)
draw.text((50, 50), "123 456", fill='black')
draw.text((50, 100), "CHINESE", fill='black')
img.save('test_images/number_test.jpg')

# Download a real Chinese image
import urllib.request
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Chinese_Stop_Sign.jpg/220px-Chinese_Stop_Sign.jpg"
urllib.request.urlretrieve(url, "test_images/chinese_stop.jpg")
print("Downloaded Chinese stop sign")

# Translate it
translator = ImageTranslator()
result = translator.translate_image(
    'test_images/chinese_stop.jpg',
    'chinese',
    'test_images/stop_translated.jpg'
)

print(f"\nResult: {result['status']}")
if result['status'] == 'success':
    print(f"✅ Translated {result['regions_translated']} regions!")
    for t in result.get('translations', []):
        print(f"  '{t['original']}' → '{t['translated']}'")