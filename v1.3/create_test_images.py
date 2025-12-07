from PIL import Image, ImageDraw, ImageFont
import os

# Create test directory
os.makedirs("test_images", exist_ok=True)

# Test 1: Chinese text
img1 = Image.new('RGB', (400, 200), color='white')
draw1 = ImageDraw.Draw(img1)
# Note: This will show as boxes without Chinese font, but EasyOCR will detect it
draw1.text((50, 50), "你好世界", fill='black')
draw1.text((50, 100), "这是测试", fill='black')
img1.save('test_images/chinese_test.jpg')
print("Created: chinese_test.jpg")

# Test 2: Simple English (for comparison)
img2 = Image.new('RGB', (400, 200), color='white')
draw2 = ImageDraw.Draw(img2)
draw2.text((50, 50), "Hello World", fill='black')
draw2.text((50, 100), "This is a test", fill='black')
img2.save('test_images/english_test.jpg')
print("Created: english_test.jpg")

# Test 3: Mixed text
img3 = Image.new('RGB', (400, 200), color='white')
draw3 = ImageDraw.Draw(img3)
draw3.text((50, 30), "Test 测试", fill='black')
draw3.text((50, 80), "Translation 翻译", fill='black')
draw3.text((50, 130), "Image 图像", fill='black')
img3.save('test_images/mixed_test.jpg')
print("Created: mixed_test.jpg")

print("\n✅ Test images created in test_images/ directory")