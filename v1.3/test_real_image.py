from image_translator import ImageTranslator
from PIL import Image

# First, let's create a simple test image
img = Image.new('RGB', (400, 200), 'white')
from PIL import ImageDraw
draw = ImageDraw.Draw(img)

# Add clear black text
for i in range(3):  # Make it bold
    draw.text((50+i, 50), "HELLO WORLD", fill='black')
    draw.text((50+i, 100), "TEST IMAGE", fill='black')

img.save('test_real.jpg')
print("Created test_real.jpg")

# Now test with debug mode
translator = ImageTranslator()
result = translator.translate_image(
    'test_real.jpg',
    'english',  # or 'auto'
    'test_real_output.jpg',
    debug_mode=True  # This creates intermediate files!
)

print(f"\nResult: {result}")

if result['status'] == 'success':
    print("\n✅ Check these files:")
    print("1. test_real_output_cleaned.jpg - Should have text REMOVED")
    print("2. test_real_output.jpg - Should have English text ADDED")