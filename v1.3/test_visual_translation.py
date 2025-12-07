from image_translator import ImageTranslator
import os

print("🧪 Testing Visual Translation System")
print("="*50)

# Initialize translator
print("Initializing ImageTranslator...")
translator = ImageTranslator()

# Test with an image
test_image = "test_images/chinese_test.jpg"

if os.path.exists(test_image):
    print(f"\nTranslating: {test_image}")
    result = translator.translate_image(
        image_path=test_image,
        source_language='chinese',
        output_path='test_images/chinese_translated.jpg',
        debug_mode=True
    )
    
    print(f"\nResult: {result}")
    
    if result['status'] == 'success':
        print(f"✅ Success! Check: {result['output_path']}")
    else:
        print(f"❌ Failed: {result.get('message')}")
else:
    print(f"❌ Test image not found: {test_image}")