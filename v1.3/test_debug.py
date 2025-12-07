from translator_integration import translator_service
import os

# Use your uploaded image
image_path = "test_images/chinese_test_real.jpg"  # or any test image

# Process with debug
result = translator_service.translate_image_with_visual_replacement(
    image_path,
    source_language='chinese'
)

print(f"Result: {result}")

# Check if the image translator is working
if translator_service.image_translator:
    # Try with debug mode
    from image_translator import ImageTranslator
    translator = ImageTranslator()
    
    debug_result = translator.translate_image(
        image_path,
        source_language='chinese',
        output_path='test_debug_output.jpg',
        debug_mode=True  # This saves intermediate steps!
    )
    
    print("\nDebug files created:")
    print("- test_debug_output_cleaned.jpg (text removed)")
    print("- test_debug_output.jpg (final result)")