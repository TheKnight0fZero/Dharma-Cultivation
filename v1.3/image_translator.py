# image_translator.py - Fix 4 with Universal Image Mode Support
"""
Image Translation Module for Universal Translator v1.5
Simple overlay approach with smart image mode handling
Supports grayscale, RGB, RGBA, and exotic image modes
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import tempfile
import logging

# Image processing
from PIL import Image, ImageDraw, ImageFont, ImageOps
import easyocr

# Translation
from deep_translator import GoogleTranslator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageTranslator:
    """
    Simple image translation system using overlay approach.
    Covers original text with white boxes and adds English text.
    Handles all common image modes (grayscale, RGB, RGBA, etc.)
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ImageTranslator with language models.
        
        Args:
            cache_dir: Directory to cache EasyOCR models
        """
        self.cache_dir = cache_dir or os.path.expanduser('~/.EasyOCR/')
        self.readers = {}
        self.translator = None
        self.supported_languages = {
            'chinese': ['ch_sim', 'en'],
            'japanese': ['ja', 'en'],
            'korean': ['ko', 'en'],
            'hindi': ['hi', 'en'],
            'english': ['en'],
            'auto': ['en']
        }
        
        # Font settings
        self.font_paths = {
            'default': "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            'bold': "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            'fallback': "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        }
        
        # Initialize translator
        self._init_translator()
        
        logger.info("ImageTranslator initialized (Universal Mode Support)")
    
    def _init_translator(self):
        """Initialize Google Translator"""
        try:
            self.translator = GoogleTranslator(source='auto', target='en')
            logger.info("Google Translator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize translator: {e}")
            raise
    
    def _get_reader(self, language: str) -> easyocr.Reader:
        """
        Get or create EasyOCR reader for specified language.
        
        Args:
            language: Language name
            
        Returns:
            EasyOCR Reader object
        """
        if language not in self.readers:
            try:
                lang_codes = self.supported_languages.get(
                    language.lower(), 
                    self.supported_languages['auto']
                )
                
                logger.info(f"Initializing EasyOCR for {language}: {lang_codes}")
                self.readers[language] = easyocr.Reader(
                    lang_codes,
                    gpu=False,
                    model_storage_directory=self.cache_dir,
                    download_enabled=True
                )
                logger.info(f"EasyOCR reader created for {language}")
                
            except Exception as e:
                logger.error(f"Failed to create reader for {language}: {e}")
                raise
        
        return self.readers[language]
    
    def translate_image(
        self,
        image_path: str,
        source_language: str = 'auto',
        output_path: Optional[str] = None,
        preserve_style: bool = True,
        debug_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Main function to translate text in image using simple overlay.
        
        Args:
            image_path: Path to input image
            source_language: Source language or 'auto'
            output_path: Where to save result
            preserve_style: Ignored in simple mode
            debug_mode: Save intermediate steps
            
        Returns:
            Dictionary with results and metadata
        """
        logger.info(f"Starting translation for: {image_path}")
        
        # Validate input
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Generate output path if not provided
        if not output_path:
            base = Path(image_path).stem
            ext = Path(image_path).suffix
            output_path = f"{base}_translated{ext}"
        
        try:
            # Step 1: Detect text regions
            logger.info("Detecting text regions...")
            text_regions = self.detect_text_regions(
                image_path,
                source_language
            )
            
            if not text_regions:
                logger.warning("No text detected in image")
                return {
                    'status': 'no_text',
                    'message': 'No text found in image',
                    'output_path': None
                }
            
            # Step 2: Translate text
            logger.info(f"Translating {len(text_regions)} text regions...")
            translated_regions = self.translate_text_regions(text_regions)
            
            # Step 3: Simple overlay - Load image and overlay translated text
            logger.info("Applying simple overlay translation...")
            final_image = self.simple_overlay_translation(
                image_path,
                translated_regions
            )
            
            # Step 4: Save result
            final_image.save(output_path, quality=95)
            logger.info(f"Saved translated image: {output_path}")
            
            return {
                'status': 'success',
                'output_path': output_path,
                'regions_detected': len(text_regions),
                'regions_translated': len(translated_regions),
                'translations': [
                    {
                        'original': r['text'],
                        'translated': r['translated_text']
                    } for r in translated_regions
                ]
            }
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'output_path': None
            }
    
    def detect_text_regions(
        self,
        image_path: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """
        Detect text regions with bounding boxes.
        
        Args:
            image_path: Path to image
            language: Language to detect
            
        Returns:
            List of text regions with coordinates and text
        """
        reader = self._get_reader(language)
        
        # Read text from image
        results = reader.readtext(
            image_path,
            detail=1,
            paragraph=False,
            width_ths=0.3,
            height_ths=0.3,
            text_threshold=0.5,
            low_text=0.3
        )
        
        text_regions = []
        for idx, (bbox, text, confidence) in enumerate(results):
            if confidence > 0.1:
                # Convert bbox to numpy array
                bbox_array = np.array(bbox, dtype=np.int32)
                
                # Calculate region properties
                x_coords = bbox_array[:, 0]
                y_coords = bbox_array[:, 1]
                
                region = {
                    'id': idx,
                    'bbox': bbox,
                    'bbox_array': bbox_array,
                    'text': text,
                    'confidence': confidence,
                    'x_min': int(np.min(x_coords)),
                    'y_min': int(np.min(y_coords)),
                    'x_max': int(np.max(x_coords)),
                    'y_max': int(np.max(y_coords)),
                    'width': int(np.max(x_coords) - np.min(x_coords)),
                    'height': int(np.max(y_coords) - np.min(y_coords)),
                    'center': (
                        int((np.min(x_coords) + np.max(x_coords)) / 2),
                        int((np.min(y_coords) + np.max(y_coords)) / 2)
                    )
                }
                
                text_regions.append(region)
                logger.debug(f"Detected: '{text}' at {region['center']} (conf: {confidence:.2f})")
        
        logger.info(f"Detected {len(text_regions)} text regions")
        return text_regions
    
    def translate_text_regions(
        self,
        text_regions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Translate text for each region.
        
        Args:
            text_regions: List of text regions to translate
            
        Returns:
            Updated regions with translated text
        """
        translated_regions = []
        
        for region in text_regions:
            original_text = region['text']
            
            try:
                # Skip if already English
                if self._is_english(original_text):
                    translated_text = original_text
                else:
                    # Translate to English
                    translated_text = self.translator.translate(original_text)
                
                # Create new region with translation
                translated_region = region.copy()
                translated_region['original_text'] = original_text
                translated_region['translated_text'] = translated_text
                translated_region['is_translated'] = (original_text != translated_text)
                
                translated_regions.append(translated_region)
                
                logger.debug(f"Translated: '{original_text}' -> '{translated_text}'")
                
            except Exception as e:
                logger.error(f"Translation failed for '{original_text}': {e}")
                # Keep original on error
                translated_region = region.copy()
                translated_region['original_text'] = original_text
                translated_region['translated_text'] = original_text
                translated_region['is_translated'] = False
                translated_region['error'] = str(e)
                translated_regions.append(translated_region)
        
        return translated_regions
    
    # ============= UNIVERSAL MODE HANDLING VERSION =============
    def simple_overlay_translation(
        self,
        image_path: str,
        translated_regions: List[Dict[str, Any]]
    ) -> Image.Image:
        """
        Simple overlay approach with smart mode handling.
        Handles grayscale, RGB, RGBA, and exotic image modes properly.
        
        Args:
            image_path: Path to original image
            translated_regions: Regions with translated text
            
        Returns:
            PIL Image with overlaid translations
        """
        # Load the image
        img = Image.open(image_path)
        original_mode = img.mode
        
        logger.info(f"Processing image with mode: {original_mode}")
        
        # Handle different image modes intelligently
        if img.mode == 'RGBA':
            # Flatten transparency to white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
            img = background
            draw_mode = 'RGB'
        elif img.mode in ['P', 'CMYK', '1', 'LAB', 'HSV', 'YCbCr']:
            # Convert exotic modes to RGB
            img = img.convert('RGB')
            draw_mode = 'RGB'
        elif img.mode == 'LA':
            # Grayscale with alpha - flatten to grayscale
            background = Image.new('L', img.size, 255)
            background.paste(img, mask=img.split()[1])
            img = background
            draw_mode = 'L'
        elif img.mode == 'L':
            # Keep grayscale as grayscale
            draw_mode = 'L'
        else:  # RGB or other supported modes
            draw_mode = img.mode
        
        draw = ImageDraw.Draw(img)
        
        # Process each text region
        for region in translated_regions:
            # Get coordinates with padding
            padding = 8
            x_min = region['x_min'] - padding
            y_min = region['y_min'] - padding
            x_max = region['x_max'] + padding
            y_max = region['y_max'] + padding
            
            # Set colors based on mode
            if draw_mode == 'L':
                # Grayscale mode - use single values
                white_color = 255
                black_color = 0
                gray_color = 200
            else:
                # RGB mode - use tuples
                white_color = (255, 255, 255)
                black_color = (0, 0, 0)
                gray_color = (200, 200, 200)
            
            # Draw white rectangle to cover text
            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                fill=white_color,
                outline=white_color,
                width=2
            )
            
            # Get translated text
            text = region.get('translated_text', '')
            
            if text:
                # Calculate available space
                box_width = x_max - x_min - (padding * 2)
                box_height = y_max - y_min - (padding * 2)
                
                # Find best font size
                font_size = self._find_best_font_size(
                    text, box_width, box_height, draw
                )
                
                # Get font
                font = self._get_font(font_size)
                
                # Wrap text if needed
                lines = self._wrap_text(text, font, box_width, draw)
                
                # Draw text centered in box
                y_offset = y_min + padding
                
                for line in lines:
                    # Get line dimensions
                    bbox = draw.textbbox((0, 0), line, font=font)
                    line_width = bbox[2] - bbox[0]
                    line_height = bbox[3] - bbox[1]
                    
                    # Center horizontally
                    x_pos = x_min + padding + (box_width - line_width) // 2
                    
                    # Draw with appropriate color
                    draw.text(
                        (x_pos, y_offset),
                        line,
                        fill=black_color,
                        font=font
                    )
                    
                    y_offset += line_height + 2
                    
                    # Stop if we run out of space
                    if y_offset > y_max - padding:
                        break
            
            # Add subtle border for clarity
            draw.rectangle(
                [x_min, y_min, x_max, y_max],
                outline=gray_color,
                width=1
            )
        
        # Convert back to original mode if it was simple grayscale
        if original_mode == 'L' and draw_mode == 'L':
            logger.info("Keeping grayscale mode")
            return img  # Keep as grayscale
        elif original_mode == 'P' and draw_mode == 'RGB':
            # Try to convert back to palette if possible
            try:
                if len(img.getcolors(256)) <= 256:
                    logger.info("Converting back to palette mode")
                    return img.convert('P')
            except:
                pass
        
        logger.info(f"Output mode: {img.mode}")
        return img
    
    def _find_best_font_size(
        self,
        text: str,
        max_width: int,
        max_height: int,
        draw: ImageDraw.Draw
    ) -> int:
        """
        Find the best font size to fit text in given space.
        
        Args:
            text: Text to fit
            max_width: Maximum width
            max_height: Maximum height
            draw: Draw object for measurements
            
        Returns:
            Optimal font size
        """
        min_size = 8
        max_size = 60
        
        # Binary search for best size
        while min_size < max_size:
            mid_size = (min_size + max_size + 1) // 2
            font = self._get_font(mid_size)
            
            # Check if text fits
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Check with potential wrapping
            if len(text.split()) > 1:
                # Estimate wrapped size
                words_per_line = max(1, int(text_width / max_width) + 1)
                estimated_lines = len(text.split()) // words_per_line + 1
                text_height = text_height * estimated_lines
            
            if text_width <= max_width and text_height <= max_height:
                min_size = mid_size
            else:
                max_size = mid_size - 1
        
        return min_size
    
    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        draw: ImageDraw.Draw
    ) -> List[str]:
        """
        Wrap text to fit within maximum width.
        
        Args:
            text: Text to wrap
            font: Font to use
            max_width: Maximum line width
            draw: Draw object for measurements
            
        Returns:
            List of wrapped lines
        """
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            # Test adding this word
            test_line = ' '.join(current_line + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]
            
            if line_width <= max_width:
                current_line.append(word)
            else:
                # Start new line
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        # Add last line
        if current_line:
            lines.append(' '.join(current_line))
        
        # If no lines fit, force at least the text
        if not lines:
            lines = [text]
        
        return lines
    
    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        """
        Get font object with fallback options.
        
        Args:
            size: Font size
            
        Returns:
            PIL Font object
        """
        for font_type in ['bold', 'default', 'fallback']:
            font_path = self.font_paths.get(font_type)
            if font_path and os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, size)
                except Exception as e:
                    logger.warning(f"Failed to load font {font_path}: {e}")
        
        # Ultimate fallback
        return ImageFont.load_default()
    
    def _is_english(self, text: str) -> bool:
        """
        Check if text is already in English.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be English
        """
        if not text:
            return True
        
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return (ascii_chars / len(text)) > 0.8
    
    def process_batch(
        self,
        image_paths: List[str],
        source_language: str = 'auto',
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of image paths
            source_language: Source language for all images
            output_dir: Directory for output files
            
        Returns:
            List of results for each image
        """
        results = []
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {idx}/{len(image_paths)}: {image_path}")
            
            # Generate output path
            if output_dir:
                filename = Path(image_path).name
                base = Path(filename).stem
                ext = Path(filename).suffix
                output_path = os.path.join(output_dir, f"{base}_translated{ext}")
            else:
                output_path = None
            
            # Process image
            result = self.translate_image(
                image_path,
                source_language,
                output_path
            )
            
            result['source_path'] = image_path
            results.append(result)
        
        return results


# Test function
def test_image_translator():
    """Test the image translator with a sample image"""
    print("Testing ImageTranslator (Universal Mode Support)...")
    
    # Create test images in different modes
    from PIL import Image, ImageDraw
    
    # Test 1: RGB image
    test_rgb = Image.new('RGB', (400, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(test_rgb)
    draw.text((50, 50), "RGB Test", fill=(0, 0, 0))
    test_rgb.save('test_rgb.jpg')
    
    # Test 2: Grayscale image
    test_gray = Image.new('L', (400, 200), color=255)
    draw = ImageDraw.Draw(test_gray)
    draw.text((50, 50), "Grayscale Test", fill=0)
    test_gray.save('test_gray.jpg')
    
    # Test translation
    translator = ImageTranslator()
    
    # Test RGB
    result_rgb = translator.translate_image(
        'test_rgb.jpg',
        source_language='auto',
        output_path='test_rgb_output.jpg'
    )
    print(f"RGB test result: {result_rgb['status']}")
    
    # Test Grayscale
    result_gray = translator.translate_image(
        'test_gray.jpg',
        source_language='auto',
        output_path='test_gray_output.jpg'
    )
    print(f"Grayscale test result: {result_gray['status']}")
    
    # Cleanup
    import os
    for file in ['test_rgb.jpg', 'test_gray.jpg', 'test_rgb_output.jpg', 'test_gray_output.jpg']:
        if os.path.exists(file):
            os.remove(file)
    
    return {'rgb': result_rgb, 'gray': result_gray}


if __name__ == "__main__":
    test_image_translator()