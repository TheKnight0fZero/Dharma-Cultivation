# translator_integration.py - UPDATED with Image Translation
"""
Integration layer between Streamlit UI and Universal Translator.
Now with ACTUAL image translation - replaces text in images!
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import tempfile
import shutil
from datetime import datetime

# Import from cleaned file
try:
    from translator_core_clean import (
        UniversalTranslator,
        FileHandler,
        PDFProcessor,
        ZIPProcessor,
        OutputGenerator,
        Language,
        Config
    )
    print("✅ All translator components imported successfully")
    TRANSLATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    TRANSLATOR_AVAILABLE = False

# Import deep_translator for text translation
try:
    from deep_translator import GoogleTranslator
    GOOGLE_TRANSLATE_AVAILABLE = True
    print("✅ Google Translator ready")
except ImportError:
    print("⚠️ deep_translator not available - install with: pip install deep-translator")
    GOOGLE_TRANSLATE_AVAILABLE = False

# Import the new ImageTranslator
try:
    from image_translator import ImageTranslator
    IMAGE_TRANSLATOR_AVAILABLE = True
    print("✅ ImageTranslator ready for visual translation")
except ImportError as e:
    print(f"⚠️ ImageTranslator not available: {e}")
    IMAGE_TRANSLATOR_AVAILABLE = False

# Import PDF and image processing libraries for direct use
# --- CHANGE 1: Removed redundant PyPDF2 and zipfile imports from this block ---
try:
    from PIL import Image
    from pdf2image import convert_from_path
    PDF_IMAGE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ PDF/Image libraries missing: {e}")
    PDF_IMAGE_AVAILABLE = False


class TranslatorService:
    """Service layer for Streamlit app with REAL image translation."""
    
    def __init__(self):
        """Initialize all translator components."""
        self.ready = False
        self.image_translator = None
        self.temp_dir = None
        
        if TRANSLATOR_AVAILABLE:
            try:
                # Turn off verbose for UI
                Config.Debug.VERBOSE = False
                
                # Initialize ALL components
                self.file_handler = FileHandler(verbose=False)
                self.pdf_processor = PDFProcessor(
                    file_handler=self.file_handler,
                    verbose=False
                )
                self.zip_processor = ZIPProcessor(
                    file_handler=self.file_handler,
                    pdf_processor=self.pdf_processor,
                    verbose=False
                )
                self.output_generator = OutputGenerator(
                    file_handler=self.file_handler,
                    pdf_processor=self.pdf_processor,
                    zip_processor=self.zip_processor,
                    verbose=False
                )
                self.translator = UniversalTranslator()
                
                # --- CHANGE 2: Initialize GoogleTranslator instance once ---
                if GOOGLE_TRANSLATE_AVAILABLE:
                    self.google_translator_instance = GoogleTranslator(source='auto', target='en')
                else:
                    self.google_translator_instance = None # Handle case where deep_translator is not available
                
                # Initialize ImageTranslator if available
                if IMAGE_TRANSLATOR_AVAILABLE:
                    self.image_translator = ImageTranslator()
                    print("✅ Image translation system ready")
                else:
                    print("⚠️ Image translation not available - text only mode")
                
                # Create temp directory for session
                self.temp_dir = tempfile.mkdtemp(prefix="translator_session_")
                
                self.ready = True
                print("✅ All components initialized")
                
            except Exception as e:
                print(f"⚠️ Initialization error: {e}")
                self.ready = False
        else:
            self.ready = False
            print("❌ Translator not available")
    
    def translate_text_to_english(
        self,
        text: str,
        source_language: str = 'auto'
    ) -> str:
        """
        Translate text to English using Google Translate.
        
        Args:
            text: Text to translate
            source_language: Source language or 'auto'
            
        Returns:
            Translated English text
        """
        if not text or not text.strip():
            return ""
        
        if not GOOGLE_TRANSLATE_AVAILABLE or self.google_translator_instance is None:
            return f"[Translation not available - {text[:100]}...]"
        
        try:
            # Map UI language to Google Translate codes
            lang_map = {
                'chinese': 'zh-CN',
                'japanese': 'ja',
                'korean': 'ko',
                'hindi': 'hi',
                'auto-detect': 'auto'
            }
            
            source_lang = lang_map.get(source_language.lower(), 'auto')
            
            # --- CHANGE 3: Use the pre-initialized translator instance and remove manual chunking ---
            # Dynamically set the source language for this translation
            self.google_translator_instance.source = source_lang
            
            # Rely on deep_translator's internal chunking
            return self.google_translator_instance.translate(text)
                
        except Exception as e:
            print(f"Translation error: {e}")
            return f"[Translation failed: {str(e)[:50]}]"
    
    def translate_image_with_visual_replacement(
        self,
        image_path: str,
        source_language: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Translate image by replacing text visually in the image.
        
        Args:
            image_path: Path to image file
            source_language: Source language or 'auto-detect'
            
        Returns:
            Dictionary with translated image path and metadata
        """
        if not self.image_translator:
            return {
                'status': 'error',
                'message': 'Image translator not initialized',
                'output_path': None
            }
        
        try:
            # Map language names for image translator
            # --- CHANGE 4: Corrected language_map to pass string names to ImageTranslator ---
            language_map = {
                'auto-detect': 'auto',
                'chinese': 'chinese',
                'japanese': 'japanese',
                'korean': 'korean',
                'hindi': 'hindi'
            }
            
            lang = language_map.get(source_language.lower(), 'auto')
            
            # Generate output path in temp directory
            filename = Path(image_path).name
            base = Path(filename).stem
            ext = Path(filename).suffix
            output_path = os.path.join(self.temp_dir, f"{base}_translated{ext}")
            
            print(f"🖼️ Processing image with visual translation...")
            print(f"   Source: {filename}")
            print(f"   Language: {lang}")
            
            # Translate the image
            result = self.image_translator.translate_image(
                image_path=image_path,
                source_language=lang,
                output_path=output_path,
                preserve_style=True,
                debug_mode=False
            )
            
            if result['status'] == 'success':
                print(f"✅ Image translated successfully")
                print(f"   Regions found: {result.get('regions_detected', 0)}")
                print(f"   Regions translated: {result.get('regions_translated', 0)}")
                
                # Get sample translations for preview
                translations = result.get('translations', [])
                sample_text = ""
                if translations:
                    for t in translations[:3]:  # First 3 translations
                        sample_text += f"{t['original']} → {t['translated']}\n"
                
                return {
                    'status': 'success',
                    'output_path': result['output_path'],
                    'original': f"Detected {len(translations)} text regions",
                    'translated': sample_text if sample_text else "Image translated",
                    'method': f"Visual Translation ({lang})",
                    'file_type': 'image',
                    'regions': result.get('regions_detected', 0)
                }
            else:
                return {
                    'status': result['status'],
                    'output_path': None,
                    'original': result.get('message', 'Processing failed'),
                    'translated': 'Could not translate image',
                    'method': 'Visual Translation Failed',
                    'file_type': 'image'
                }
                
        except Exception as e:
            print(f"❌ Image translation error: {e}")
            return {
                'status': 'error',
                'output_path': None,
                'original': f'Error: {str(e)}',
                'translated': 'Translation failed',
                'method': 'Error',
                'file_type': 'image'
            }
    
    def translate_pdf_with_images(
        self,
        pdf_path: str,
        source_language: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Translate PDF by converting to images and translating each page.
        
        Args:
            pdf_path: Path to PDF file
            source_language: Source language
            
        Returns:
            Dictionary with results
        """
        if not PDF_IMAGE_AVAILABLE:
            return {
                'status': 'error',
                'message': 'PDF image processing not available'
            }
        
        try:
            print(f"📄 Converting PDF to images for translation...")
            
            # Create temp directory for PDF pages
            pdf_temp_dir = os.path.join(self.temp_dir, "pdf_pages")
            os.makedirs(pdf_temp_dir, exist_ok=True)
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=200,
                fmt='JPEG',
                output_folder=pdf_temp_dir
            )
            
            print(f"   Converted {len(images)} pages")
            
            # Translate each page image
            translated_images = []
            translations_summary = []
            
            # --- CHANGE 5: Removed redundant image.save() and used page_image_path directly ---
            for idx, page_image_path in enumerate(images, 1): # Renamed 'image' to 'page_image_path' for clarity
                # pdf2image already saved the image to page_image_path when output_folder was specified.
                # No need to re-save it.
                
                print(f"   Translating page {idx}/{len(images)}...")
                
                # Translate the page image
                if self.image_translator:
                    result = self.translate_image_with_visual_replacement(
                        page_image_path, # Use the path directly
                        source_language
                    )
                    
                    if result['status'] == 'success':
                        translated_images.append(result['output_path'])
                        translations_summary.append(f"Page {idx}: {result.get('regions', 0)} regions")
                    else:
                        # Keep original if translation failed
                        translated_images.append(page_image_path) # Use the original page_image_path
                        translations_summary.append(f"Page {idx}: Failed")
                else:
                    translated_images.append(page_image_path) # Use the original page_image_path
            
            # Create new PDF from translated images
            if translated_images:
                output_pdf_path = os.path.join(
                    self.temp_dir,
                    f"{Path(pdf_path).stem}_translated.pdf"
                )
                
                # Convert images back to PDF
                if translated_images:
                    img_list = []
                    for img_path in translated_images:
                        img = Image.open(img_path)
                        img_list.append(img.convert('RGB'))
                    
                    # Save as PDF
                    if img_list:
                        img_list[0].save(
                            output_pdf_path,
                            save_all=True,
                            append_images=img_list[1:] if len(img_list) > 1 else []
                        )
                        
                        print(f"✅ Created translated PDF: {Path(output_pdf_path).name}")
                        
                        return {
                            'status': 'success',
                            'output_path': output_pdf_path,
                            'original': f"PDF with {len(images)} pages",
                            'translated': '\n'.join(translations_summary[:5]),
                            'method': f'PDF Visual Translation ({len(images)} pages)',
                            'file_type': 'pdf',
                            'pages': len(images)
                        }
            
            return {
                'status': 'error',
                'message': 'Failed to create translated PDF'
            }
            
        except Exception as e:
            print(f"❌ PDF translation error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def translate_file(
        self,
        file_path: str,
        source_language: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Main translation function - handles all file types.
        Now returns TRANSLATED IMAGES instead of just text!
        """
        if not self.ready:
            return {
                'status': 'error',
                'original': 'Translator not ready',
                'translated': 'Check console for errors',
                'output_path': None
            }
        
        try:
            # Detect file type
            file_type = self.file_handler.file_type_detector(file_path)
            
            if file_type == 'image':
                # Use visual translation for images
                if self.image_translator:
                    return self.translate_image_with_visual_replacement(
                        file_path,
                        source_language
                    )
                else:
                    # Fallback to text extraction only
                    print("⚠️ Visual translation not available, extracting text only")
                    # --- CHANGE 6: Corrected language mapping for UniversalTranslator (Tesseract) fallback ---
                    lang_map = {
                        'chinese': Language.CHINESE,
                        'japanese': Language.JAPANESE,
                        'korean': Language.KOREAN,
                        'hindi': Language.HINDI,
                        'auto-detect': Language.ENGLISH # Changed to ENGLISH
                    }
                    lang = lang_map.get(source_language.lower(), Language.ENGLISH) # Changed to ENGLISH
                    result = self.translator.process(file_path, lang)
                    
                    return {
                        'status': 'success',
                        'output_path': None,
                        'original': result.get('original', ''),
                        'translated': result.get('translated', ''),
                        'method': 'Text Extraction Only',
                        'file_type': 'text'
                    }
                    
            elif file_type == 'pdf':
                # Translate PDF with visual replacement
                if self.image_translator and PDF_IMAGE_AVAILABLE:
                    return self.translate_pdf_with_images(
                        file_path,
                        source_language
                    )
                else:
                    # Fallback to text extraction
                    print("⚠️ Visual PDF translation not available")
                    pdf_result = self.pdf_processor.process_pdf_for_translation(file_path)
                    pages = pdf_result.get('pages', [])
                    
                    if pages:
                        all_text = ' '.join([p.get('text', '') for p in pages])[:2000]
                        translated_text = self.translate_text_to_english(all_text, source_language)
                        
                        return {
                            'status': 'success',
                            'output_path': None,
                            'original': all_text[:500],
                            'translated': translated_text[:500],
                            'method': f'PDF Text Extraction ({len(pages)} pages)',
                            'file_type': 'text'
                        }
                    else: # --- CHANGE 7: Added return for empty PDF text extraction fallback ---
                        return {
                            'status': 'no_text',
                            'message': 'No text found in PDF for text extraction',
                            'output_path': None
                        }
                    
            elif file_type == 'zip':
                # Process ZIP with visual translation
                return self._process_zip_with_visual_translation(
                    file_path,
                    source_language
                )
                    
            elif file_type == 'text':
                # Text files still use text-only translation
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    original_text = f.read()[:2000]
                
                translated_text = self.translate_text_to_english(
                    original_text,
                    source_language
                )
                
                # Save translated text
                output_path = os.path.join(
                    self.temp_dir,
                    f"{Path(file_path).stem}_translated.txt"
                )
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(translated_text)
                
                return {
                    'status': 'success',
                    'output_path': output_path,
                    'original': original_text[:500],
                    'translated': translated_text[:500],
                    'method': 'Text File',
                    'file_type': 'text'
                }
                
            else:
                return {
                    'status': 'error',
                    'original': f'Unknown file type: {file_type}',
                    'translated': 'Cannot process this file type',
                    'output_path': None
                }
                
        except Exception as e:
            print(f"Processing error: {e}")
            return {
                'status': 'error',
                'original': f'Error processing file',
                'translated': f'Error: {str(e)}',
                'output_path': None
            }
    
    def _process_zip_with_visual_translation(
        self,
        zip_path: str,
        source_language: str
    ) -> Dict[str, Any]:
        """
        Process ZIP file with visual translation for images.
        """
        try:
            print(f"📦 Processing ZIP archive...")
            
            # Create temp directory for extraction
            extract_dir = os.path.join(self.temp_dir, "zip_extract")
            os.makedirs(extract_dir, exist_ok=True)
            
            # Extract ZIP
            # --- CHANGE 8: Removed redundant local import zipfile ---
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
            image_files = []
            
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if Path(file).suffix.lower() in image_extensions:
                        image_files.append(os.path.join(root, file))
            
            print(f"   Found {len(image_files)} images in ZIP")
            
            if not image_files:
                return {
                    'status': 'error',
                    'message': 'No images found in ZIP'
                }
            
            # Translate each image
            # --- CHANGE 8: Removed hardcoded [:10] limit from image processing loop ---
            for idx, img_path in enumerate(image_files, 1): # Process all images
                print(f"   Translating image {idx}/{len(image_files)}...") # Updated print
                
                if self.image_translator:
                    result = self.translate_image_with_visual_replacement(
                        img_path,
                        source_language
                    )
                    
                    if result['status'] == 'success':
                        translated_files.append(result['output_path'])
                    else:
                        translated_files.append(img_path)
                else:
                    translated_files.append(img_path)
            
            # Create new ZIP with translated images
            output_zip_path = os.path.join(
                self.temp_dir,
                f"{Path(zip_path).stem}_translated.zip"
            )
            
            with zipfile.ZipFile(output_zip_path, 'w') as zf:
                for file_path in translated_files:
                    arcname = Path(file_path).name
                    zf.write(file_path, arcname)
            
            print(f"✅ Created translated ZIP: {Path(output_zip_path).name}")
            
            return {
                'status': 'success',
                'output_path': output_zip_path,
                'original': f"ZIP with {len(image_files)} images",
                'translated': f"Translated {len(translated_files)} images",
                'method': f'ZIP Visual Translation ({len(translated_files)} files)',
                'file_type': 'zip'
            }
            
        except Exception as e:
            print(f"❌ ZIP processing error: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def process_batch(
        self,
        file_paths: List[str],
        source_language: str = 'auto'
    ) -> List[Dict[str, Any]]:
        """Process multiple files with visual translation."""
        results = []
        for path in file_paths:
            try:
                result = self.translate_file(path, source_language)
                results.append(result)
            except Exception as e:
                results.append({
                    'status': 'error',
                    'original': f'Error with {Path(path).name}',
                    'translated': str(e),
                    'output_path': None
                })
        return results
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                print(f"⚠️ Could not clean temp files: {e}")
        
        if self.ready and self.file_handler:
            try:
                self.file_handler.temp_file_manager('cleanup', cleanup_all=True)
            except:
                pass


# Create global instance
translator_service = TranslatorService()

# Test translation capability (These prints are fine for initial setup)
if IMAGE_TRANSLATOR_AVAILABLE:
    print("✅ Visual image translation is available!")
    print("   - Images will have text replaced visually")
    print("   - PDFs will be converted and translated")
    print("   - ZIPs will have all images translated")
else:
    print("⚠️ Visual translation not available - text extraction only mode")

if GOOGLE_TRANSLATE_AVAILABLE:
    test_text = "测试"  # Chinese for "test"
    try:
        # --- Using the newly created google_translator_instance ---
        translator_service.google_translator_instance.source = 'zh-CN'
        result = translator_service.google_translator_instance.translate(test_text)
        print(f"✅ Text translation test: {test_text} → {result}")
    except Exception as e:
        print(f"⚠️ Translation test failed: {e}")

