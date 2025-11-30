# translator_integration.py - FIXED with actual translation
"""
Integration layer between Streamlit UI and Universal Translator.
Now with ACTUAL translation to English!
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List

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

# Import deep_translator for actual translation
try:
    from deep_translator import GoogleTranslator
    GOOGLE_TRANSLATE_AVAILABLE = True
    print("✅ Google Translator ready")
except ImportError:
    print("⚠️ deep_translator not available - install with: pip install deep-translator")
    GOOGLE_TRANSLATE_AVAILABLE = False

class TranslatorService:
    """Service layer for Streamlit app with REAL translation."""
    
    def __init__(self):
        """Initialize all translator components."""
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
        Actually translate text to English using Google Translate.
        
        Args:
            text: Text to translate
            source_language: Source language or 'auto'
            
        Returns:
            Translated English text
        """
        if not text or not text.strip():
            return ""
        
        if not GOOGLE_TRANSLATE_AVAILABLE:
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
            
            # Create translator
            translator = GoogleTranslator(source=source_lang, target='en')
            
            # Translate in chunks if text is too long (5000 char limit)
            if len(text) > 4500:
                chunks = []
                for i in range(0, len(text), 4500):
                    chunk = text[i:i+4500]
                    translated_chunk = translator.translate(chunk)
                    chunks.append(translated_chunk)
                return ' '.join(chunks)
            else:
                # Translate directly
                return translator.translate(text)
                
        except Exception as e:
            print(f"Translation error: {e}")
            return f"[Translation failed: {str(e)[:50]}]"
    
    def translate_file(
        self,
        file_path: str,
        source_language: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Translate file with ACTUAL English translation.
        """
        if not self.ready:
            return {
                'original': 'Translator not ready',
                'translated': 'Check console for errors',
                'status': 'error'
            }
        
        # Map languages for OCR
        lang_map = {
            'chinese': Language.CHINESE,
            'japanese': Language.JAPANESE,
            'korean': Language.KOREAN,
            'hindi': Language.HINDI,
            'auto-detect': Language.CHINESE
        }
        
        try:
            # Detect file type
            file_type = self.file_handler.file_type_detector(file_path)
            
            if file_type == 'image':
                # Process image with OCR
                lang = lang_map.get(source_language.lower(), Language.CHINESE)
                result = self.translator.process(file_path, lang)
                
                # Get extracted text
                original_text = result.get('original', '')
                
                # ACTUALLY TRANSLATE TO ENGLISH
                if original_text:
                    translated_text = self.translate_text_to_english(
                        original_text,
                        source_language
                    )
                else:
                    translated_text = "No text found in image"
                
                return {
                    'original': original_text[:500],
                    'translated': translated_text[:500],
                    'status': 'success',
                    'method': 'OCR + Translation'
                }
                
            elif file_type == 'pdf':
                # Extract text from PDF
                pdf_result = self.pdf_processor.process_pdf_for_translation(file_path)
                pages = pdf_result.get('pages', [])
                
                if pages:
                    # Combine text from all pages
                    all_text = []
                    for page in pages:
                        page_text = page.get('text', '')
                        if page_text:
                            all_text.append(page_text)
                    
                    original_text = ' '.join(all_text)[:2000]  # Limit for preview
                    
                    # ACTUALLY TRANSLATE THE PDF TEXT
                    if original_text:
                        translated_text = self.translate_text_to_english(
                            original_text,
                            source_language
                        )
                        status_msg = f"Translated {len(pages)} pages from PDF"
                    else:
                        translated_text = "No text found in PDF"
                        status_msg = "PDF appears to be empty or image-based"
                    
                    return {
                        'original': original_text[:500],
                        'translated': translated_text[:500],
                        'status': 'success',
                        'method': f'PDF Extraction ({len(pages)} pages)'
                    }
                else:
                    # Try OCR on PDF pages if no text found
                    if pdf_result.get('needs_ocr'):
                        return {
                            'original': 'PDF needs OCR (scanned document)',
                            'translated': 'Please convert PDF to images first',
                            'status': 'error',
                            'method': 'PDF (needs OCR)'
                        }
                    else:
                        return {
                            'original': 'No text found in PDF',
                            'translated': 'Could not extract text',
                            'status': 'error',
                            'method': 'PDF Extraction Failed'
                        }
                    
            elif file_type == 'zip':
                # Process ZIP archive
                self._process_zip_with_translation(file_path, source_language)
                
            elif file_type == 'text':
                # Process text file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    original_text = f.read()[:2000]  # Limit for preview
                
                # TRANSLATE TEXT FILE
                if original_text:
                    translated_text = self.translate_text_to_english(
                        original_text,
                        source_language
                    )
                else:
                    translated_text = "Empty text file"
                
                return {
                    'original': original_text[:500],
                    'translated': translated_text[:500],
                    'status': 'success',
                    'method': 'Text File'
                }
                
            else:
                return {
                    'original': f'Unknown file type: {file_type}',
                    'translated': 'Cannot process this file type',
                    'status': 'error',
                    'method': 'Unknown'
                }
                
        except Exception as e:
            print(f"Processing error: {e}")
            return {
                'original': f'Error processing file',
                'translated': f'Error: {str(e)}',
                'status': 'error',
                'method': 'Error'
            }
    
    def _process_zip_with_translation(
        self,
        zip_path: str,
        source_language: str
    ) -> Dict[str, Any]:
        """
        Process and translate contents of ZIP file.
        """
        try:
            # Extract ZIP
            extracted_files = self.zip_processor.extract_zip(zip_path)
            
            if not extracted_files:
                return {
                    'original': 'Empty ZIP file',
                    'translated': 'No files to process',
                    'status': 'error',
                    'method': 'ZIP Processing'
                }
            
            # Process each file in the ZIP
            results = []
            translated_count = 0
            
            for file_path in extracted_files[:10]:  # Limit to first 10 files
                try:
                    # Get file type
                    file_type = self.file_handler.file_type_detector(file_path)
                    
                    if file_type in ['image', 'text']:
                        # Process and translate this file
                        file_result = self.translate_file(file_path, source_language)
                        if file_result['status'] == 'success':
                            translated_count += 1
                            results.append({
                                'file': Path(file_path).name,
                                'translated': file_result['translated'][:100]
                            })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue
            
            # Clean up extracted files
            self.zip_processor.cleanup()
            
            # Prepare summary
            summary_original = (
                f"ZIP Archive: {Path(zip_path).name}\n"
                f"Total files: {len(extracted_files)}\n"
                f"Processed: {translated_count} files"
            )
            
            summary_translated = "Translated contents:\n"
            for r in results[:5]:  # Show first 5
                summary_translated += f"- {r['file']}: {r['translated']}\n"
            
            return {
                'original': summary_original,
                'translated': summary_translated[:500],
                'status': 'success',
                'method': f'ZIP ({translated_count}/{len(extracted_files)} files)'
            }
            
        except Exception as e:
            return {
                'original': f'ZIP processing error',
                'translated': f'Error: {str(e)}',
                'status': 'error',
                'method': 'ZIP Error'
            }
    
    def process_batch(
        self,
        file_paths: List[str],
        source_language: str = 'auto'
    ) -> List[Dict[str, Any]]:
        """Process multiple files with translation."""
        results = []
        for path in file_paths:
            try:
                result = self.translate_file(path, source_language)
                results.append(result)
            except Exception as e:
                results.append({
                    'original': f'Error with {Path(path).name}',
                    'translated': str(e),
                    'status': 'error'
                })
        return results
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.ready and self.file_handler:
            try:
                self.file_handler.temp_file_manager('cleanup', cleanup_all=True)
            except:
                pass

# Create global instance
translator_service = TranslatorService()

# Test translation capability
if GOOGLE_TRANSLATE_AVAILABLE:
    test_text = "测试"  # Chinese for "test"
    try:
        result = GoogleTranslator(source='zh-CN', target='en').translate(test_text)
        print(f"✅ Translation test successful: {test_text} → {result}")
    except Exception as e:
        print(f"⚠️ Translation test failed: {e}")
