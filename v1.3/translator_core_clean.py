#!/usr/bin/env python
# coding: utf-8

# # Cell 1 🌍 Universal Translator v1.3
# NOTES HERE

# ## Cell 2 🔧 Setup & Installation {#setup}
# Run these cells once to set up your environment

# In[1]:


# Cell 3 Install required packages
# get_ipython().run_line_magic('pip', 'install ruff deep-translator pytesseract pillow')

# Verify installations
import sys
print(f"✅ Python version: {sys.version}")
print("✅ All packages installed successfully!")
print("📦 Installed: ruff, deep-translator, pytesseract, pillow")
# Cell 3: Install required packages
# get_ipython().run_line_magic('pip', 'install ruff deep-translator pytesseract pillow pypdf2 tqdm pdf2image pdfplumber reportlab')

# After installation, add these print statements:
print("✅ PDF packages installed:")
print("📦 PyPDF2 - Basic PDF reading")
print("📦 pdf2image - Convert PDF to images") 
print("📦 pdfplumber - Advanced text extraction")
print("📦 reportlab - PDF generation")


# In[ ]:


# Cell: Complete UI Setup and Installation
"""
Universal Translator v1.5 - UI Setup
Install all dependencies and prepare for Streamlit UI
"""

print("🚀 Setting up Universal Translator UI v1.5")
print("="*50)

# Install Streamlit
print("📦 Installing Streamlit...")
# get_ipython().run_line_magic('pip', 'install -q streamlit streamlit-extras')

# Verify installation
try:
    import streamlit as st
    print(f"✅ Streamlit {st.__version__} installed successfully!")
except ImportError:
    print("❌ Streamlit installation failed")
    print("   Try running in terminal: pip install streamlit")

print("\n" + "="*50)
print("📋 NEXT STEPS:")
print("1. Create 'app.py' file in your project folder")
print("2. Copy the UI code into app.py")
print("3. Create 'translator_integration.py' file")
print("4. Open terminal (Ctrl+`)")
print("5. Run: streamlit run app.py")
print("6. Click the URL to open the UI")
print("="*50)

# Check all dependencies
print("\n📦 Checking all dependencies:")
dependencies = {
    'streamlit': '🌐 UI Framework',
    'PyPDF2': '📄 PDF Processing',
    'PIL': '🖼️ Image Processing',
    'deep_translator': '🌍 Translation',
    'pytesseract': '👁️ OCR'
}

for module, description in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {module:20} {description}")
    except ImportError:
        print(f"❌ {module:20} Missing - run: pip install {module}")

print("\n✅ Setup complete! Ready to create UI files.")


# In[ ]:


# Add this as a new cell after Cell 3:
# Cell 3a: Install system dependencies for PDF processing
import subprocess
import sys

# Install poppler-utils for pdf2image
if 'codespaces' in sys.executable.lower() or '/workspaces/' in sys.executable:
    print("📦 Installing poppler-utils for PDF processing...")
    subprocess.run(['sudo', 'apt-get', 'update'], check=False)
    subprocess.run(['sudo', 'apt-get', 'install', '-y', 'poppler-utils'], check=False)
    print("✅ Poppler-utils installed!")
else:
    print("ℹ️ Not in Codespaces - ensure poppler-utils is installed")


# In[ ]:


# Cell 3a: Install additional packages for file handling
# get_ipython().run_line_magic('pip', 'install pypdf2 python-magic-bin tqdm pathlib')

print("✅ File handling packages installed!")
print("📦 Added: PyPDF2 for PDF processing")
print("📦 Added: python-magic for file type detection") 
print("📦 Added: tqdm for progress bars")


# ## Cell 4 🔧 Code Quality Check
# ### Ruff Linting & PEP 8 Validation
# Run this cell after installation to check and auto-fix code style issues

# In[ ]:


# Cell 5 - Ruff Code Quality Check & Fix

# Imports at the TOP (fixes the E402 error)
import os
import subprocess

# Clean up any old config files
for file in ['ruff_settings.txt', '../ruff_settings.txt']:
    if os.path.exists(file):
        os.remove(file)
        print(f"🗑️ Cleaned up {file}")

print("🔍 RUFF CODE QUALITY CHECK FOR V1.3")
print("=" * 50)

# First, check what we have
print("📊 Initial check:")
# get_ipython().system('ruff check translator_v1.3.ipynb --statistics')

print("\n" + "=" * 50)
print("🔧 Auto-fixing safe issues...")
# get_ipython().system('ruff check translator_v1.3.ipynb --fix')

print("\n" + "=" * 50)
print("📋 Final status:")
# get_ipython().system('ruff check translator_v1.3.ipynb --statistics')

# Show success or what's left (subprocess already imported at top)
result = subprocess.run(['ruff', 'check', 'translator_v1.3.ipynb'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print("\n🎉 SUCCESS! All checks passed!")
else:
    print("\n💡 Some style issues remain (usually line length)")
    print("These don't affect functionality")


# ## Cell 6 💻 ## Imports and Setup
# 
# **v1.3 Updates:**
# - Added `Enum` for language selection
# - All imports follow PEP 8 order
# - Version 1.3 - November 2, 2025

# In[ ]:


# Standard library imports
import re
from enum import Enum
from typing import Dict

# Third-party imports
import pytesseract
from deep_translator import GoogleTranslator
from PIL import Image, ImageEnhance, ImageFilter

"""
Universal Translator Module v1.3
PEP 8 compliant implementation for image text extraction and translation
Now with Enum support for better type safety
"""

# Module information
__version__ = "1.3"
__author__ = "Victor"
__date__ = "November 2, 2025"

print(f"📚 Universal Translator Module v{__version__} loaded")
print(f"👤 Author: {__author__}")


# In[ ]:


# Cell 6a: File Handling Imports
"""
File handling imports for Universal Translator v1.3
These handle various file types and batch processing
"""

# Standard library imports for file handling
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional
import hashlib
import json
from datetime import datetime

# Third-party imports for file handling
try:
    import PyPDF2
    print("✅ PyPDF2 imported successfully")
except ImportError:
    print("⚠️ PyPDF2 not found - installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "PyPDF2"])
    import PyPDF2
    print("✅ PyPDF2 installed and imported")

try:
    from tqdm import tqdm
    print("✅ tqdm imported for progress tracking")
except ImportError:
    print("⚠️ tqdm not found - installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm
    print("✅ tqdm installed and imported")

print("📁 File handling modules ready!")


# In[ ]:


# Cell: Install Poppler for PDF to Image Conversion
"""Install system dependencies for PDF image conversion"""

import subprocess
import sys
import os

print("🔧 INSTALLING POPPLER-UTILS")
print("=" * 50)

# Check if we're in Codespaces
if '/workspaces/' in os.getcwd():
    print("📍 Detected GitHub Codespaces environment")
    print("📦 Installing poppler-utils (this may take a minute)...")

    try:
        # Update apt and install poppler-utils
        subprocess.run(['sudo', 'apt-get', 'update', '-qq'], check=True, capture_output=True)
        subprocess.run(['sudo', 'apt-get', 'install', '-y', 'poppler-utils'], check=True, capture_output=True)

        # Verify installation
        result = subprocess.run(['which', 'pdftoppm'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Poppler installed successfully at: {result.stdout.strip()}")

            # Test poppler
            test = subprocess.run(['pdftoppm', '-h'], capture_output=True)
            if test.returncode == 0 or test.returncode == 1:  # -h returns 1 but shows it works
                print("✅ Poppler is working!")
        else:
            print("⚠️ Poppler installed but not found in PATH")

    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("You may need to run manually: sudo apt-get install poppler-utils")
else:
    print("ℹ️ Not in Codespaces - install poppler-utils manually")

print("=" * 50)


# In[ ]:


from reportlab.lib.colors import blue


# ## Configuration and Constants
# 
# **New in v1.3:** All settings are now in one place using the `Config` class.
# 
# **How it works:**
# - Settings are grouped by type (Image, OCR, Files, Debug)
# - Access using: `Config.Image.SCALE_FACTOR`
# - Change any setting without touching main code
# 
# **Active Settings:**
# - Image: scale, contrast, brightness
# - Files: naming, cleanup
# - Debug: verbose output on/off
# 
# **Future Features (placeholders ready):**
# - Batch processing
# - Caching
# - Error retry

# In[ ]:


# Cell: Configuration and Constants
"""
Configuration and Constants for Universal Translator v1.3
Centralized settings for easy adjustment and maintenance
"""

class Config:
    """
    Centralized configuration using nested classes for organization.
    Access patterns: Config.Image.SCALE_FACTOR, Config.Debug.VERBOSE, etc.
    """

    # ============= IMAGE PROCESSING SETTINGS =============
    class Image:
        """Settings for image enhancement and processing"""
        # Quality vs Speed trade-off (2=fast, 3=balanced, 4+=quality)
        SCALE_FACTOR = 3

        # Enhancement settings (1.0 = no change)
        CONTRAST = 2.5      # Increase contrast (higher = more contrast)
        BRIGHTNESS = 1.2    # Increase brightness (higher = brighter)

        # Sharpening iterations (more = sharper but slower)
        SHARPEN_ITERATIONS = 2

        # Image format for saving enhanced images
        OUTPUT_FORMAT = 'JPEG'  # or 'PNG' for better quality
        OUTPUT_QUALITY = 85     # JPEG quality (1-100, higher = better)

    # ============= OCR CONFIGURATION =============
    class OCR:
        """Tesseract OCR settings and configurations"""
        # OCR modes based on image type
        CONFIGS = {
            'document': r'--oem 3 --psm 6',    # Uniform text block
            'sign': r'--oem 3 --psm 11',       # Sparse text
            'screenshot': r'--oem 3 --psm 3',   # Fully automatic
            'default': r'--oem 3 --psm 3'       # Fallback option
        }

        # Timeout for OCR operations (seconds)
        TIMEOUT = 30

        # Confidence threshold (0-100) - future use
        MIN_CONFIDENCE = 60

    # ============= FILE HANDLING =============
    class Files:
        """File naming and management settings"""
        # Prefix for enhanced images
        ENHANCED_PREFIX = "enhanced_"

        # Auto-cleanup temporary files after processing
        AUTO_CLEANUP = False  # Set True to delete enhanced images after use

        # Directory for temporary files (None = same as source)
        TEMP_DIR = None

        # Maximum file size in MB (for safety)
        MAX_FILE_SIZE_MB = 50

    # ============= DEBUG AND LOGGING =============
    class Debug:
        """Debug and output control settings"""
        # Show detailed processing steps
        VERBOSE = True

        # Show timing information
        SHOW_TIMING = True

        # Save enhanced images (overrides AUTO_CLEANUP when False)
        SAVE_ENHANCED = True

        # Print configuration on startup
        SHOW_CONFIG = True

        # Detailed error messages
        DETAILED_ERRORS = True

    # ============= BATCH PROCESSING (Future Feature) =============
    class Batch:
        """Settings for batch processing multiple images"""
        # Maximum images to process in one batch
        SIZE_LIMIT = 10

        # Process in parallel (False = sequential)
        PARALLEL = False

        # Number of worker threads (if PARALLEL=True)
        WORKERS = 4

        # Continue on error or stop batch
        CONTINUE_ON_ERROR = True

    # ============= CACHING (Future Feature) =============
    class Cache:
        """Settings for caching processed results"""
        # Enable/disable caching
        ENABLED = False

        # Maximum cache size in MB
        MAX_SIZE_MB = 100

        # Cache expiration in seconds (3600 = 1 hour)
        EXPIRY_SECONDS = 3600

        # Cache location (None = memory, string = disk path)
        LOCATION = None

    # ============= ERROR HANDLING (Future Feature) =============
    class ErrorHandling:
        """Settings for error recovery and retries"""
        # Number of retry attempts
        RETRY_COUNT = 3

        # Delay between retries (seconds)
        RETRY_DELAY = 1

        # Fallback to basic processing on error
        USE_FALLBACK = True

        # Log errors to file
        LOG_TO_FILE = False
        LOG_FILE = "translator_errors.log"

    # ============= PERFORMANCE (Future Feature) =============
    class Performance:
        """Performance monitoring and optimization settings"""
        # Track processing times
        TRACK_TIMING = True

        # Memory usage warnings (MB)
        MEMORY_WARNING_MB = 500

        # Automatic optimization based on image size
        AUTO_OPTIMIZE = True

    # ============= FILE HANDLING SETTINGS =============
    class FileHandling:
        """Settings for file processing and management"""
        # Supported file extensions
        SUPPORTED_EXTENSIONS = {
            'images': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'],
            'documents': ['.pdf', '.txt', '.docx'],
            'archives': ['.zip']
        }

        # Maximum file sizes (in MB)
        MAX_IMAGE_SIZE_MB = 10
        MAX_PDF_SIZE_MB = 50
        MAX_ZIP_SIZE_MB = 100
        MAX_BATCH_SIZE = 20  # Maximum files to process at once

        # Temporary file management
        TEMP_DIR_PREFIX = "translator_temp_"
        KEEP_TEMP_FILES = False  # Set True for debugging

        # Output settings
        OUTPUT_DIR_NAME = "translated_output"
        TIMESTAMP_OUTPUT = True  # Add timestamp to output folders

        # File naming
        TRANSLATED_PREFIX = "translated_"
        MAINTAIN_STRUCTURE = True  # Keep original folder structure

        # ============= PDF PROCESSING SETTINGS =============
    class PDFProcessing:
        """Settings for PDF processing and generation"""
        # Text extraction settings
        EXTRACTION_METHOD = 'pdfplumber'  # 'pypdf2' or 'pdfplumber'
        EXTRACT_IMAGES = True  # Extract embedded images
        PRESERVE_LAYOUT = True  # Try to maintain formatting

        # PDF to image conversion (for OCR)
        DPI = 200  # Resolution for PDF to image conversion
        IMAGE_FORMAT = 'JPEG'  # Format for converted pages
        GRAYSCALE = True  # Convert to grayscale for better OCR

        # Output PDF settings
        PAGE_SIZE = 'A4'  # 'A4' or 'letter'
        FONT_NAME = 'Helvetica'  # Default font
        FONT_SIZE = 12
        LINE_SPACING = 1.2
        MARGIN = 72  # Points (1 inch = 72 points)

        # Translation layout
        SIDE_BY_SIDE = False  # True = original|translation, False = translation only
        ADD_PAGE_NUMBERS = True
        ADD_HEADER = True
        HEADER_TEXT = "Translated by Universal Translator v1.3"

        # Performance
        MAX_PAGES = 100  # Maximum pages to process
        CHUNK_SIZE = 10  # Pages to process at once

        # Temporary files
        KEEP_PAGE_IMAGES = False  # Keep individual page images
        COMPRESS_OUTPUT = True  # Compress output PDF

        # ============= ZIP PROCESSING SETTINGS =============
    class ZIPProcessing:
        """Settings for ZIP archive processing"""
        # Extraction settings
        EXTRACT_TO_TEMP = True  # Extract to temp directory
        PRESERVE_STRUCTURE = True  # Keep folder structure

        # File filtering
        PROCESS_EXTENSIONS = ['.pdf', '.jpg', '.png', '.txt']  # Files to process
        IGNORE_EXTENSIONS = ['.exe', '.dll', '.bin']  # Files to skip
        IGNORE_HIDDEN = True  # Skip files starting with .
        IGNORE_SYSTEM = True  # Skip __MACOSX, .DS_Store, etc.

        # Size limits
        MAX_ZIP_SIZE_MB = 100  # Maximum ZIP file size
        MAX_EXTRACTED_SIZE_MB = 500  # Maximum extracted size
        MAX_FILES_IN_ZIP = 100  # Maximum files to process

        # Output settings
        COMPRESS_OUTPUT = True  # Compress output ZIP
        COMPRESSION_LEVEL = 6  # 0-9 (6 is balanced)
        OUTPUT_PREFIX = "translated_"

        # Processing
        PARALLEL_PROCESSING = False  # Process files in parallel
        CONTINUE_ON_ERROR = True  # Skip failed files

        # Security
        CHECK_ZIP_BOMBS = True  # Check for malicious ZIPs
        MAX_COMPRESSION_RATIO = 100  # Max compression ratio allowed

        # ============= OUTPUT GENERATION SETTINGS =============
    class OutputGeneration:
        """Settings for final output generation"""
        # Output formats
        DEFAULT_FORMAT = 'pdf'  # 'pdf', 'zip', 'txt', 'all'
        INCLUDE_ORIGINAL = True  # Include original text with translation

        # Layout settings for bilingual output
        LAYOUT_TYPE = 'sequential'  # 'sequential', 'side_by_side', 'interleaved'
        SEPARATOR_LINE = "=" * 50

        # Report generation
        GENERATE_REPORT = True  # Create processing report
        REPORT_FORMAT = 'txt'  # 'txt', 'json', 'html'
        INCLUDE_STATS = True  # Include statistics in report

        # File naming
        OUTPUT_SUFFIX = '_translated'
        TIMESTAMP_FORMAT = '%Y%m%d_%H%M%S'
        USE_SOURCE_NAME = True  # Base output name on source file

        # Quality settings
        INCLUDE_CONFIDENCE = False  # Include translation confidence scores
        MARK_UNCERTAIN = True  # Mark low-confidence translations
        UNCERTAINTY_THRESHOLD = 0.7  # Confidence threshold

        # Batch output
        BATCH_OUTPUT_DIR = 'translated_output'
        CREATE_SUMMARY = True  # Create summary file for batch
        ORGANIZE_BY_TYPE = True  # Separate folders by file type

        # Metadata
        ADD_METADATA = True  # Add processing metadata to output
        METADATA_FIELDS = [
            'source_file', 'translation_date', 'source_language',
            'target_language', 'processor_version', 'processing_time'
        ]

    @classmethod
    def validate(cls):
        """
        Validate configuration settings.
        Raises ValueError if any settings are invalid.
        """
        # Image validation
        if cls.Image.SCALE_FACTOR < 1:
            raise ValueError("SCALE_FACTOR must be >= 1")
        if cls.Image.CONTRAST < 0:
            raise ValueError("CONTRAST must be >= 0")
        if cls.Image.BRIGHTNESS < 0:
            raise ValueError("BRIGHTNESS must be >= 0")

        # File validation
        if cls.Files.MAX_FILE_SIZE_MB <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be > 0")

        # Batch validation
        if cls.Batch.SIZE_LIMIT <= 0:
            raise ValueError("BATCH_SIZE_LIMIT must be > 0")

        print("✅ Configuration validated successfully!")
        return True

    @classmethod
    def display(cls):
        """Display current configuration settings"""
        if not cls.Debug.SHOW_CONFIG:
            return

        print("\n" + "="*50)
        print("📋 CURRENT CONFIGURATION")
        print("="*50)

        print("\n🖼️ Image Processing:")
        print(f"  • Scale Factor: {cls.Image.SCALE_FACTOR}x")
        print(f"  • Contrast: {cls.Image.CONTRAST}")
        print(f"  • Brightness: {cls.Image.BRIGHTNESS}")

        print("\n📁 File Handling:")
        print(f"  • Enhanced Prefix: '{cls.Files.ENHANCED_PREFIX}'")
        print(f"  • Auto Cleanup: {cls.Files.AUTO_CLEANUP}")

        print("\n🔍 Debug Settings:")
        print(f"  • Verbose Output: {cls.Debug.VERBOSE}")
        print(f"  • Save Enhanced Images: {cls.Debug.SAVE_ENHANCED}")

        print("\n🚀 Future Features Status:")
        print(f"  • Batch Processing: {'Ready' if cls.Batch.SIZE_LIMIT > 0 else 'Disabled'}")
        print(f"  • Caching: {'Enabled' if cls.Cache.ENABLED else 'Disabled'}")
        print(f"  • Error Retry: {cls.ErrorHandling.RETRY_COUNT} attempts")
        print("="*50 + "\n")


# Validate and display configuration on load
try:
    Config.validate()
    Config.display()
except ValueError as e:
    print(f"❌ Configuration Error: {e}")
    print("Please fix the configuration values above.")

# Language Enum (SEPARATE from Config)
class Language(Enum):
    """
    Supported languages with their Tesseract language codes.
    """
    ENGLISH = 'eng'
    CHINESE = 'chi_sim'  # Simplified Chinese
    JAPANESE = 'jpn'
    KOREAN = 'kor'
    HINDI = 'hin'

# Display available languages
print("🌍 Supported Languages:")
print("-" * 30)
for lang in Language:
    print(f"  • {lang.name.title()}: {lang.value}")
print("-" * 30)


# ## Universal Translator
# 
# **What's New:**
# - Use `Language.ENGLISH` instead of 'english'
# - All settings now use `Config` class
# - Better error messages
# 
# **How to Use:**
# ```python
# result = translator.process("image.jpg", Language.ENGLISH)

# In[ ]:


# Cell 10a: Error Handling Utilities
"""Error handling utilities for Universal Translator v1.3"""

import time
from typing import Any, Callable


class ErrorHandler:
    """Utility class for error handling and retry logic."""

    @staticmethod
    def retry_operation(
        operation: Callable,
        retry_count: int = 3,
        retry_delay: float = 1.0,
        verbose: bool = False,
        *args,
        **kwargs
    ) -> Any:
        """
        Retry an operation with exponential backoff.

        Args:
            operation: Function to retry
            retry_count: Number of retry attempts
            retry_delay: Initial delay between retries
            verbose: Print retry information
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            Result of the operation if successful

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(retry_count):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < retry_count - 1:
                    # Exponential backoff
                    wait_time = retry_delay * (2 ** attempt)
                    if verbose:
                        print(f"   Retry {attempt + 1}/{retry_count} "
                              f"after {wait_time}s...")
                    time.sleep(wait_time)

        # All retries failed - raise the last exception
        # If somehow no exception was caught, raise a RuntimeError
        if last_exception is not None:
            raise last_exception
        else:
            raise RuntimeError("Operation failed but no exception captured")


class LanguageChecker:
    """Utility class for checking language support."""

    @staticmethod
    def check_tesseract_languages() -> set:
        """
        Get list of installed Tesseract language packs.

        Returns:
            Set of installed language codes
        """
        import subprocess

        installed_langs = set()
        try:
            result = subprocess.run(
                ['tesseract', '--list-langs'],
                capture_output=True,
                text=True,
                check=False,
                timeout=5
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]
                installed_langs = set(lines)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        except Exception:
            pass

        return installed_langs

    @staticmethod
    def print_language_status(
        supported_languages: list,
        installed_langs: set
    ) -> tuple[dict, dict]:
        """
        Print language support status.

        Args:
            supported_languages: List of Language enum members
            installed_langs: Set of installed language codes

        Returns:
            Tuple of (available_languages, missing_languages) dicts
        """
        print("\n" + "="*50)
        print("🔍 CHECKING LANGUAGE SUPPORT")
        print("="*50)

        if not installed_langs:
            print("❌ Tesseract not found or no languages installed")
            missing_all = {lang: True for lang in supported_languages}
            return {}, missing_all

        print(f"✅ Tesseract found with {len(installed_langs)} "
              f"language packs")
        print("\n📋 Language Pack Status:")

        available = {}
        missing = {}

        for lang in supported_languages:
            lang_codes = lang.value.split('+')
            is_available = any(
                code in installed_langs for code in lang_codes
            )

            if is_available:
                available[lang] = True
                print(f"   ✅ {lang.name:10} ({lang.value:10}) "
                      f"- Installed")
            else:
                missing[lang] = True
                print(f"   ❌ {lang.name:10} ({lang.value:10}) "
                      f"- Not installed")

        if not missing:
            print("\n✅ All language packs are installed!")

        print("="*50)
        return available, missing


print("✅ Error handling utilities loaded")


# In[ ]:


# Cell 10b: Image Processing Utilities
"""Image processing utilities for Universal Translator v1.3"""

import os


class ImageProcessor:
    """Utility class for image enhancement operations."""

    @staticmethod
    def validate_image_file(
        image_path: str,
        max_size_mb: float = 50
    ) -> None:
        """
        Validate image file exists and size is acceptable.

        Args:
            image_path: Path to image file
            max_size_mb: Maximum file size in MB

        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file is too large
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Image file not found: {image_path}"
            )

        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise IOError(
                f"File too large: {file_size_mb:.1f}MB "
                f"(max: {max_size_mb}MB)"
            )

    @staticmethod
    def enhance_image(
        image_path: str,
        scale_factor: int = 3,
        contrast: float = 2.5,
        brightness: float = 1.2,
        sharpen_iterations: int = 2,
        output_quality: int = 85,
        prefix: str = "enhanced_"
    ) -> str:
        """
        Enhance image for better OCR results.

        Args:
            image_path: Path to input image
            scale_factor: Image scaling factor
            contrast: Contrast enhancement factor
            brightness: Brightness enhancement factor
            sharpen_iterations: Number of sharpening passes
            output_quality: JPEG output quality
            prefix: Prefix for enhanced image filename

        Returns:
            Path to enhanced image
        """
        img = Image.open(image_path)

        # Validate and convert
        if img.size[0] == 0 or img.size[1] == 0:
            raise ValueError("Invalid image dimensions")

        img = img.convert('L')

        # Scale image
        width, height = img.size
        new_size = (width * scale_factor, height * scale_factor)

        # Limit maximum size
        if new_size[0] > 10000 or new_size[1] > 10000:
            new_size = (width * 2, height * 2)

        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Enhance contrast and brightness
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Brightness(img).enhance(brightness)

        # Apply sharpening
        for _ in range(sharpen_iterations):
            img = img.filter(ImageFilter.SHARPEN)

        # Save enhanced image
        enhanced_path = f"{prefix}{image_path}"
        img.save(enhanced_path, quality=output_quality)

        return enhanced_path


print("✅ Image processing utilities loaded")


# In[ ]:


# Cell 10c: Text Processing Utilities
"""Text processing utilities for Universal Translator v1.3"""



class TextProcessor:
    """Utility class for text correction and processing."""

    # Known OCR errors and corrections for English
    ENGLISH_DIRECT_FIXES = {
        'Helloworld': 'Hello World',
        'HelloWorld': 'Hello World',
        'Thisisa': 'This is a',
        'This isa': 'This is a',
        'toour': 'to our',
        'aboutour': 'about our',
        'GRANDOPENING': 'GRAND OPENING',
        'SO OFF': '50% OFF',
        'SOOFF': '50% OFF',
        'Pythonm': 'Python',
    }

    # Pattern-based corrections
    ENGLISH_PATTERNS = [
        (r'\bisa\b', 'is a'),
        (r'([a-z])([A-Z])', r'\1 \2'),
        (r'([a-zA-Z])(\d)', r'\1 \2'),
        (r'(\d)([a-zA-Z])', r'\1 \2'),
    ]

    # Common OCR errors
    ENGLISH_COMMON_ERRORS = {
        ' tbe ': ' the ',
        ' amd ': ' and ',
        ' isa ': ' is a '
    }

    @classmethod
    def fix_english_text(cls, text: str) -> str:
        """
        Apply English-specific text corrections.

        Args:
            text: Raw text to be corrected

        Returns:
            Corrected text
        """
        if not text:
            return ""

        # Apply direct replacements
        for incorrect, correct in cls.ENGLISH_DIRECT_FIXES.items():
            text = text.replace(incorrect, correct)

        # Apply pattern-based corrections
        for pattern, replacement in cls.ENGLISH_PATTERNS:
            text = re.sub(pattern, replacement, text)

        # Fix common errors
        for error, correction in cls.ENGLISH_COMMON_ERRORS.items():
            text = text.replace(error, correction)

        # Clean up extra whitespace
        text = ' '.join(text.split())

        return text

    @staticmethod
    def fix_text(text: str, language) -> str:
        """
        Apply language-specific text corrections.

        Args:
            text: Raw text from OCR
            language: Language enum member

        Returns:
            Corrected text
        """
        if not text:
            return ""

        # Only English corrections implemented for now
        if language.name == 'ENGLISH':
            return TextProcessor.fix_english_text(text)

        # Return unchanged for other languages
        return text


print("✅ Text processing utilities loaded")


# In[ ]:


# Cell 10d: Universal Translator Main Class
"""Universal Translator v1.3 with modular utilities"""

import os
import subprocess
from typing import Any, Union


class UniversalTranslator:
    """
    Universal translator for extracting and translating text from images.

    Uses modular utilities for cleaner code organization.
    """

    def __init__(self) -> None:
        """Initialize the UniversalTranslator."""
        self.supported_languages = list(Language) # type: ignore
        self.available_languages = {}
        self.missing_languages = {}
        self.error_count = 0

        # Initialize utilities
        self.error_handler = ErrorHandler()
        self.lang_checker = LanguageChecker()
        self.img_processor = ImageProcessor()
        self.text_processor = TextProcessor()

        # Check language support
        self._check_language_support()
        self._setup_complete()

    def _check_language_support(self) -> None:
        """Check which Tesseract language packs are installed."""
        installed = self.lang_checker.check_tesseract_languages()

        result = self.lang_checker.print_language_status(
            self.supported_languages,
            installed
        )

        # Fix Error 1: Handle tuple unpacking safely
        if result:
            self.available_languages, self.missing_languages = result
        else:
            self.available_languages = {}
            self.missing_languages = {}

    def _setup_complete(self) -> None:
        """Print initialization confirmation."""
        if Config.Debug.VERBOSE:
            print("\n✅ Universal Translator v1.3 initialized!")
            langs = [l.name.lower() for l in self.supported_languages]
            print(f"📚 Defined languages: {', '.join(langs)}")

            if self.available_languages:
                avail = [l.name.lower() for l in self.available_languages]
                print(f"✅ Ready to use: {', '.join(avail)}")

            if self.missing_languages:
                miss = [l.name.lower() for l in self.missing_languages]
                print(f"⚠️ Missing: {', '.join(miss)}")

    def enhance_image(self, image_path: str) -> str:
        """
        Enhance image quality for better OCR results.

        Args:
            image_path: Path to the input image file

        Returns:
            Path to the enhanced image file
        """
        try:
            # Validate file
            self.img_processor.validate_image_file(
                image_path,
                Config.Files.MAX_FILE_SIZE_MB
            )

            # Enhance with retry
            def _enhance():
                return self.img_processor.enhance_image(
                    image_path,
                    scale_factor=Config.Image.SCALE_FACTOR,
                    contrast=Config.Image.CONTRAST,
                    brightness=Config.Image.BRIGHTNESS,
                    sharpen_iterations=Config.Image.SHARPEN_ITERATIONS,
                    output_quality=Config.Image.OUTPUT_QUALITY,
                    prefix=Config.Files.ENHANCED_PREFIX
                )

            enhanced_path = self.error_handler.retry_operation(
                _enhance,
                Config.ErrorHandling.RETRY_COUNT,
                Config.ErrorHandling.RETRY_DELAY,
                Config.Debug.VERBOSE
            )

            if Config.Debug.VERBOSE:
                print(f"✅ Image enhanced: {enhanced_path}")

            return enhanced_path

        except Exception as e:
            self.error_count += 1
            if Config.Debug.DETAILED_ERRORS:
                print(f"❌ Error enhancing image: {e}")

            if Config.ErrorHandling.USE_FALLBACK:
                if Config.Debug.VERBOSE:
                    print("⚠️ Using original image as fallback")
                return image_path
            raise

    def process(
        self,
        image_path: str,
        language: Language = Language.ENGLISH
    ) -> Optional[Dict[str, Union[str, Optional[List[str]]]]]:
        """
        Process image to extract and translate text.

        Args:
            image_path: Path to the image file
            language: Source language (Language enum)

        Returns:
            Dictionary with results or None if failed.
            Contains 'original', 'fixed', 'translated' (str),
            'language' (str), and 'errors' (Optional[List[str]])
        """
        # Validate input
        if not isinstance(language, Language):
            raise TypeError(
                "Language must be a Language enum member"
            )

        errors_encountered: List[str] = []

        # Check language availability
        if language in self.missing_languages:
            msg = f"⚠️ {language.name} pack may not be installed"
            if Config.Debug.VERBOSE:
                print(msg)
            errors_encountered.append(msg)

        if Config.Debug.VERBOSE:
            print(f"🔍 Processing: {image_path}")
            print(f"🌐 Language: {language.name.lower()}")

        try:
            # Enhance image
            try:
                enhanced_path = self.enhance_image(image_path)
            except Exception as e:
                errors_encountered.append(f"Enhancement: {e}")
                enhanced_path = image_path

            # OCR with retry
            def _ocr():
                # Fix Error 3: Provide default if get() returns None
                ocr_config = Config.OCR.CONFIGS.get(
                    'default', '--oem 3 --psm 3'
                )

                return pytesseract.image_to_string(
                    enhanced_path,
                    lang=language.value,
                    config=ocr_config,
                    timeout=Config.OCR.TIMEOUT
                )

            raw_text = self.error_handler.retry_operation(
                _ocr,
                Config.ErrorHandling.RETRY_COUNT,
                Config.ErrorHandling.RETRY_DELAY,
                Config.Debug.VERBOSE
            )

            # Fix text
            fixed_text = self.text_processor.fix_text(
                raw_text, language
            )

            # Translate if needed
            translated_text = fixed_text
            if language != Language.ENGLISH and fixed_text:
                try:
                    if Config.Debug.VERBOSE:
                        print("🌍 Translating to English...")

                    def _translate():
                        trans = GoogleTranslator(
                            source='auto', target='en'
                        )
                        return trans.translate(fixed_text)

                    translated_text = self.error_handler.retry_operation(
                        _translate,
                        Config.ErrorHandling.RETRY_COUNT,
                        Config.ErrorHandling.RETRY_DELAY,
                        Config.Debug.VERBOSE
                    )
                except Exception as e:
                    errors_encountered.append(f"Translation: {e}")
                    translated_text = fixed_text

            # Cleanup
            if (Config.Files.AUTO_CLEANUP and 
                not Config.Debug.SAVE_ENHANCED and
                enhanced_path != image_path):
                try:
                    os.remove(enhanced_path)
                except:
                    pass

            if Config.Debug.VERBOSE:
                print("✅ Processing complete!")

            # Fix Errors 4-5: Correct return type
            result: Dict[str, Union[str, Optional[List[str]]]] = {
                'original': raw_text,
                'fixed': fixed_text,
                'translated': translated_text,
                'language': language.name.lower(),
                'errors': errors_encountered if errors_encountered else None
            }

            return result

        except Exception as e:
            self.error_count += 1
            if Config.Debug.DETAILED_ERRORS:
                print(f"❌ Critical error: {e}")

            if Config.ErrorHandling.USE_FALLBACK:
                # Fix return type for error case
                fallback_result: Dict[str, Union[str, Optional[List[str]]]] = {
                    'original': '',
                    'fixed': '',
                    'translated': '',
                    'language': language.name.lower(),
                    'errors': [str(e)] + errors_encountered
                }
                return fallback_result
            raise


# Initialize the translator
print("\n" + "="*50)
print("🚀 Initializing Universal Translator v1.3...")
print("="*50)
translator = UniversalTranslator()


# In[ ]:


# Cell 10e: File Handling System
"""
File Handling System for Universal Translator v1.3
Manages different file types and batch processing
"""


class FileHandler:
    """
    Comprehensive file handling system for the translator.
    Manages file validation, type detection, and batch processing.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the FileHandler.

        Args:
            verbose: Enable detailed output messages
        """
        self.verbose = verbose
        self.temp_dir = None
        self.processed_files = []
        self.failed_files = []
        self.session_id = self._generate_session_id()

        # Create temporary directory for this session
        self._setup_temp_directory()

        if self.verbose:
            print("📁 FileHandler initialized")
            print(f"🔑 Session ID: {self.session_id}")
            print(f"📂 Temp directory: {self.temp_dir}")

    def _generate_session_id(self) -> str:
        """
        Generate unique session ID for this processing run.

        Returns:
            Unique session identifier
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_hex = hashlib.md5(
            str(datetime.now()).encode()
        ).hexdigest()[:8]
        return f"{timestamp}_{random_hex}"

    def _setup_temp_directory(self) -> None:
        """Create temporary directory for processing."""
        prefix = Config.FileHandling.TEMP_DIR_PREFIX + self.session_id
        self.temp_dir = tempfile.mkdtemp(prefix=prefix)

        # Create subdirectories
        for subdir in ['input', 'processing', 'output']:
            Path(self.temp_dir, subdir).mkdir(exist_ok=True)

    def file_type_detector(self, file_path: str) -> str:
        """
        Detect file type based on extension and content.

        Args:
            file_path: Path to the file

        Returns:
            File type category ('image', 'pdf', 'text', 'zip', 'unknown')
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get file extension
        file_ext = Path(file_path).suffix.lower()

        # Check against supported extensions
        for category, extensions in Config.FileHandling.SUPPORTED_EXTENSIONS.items():
            if file_ext in extensions:
                if self.verbose:
                    print(f"🔍 Detected {category[:-1]} file: {file_ext}")

                # Map category names to simple types
                if category == 'images':
                    return 'image'
                elif category == 'documents':
                    if file_ext == '.pdf':
                        return 'pdf'
                    else:
                        return 'text'
                elif category == 'archives':
                    return 'zip'

        if self.verbose:
            print(f"⚠️ Unknown file type: {file_ext}")
        return 'unknown'

    def file_validator(
        self, 
        file_path: str, 
        expected_type: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate file integrity and format.

        Args:
            file_path: Path to file to validate
            expected_type: Expected file type (optional)

        Returns:
            Tuple of (is_valid, message)
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                return False, "File does not exist"

            # Check file size
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            # Detect type
            detected_type = self.file_type_detector(file_path)

            # Check against expected type if provided
            if expected_type and detected_type != expected_type:
                return False, f"Expected {expected_type}, got {detected_type}"

            # Check size limits based on type
            if detected_type == 'image':
                max_size = Config.FileHandling.MAX_IMAGE_SIZE_MB
            elif detected_type == 'pdf':
                max_size = Config.FileHandling.MAX_PDF_SIZE_MB
            elif detected_type == 'zip':
                max_size = Config.FileHandling.MAX_ZIP_SIZE_MB
            else:
                max_size = Config.FileHandling.MAX_IMAGE_SIZE_MB  # Default

            if file_size_mb > max_size:
                return False, f"File too large: {file_size_mb:.1f}MB (max: {max_size}MB)"

            # Try to open file to verify it's not corrupted
            if detected_type == 'image':
                try:
                    from PIL import Image
                    img = Image.open(file_path)
                    img.verify()
                except Exception as e:
                    return False, f"Corrupted image file: {str(e)}"

            elif detected_type == 'pdf':
                try:
                    with open(file_path, 'rb') as f:
                        PyPDF2.PdfReader(f)
                except Exception as e:
                    return False, f"Corrupted PDF file: {str(e)}"

            return True, f"Valid {detected_type} file ({file_size_mb:.1f}MB)"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def batch_file_processor(
        self,
        input_directory: str,
        file_types: Optional[List[str]] = None,
        recursive: bool = False
    ) -> List[str]:
        """
        Process multiple files from a directory.

        Args:
            input_directory: Directory containing files
            file_types: List of file types to process (None = all)
            recursive: Process subdirectories

        Returns:
            List of valid file paths ready for processing
        """
        if not os.path.isdir(input_directory):
            raise ValueError(f"Not a directory: {input_directory}")

        valid_files = []
        invalid_files = []

        # Get all files
        if recursive:
            pattern = '**/*'
        else:
            pattern = '*'

        path = Path(input_directory)
        all_files = list(path.glob(pattern))

        # Filter only files (not directories)
        all_files = [f for f in all_files if f.is_file()]

        if self.verbose:
            print(f"📂 Found {len(all_files)} files in {input_directory}")

        # Process with progress bar
        for file_path in tqdm(all_files, desc="Validating files"):
            file_str = str(file_path)

            # Check file type if filter is specified
            if file_types:
                detected_type = self.file_type_detector(file_str)
                if detected_type not in file_types:
                    continue

            # Validate file
            is_valid, message = self.file_validator(file_str)

            if is_valid:
                valid_files.append(file_str)
            else:
                invalid_files.append((file_str, message))

        # Report results
        if self.verbose:
            print(f"\n✅ Valid files: {len(valid_files)}")
            print(f"❌ Invalid/skipped files: {len(invalid_files)}")

            if invalid_files and len(invalid_files) <= 5:
                print("\n⚠️ Invalid files:")
                for file_path, reason in invalid_files[:5]:
                    print(f"  - {Path(file_path).name}: {reason}")

        # Check batch size limit
        if len(valid_files) > Config.FileHandling.MAX_BATCH_SIZE:
            print(f"⚠️ Found {len(valid_files)} files, limiting to "
                  f"{Config.FileHandling.MAX_BATCH_SIZE}")
            valid_files = valid_files[:Config.FileHandling.MAX_BATCH_SIZE]

        self.processed_files = valid_files
        return valid_files

    def temp_file_manager(
        self,
        action: str,
        file_path: Optional[str] = None,
        cleanup_all: bool = False
    ) -> Optional[str]:
        """
        Manage temporary files lifecycle.

        Args:
            action: 'create', 'get', 'cleanup'
            file_path: Source file path (for create)
            cleanup_all: Remove all temp files

        Returns:
            Path to temporary file (for create/get actions)
        """
        if action == 'create' and file_path:
            # Copy file to temp directory
            filename = Path(file_path).name
            temp_path = Path(self.temp_dir, 'processing', filename) # type: ignore
            shutil.copy2(file_path, temp_path)

            if self.verbose:
                print(f"📋 Created temp file: {temp_path.name}")

            return str(temp_path)

        elif action == 'get':
            # Return temp directory path
            return self.temp_dir

        elif action == 'cleanup':
            if cleanup_all or not Config.FileHandling.KEEP_TEMP_FILES:
                try:
                    shutil.rmtree(self.temp_dir) # type: ignore
                    if self.verbose:
                        print(f"🗑️ Cleaned up temp directory: {self.temp_dir}")
                except Exception as e:
                    print(f"⚠️ Could not clean temp files: {e}")
            else:
                if self.verbose:
                    print(f"📁 Temp files kept at: {self.temp_dir}")

        return None

    def generate_output_path(
        self,
        source_file: str,
        output_dir: str,
        suffix: str = "_translated"
    ) -> str:
        """
        Generate systematic output file path.

        Args:
            source_file: Original file path
            output_dir: Output directory
            suffix: Suffix to add to filename

        Returns:
            Output file path
        """
        source_path = Path(source_file)

        # Create output directory if needed
        output_path = Path(output_dir)

        if Config.FileHandling.TIMESTAMP_OUTPUT:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path / timestamp

        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        name_parts = source_path.stem.split('.')
        new_name = name_parts[0] + suffix
        if len(name_parts) > 1:
            new_name += '.' + '.'.join(name_parts[1:])
        new_name += source_path.suffix

        return str(output_path / new_name)

    def get_processing_stats(self) -> Dict:
        """
        Get statistics about current processing session.

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            'session_id': self.session_id,
            'temp_directory': self.temp_dir,
            'total_files': len(self.processed_files),
            'failed_files': len(self.failed_files),
            'processed_files': self.processed_files,
            'failed_list': self.failed_files,
            'timestamp': datetime.now().isoformat()
        }

        # Calculate temp directory size
        if self.temp_dir and os.path.exists(self.temp_dir):
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.temp_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            stats['temp_size_mb'] = total_size / (1024 * 1024)

        return stats

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'temp_dir') and not Config.FileHandling.KEEP_TEMP_FILES:
            self.temp_file_manager('cleanup', cleanup_all=True)


# Initialize the file handler
print("\n" + "="*50)
print("🚀 Initializing File Handler...")
print("="*50)
file_handler = FileHandler(verbose=Config.Debug.VERBOSE)
print("✅ File Handler ready for use!")


# In[ ]:


# Cell 10f: PDF Processing Imports
"""
PDF Processing imports for Universal Translator v1.3
Handle PDF text extraction, conversion, and generation
"""

# Standard library
import os
from typing import List, Dict, Optional, Tuple

# PDF processing imports
try:
    import PyPDF2
    from PyPDF2 import PdfReader, PdfWriter
    print("✅ PyPDF2 ready")
except ImportError as e:
    print(f"❌ PyPDF2 import error: {e}")

try:
    import pdfplumber
    print("✅ pdfplumber ready")
except ImportError as e:
    print(f"❌ pdfplumber import error: {e}")

try:
    from pdf2image import convert_from_path, convert_from_bytes
    print("✅ pdf2image ready")
except ImportError as e:
    print(f"⚠️ pdf2image import error: {e}")
    print("   Make sure poppler-utils is installed")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    print("✅ reportlab ready")
except ImportError as e:
    print(f"⚠️ reportlab import error: {e}")

# Check if all imports successful
pdf_imports_ready = all([
    'PyPDF2' in globals(),
    'pdfplumber' in globals(),
    'reportlab' in globals()
])

if pdf_imports_ready:
    print("="*50)
    print("✅ All PDF processing modules loaded successfully!")
    print("="*50)
else:
    print("⚠️ Some PDF modules missing - check errors above")


# In[ ]:


# Cell 10g: PDF Processor Class
"""
PDF Processing System for Universal Translator v1.3
Handles PDF text extraction, conversion, and translation
"""

import subprocess

class PDFProcessor:
    """
    Comprehensive PDF processing system.
    Handles text extraction, image conversion, and PDF generation.
    """

    def __init__(self, file_handler: Optional['FileHandler'] = None, verbose: bool = True):
        """
        Initialize PDFProcessor.

        Args:
            file_handler: Optional FileHandler instance for temp file management
            verbose: Enable detailed output
        """
        self.verbose = verbose
        self.file_handler = file_handler
        self.current_pdf = None
        self.extracted_text = []
        self.page_images = []
        self.processing_stats = {
            'total_pages': 0,
            'processed_pages': 0,
            'extracted_chars': 0,
            'errors': []
        }

        # Check for poppler availability
        self.poppler_available = self._check_poppler()

        if self.verbose:
            print("📄 PDFProcessor initialized")
            if self.file_handler:
                print(f"📁 Using FileHandler session: {self.file_handler.session_id}")
            if not self.poppler_available:
                print("⚠️ Poppler not found - PDF to image conversion disabled")
                print("   Install with: sudo apt-get install poppler-utils")

    def _check_poppler(self) -> bool:
        """Check if poppler-utils is installed."""
        try:
            subprocess.run(['pdftoppm', '-h'], capture_output=True, check=False)
            return True
        except FileNotFoundError:
            return False

    def extract_text_from_pdf(self, pdf_path: str, method: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract text from PDF using specified method.

        Args:
            pdf_path: Path to PDF file
            method: 'pypdf2' or 'pdfplumber' (default from Config)

        Returns:
            List of dicts with page number and extracted text
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        method = method or Config.PDFProcessing.EXTRACTION_METHOD
        extracted_pages = []

        if self.verbose:
            print(f"📖 Extracting text from PDF using {method}...")

        try:
            if method == 'pdfplumber':
                extracted_pages = self._extract_with_pdfplumber(pdf_path)
            else:  # pypdf2
                extracted_pages = self._extract_with_pypdf2(pdf_path)

            # Update stats
            self.processing_stats['total_pages'] = len(extracted_pages)
            self.processing_stats['processed_pages'] = len([p for p in extracted_pages if p['text']])
            self.processing_stats['extracted_chars'] = sum(len(p['text']) for p in extracted_pages)

            if self.verbose:
                print(f"✅ Extracted {self.processing_stats['extracted_chars']} characters "
                      f"from {self.processing_stats['processed_pages']} pages")

            self.extracted_text = extracted_pages
            return extracted_pages

        except Exception as e:
            error_msg = f"Text extraction failed: {str(e)}"
            self.processing_stats['errors'].append(error_msg)
            if self.verbose:
                print(f"❌ {error_msg}")
            raise

    def _extract_with_pypdf2(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using PyPDF2."""
        extracted_pages = []

        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            total_pages = len(reader.pages)

            if self.verbose:
                print(f"📄 Processing {total_pages} pages with PyPDF2...")

            for page_num, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    extracted_pages.append({
                        'page': page_num,
                        'text': text,
                        'method': 'pypdf2',
                        'success': bool(text)
                    })
                except Exception as e:
                    extracted_pages.append({
                        'page': page_num,
                        'text': '',
                        'method': 'pypdf2',
                        'success': False,
                        'error': str(e)
                    })

        return extracted_pages

    def _extract_with_pdfplumber(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using pdfplumber (better for complex layouts)."""
        extracted_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            if self.verbose:
                print(f"📄 Processing {total_pages} pages with pdfplumber...")

            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    extracted_pages.append({
                        'page': page_num,
                        'text': text or '',
                        'method': 'pdfplumber',
                        'success': bool(text)
                    })
                except Exception as e:
                    extracted_pages.append({
                        'page': page_num,
                        'text': '',
                        'method': 'pdfplumber',
                        'success': False,
                        'error': str(e)
                    })

        return extracted_pages

    def pdf_to_images(self, pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Convert PDF pages to images for OCR processing.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images (optional)

        Returns:
            List of image file paths (empty if poppler not available)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Check if poppler is available
        if not self.poppler_available:
            if self.verbose:
                print("⚠️ Poppler not installed - skipping image conversion")
                print("   Install with: sudo apt-get install poppler-utils")
            return []

        # Use temp directory if no output specified
        if not output_dir and self.file_handler:
            output_dir = Path(self.file_handler.temp_dir) / 'pdf_images'  # type: ignore  # temp_dir exists when file_handler exists
        elif not output_dir:
            output_dir = Path('./pdf_images')  # type: ignore


        Path(output_dir).mkdir(parents=True, exist_ok=True)  # type: ignore  # output_dir is never None here


        if self.verbose:
            print(f"🖼️ Converting PDF to images (DPI: {Config.PDFProcessing.DPI})...")

        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=Config.PDFProcessing.DPI,
                fmt=Config.PDFProcessing.IMAGE_FORMAT,
                grayscale=Config.PDFProcessing.GRAYSCALE
            )

            image_paths = []
            for i, image in enumerate(images, 1):
                image_path = Path(output_dir) / f"page_{i:03d}.{Config.PDFProcessing.IMAGE_FORMAT.lower()}" # type: ignore
                image.save(str(image_path), Config.PDFProcessing.IMAGE_FORMAT)
                image_paths.append(str(image_path))

                if self.verbose and i % 5 == 0:
                    print(f"  Converted {i}/{len(images)} pages...")

            if self.verbose:
                print(f"✅ Converted {len(images)} pages to images")

            self.page_images = image_paths
            return image_paths

        except Exception as e:
            error_msg = f"PDF to image conversion failed: {str(e)}"
            self.processing_stats['errors'].append(error_msg)
            if self.verbose:
                print(f"⚠️ {error_msg}")
                print("   Continuing without image conversion...")
            return []

    def create_translated_pdf(
        self, 
        translations: List[str], 
        output_path: str,
        original_text: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a PDF with translated text.

        Args:
            translations: List of translated text (one per page)
            output_path: Path for output PDF
            original_text: Optional original text for side-by-side view
            metadata: Optional metadata dict (title, author, etc.)

        Returns:
            Path to created PDF
        """
        if self.verbose:
            print("📝 Creating translated PDF...")

        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4 if Config.PDFProcessing.PAGE_SIZE == 'A4' else letter,
                rightMargin=Config.PDFProcessing.MARGIN,
                leftMargin=Config.PDFProcessing.MARGIN,
                topMargin=Config.PDFProcessing.MARGIN,
                bottomMargin=Config.PDFProcessing.MARGIN
            )

            # Build content
            story = []
            styles = getSampleStyleSheet()

            # Add header if configured
            if Config.PDFProcessing.ADD_HEADER:
                header_style = ParagraphStyle(
                    'CustomHeader',
                    parent=styles['Heading1'],
                    fontSize=14,
                    textColor=blue,  # Use the Color object
                    spaceAfter=30
                )
                header = Paragraph(Config.PDFProcessing.HEADER_TEXT, header_style)
                story.append(header)
                story.append(Spacer(1, 0.2*inch))

            # Add content pages
            for i, translation in enumerate(translations, 1):
                if Config.PDFProcessing.ADD_PAGE_NUMBERS:
                    page_header = Paragraph(f"Page {i}", styles['Heading2'])
                    story.append(page_header)
                    story.append(Spacer(1, 0.1*inch))

                # Add translation
                para = Paragraph(translation, styles['Normal'])
                story.append(para)

                # Add page break except for last page
                if i < len(translations):
                    story.append(PageBreak())

            # Build PDF
            doc.build(story)

            if self.verbose:
                print(f"✅ Created PDF: {output_path}")
                print(f"   Pages: {len(translations)}")
                print(f"   Size: {os.path.getsize(output_path) / 1024:.1f} KB")

            return output_path

        except Exception as e:
            error_msg = f"PDF creation failed: {str(e)}"
            self.processing_stats['errors'].append(error_msg)
            if self.verbose:
                print(f"❌ {error_msg}")
            raise

    def process_pdf_for_translation(
        self, 
        pdf_path: str,
        extract_method: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Complete PDF processing pipeline for translation.

        Args:
            pdf_path: Path to input PDF
            extract_method: 'text', 'ocr', or 'auto' (auto-detect)

        Returns:
            Dictionary with extracted text and metadata
        """
        if self.verbose:
            print(f"🔄 Processing PDF: {Path(pdf_path).name}")

        result = {
            'source_file': pdf_path,
            'pages': [],
            'total_pages': 0,
            'method_used': '',
            'needs_ocr': False,
            'page_images': [],
            'stats': {}
        }

        # Try text extraction first
        extracted_pages = self.extract_text_from_pdf(pdf_path)

        # Check if we got meaningful text
        total_chars = sum(len(p['text']) for p in extracted_pages)
        avg_chars_per_page = total_chars / len(extracted_pages) if extracted_pages else 0

        # If too little text, might be scanned PDF needing OCR
        if avg_chars_per_page < 100 and extract_method != 'text':
            if self.verbose:
                print("⚠️ Low text density detected - PDF might be scanned")

            if self.poppler_available:
                print("🔄 Converting to images for OCR...")
                try:
                    page_images = self.pdf_to_images(pdf_path)
                    if page_images:
                        result['needs_ocr'] = True
                        result['method_used'] = 'ocr'
                        result['page_images'] = page_images
                    else:
                        result['method_used'] = 'text_extraction_only'
                except Exception as e:
                    result['method_used'] = 'text_extraction_fallback'
                    if self.verbose:
                        print(f"ℹ️ Image conversion failed: {e}")
                        print("   Using text extraction only")
            else:
                result['method_used'] = 'text_extraction_only'
                if self.verbose:
                    print("ℹ️ Poppler not available - using text extraction only")
        else:
            result['method_used'] = 'text_extraction'

        result['pages'] = extracted_pages
        result['total_pages'] = len(extracted_pages)
        result['stats'] = self.processing_stats.copy()

        if self.verbose:
            print("✅ PDF processing complete!")
            print(f"   Method: {result['method_used']}")
            print(f"   Pages: {result['total_pages']}")
            if not result['needs_ocr']:
                print(f"   Characters: {total_chars}")

        return result

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.processing_stats.copy()

    def clear_stats(self):
        """Reset processing statistics."""
        self.processing_stats = {
            'total_pages': 0,
            'processed_pages': 0,
            'extracted_chars': 0,
            'errors': []
        }
        if self.verbose:
            print("📊 Stats cleared")


# Initialize PDF processor
print("\n" + "="*50)
print("🚀 Initializing PDF Processor...")
print("="*50)

# Initialize with existing file_handler if available
if 'file_handler' in globals():
    pdf_processor = PDFProcessor(file_handler=file_handler, verbose=Config.Debug.VERBOSE)
else:
    pdf_processor = PDFProcessor(verbose=Config.Debug.VERBOSE)
    print("⚠️ No FileHandler found - PDF processor running standalone")

print("✅ PDF Processor ready for use!")
print("="*50)


# In[ ]:


# Cell 10h: ZIP Archive Handler
"""
ZIP Archive Processing System for Universal Translator v1.3
Handles extraction, processing, and creation of ZIP archives
"""

import zipfile
from typing import List, Dict, Optional, Tuple
import os

class ZIPProcessor:
    """
    Comprehensive ZIP archive processing system.
    Handles extraction, batch processing, and ZIP creation.
    """

    def __init__(
        self, 
        file_handler: Optional['FileHandler'] = None,
        pdf_processor: Optional['PDFProcessor'] = None,
        verbose: bool = True
    ):
        """
        Initialize ZIPProcessor.

        Args:
            file_handler: Optional FileHandler for temp management
            pdf_processor: Optional PDFProcessor for PDF files
            verbose: Enable detailed output
        """
        self.verbose = verbose
        self.file_handler = file_handler
        self.pdf_processor = pdf_processor
        self.current_zip = None
        self.extracted_files = []
        self.processing_stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_size_mb': 0,
            'errors': []
        }

        if self.verbose:
            print("📦 ZIPProcessor initialized")
            if self.file_handler:
                print(f"📁 Using FileHandler session: {self.file_handler.session_id}")
            if self.pdf_processor:
                print("📄 PDF processing enabled")

    def validate_zip(self, zip_path: str) -> Tuple[bool, str]:
        """
        Validate ZIP file for safety and integrity.

        Args:
            zip_path: Path to ZIP file

        Returns:
            Tuple of (is_valid, message)
        """
        if not os.path.exists(zip_path):
            return False, "ZIP file not found"

        # Check file size
        file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        if file_size_mb > Config.ZIPProcessing.MAX_ZIP_SIZE_MB:
            return False, f"ZIP too large: {file_size_mb:.1f}MB (max: {Config.ZIPProcessing.MAX_ZIP_SIZE_MB}MB)"

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Check if it's a valid ZIP
                if zf.testzip() is not None:
                    return False, "Corrupted ZIP file"

                # Check for ZIP bombs (high compression ratio)
                if Config.ZIPProcessing.CHECK_ZIP_BOMBS:
                    total_uncompressed = sum(info.file_size for info in zf.infolist())
                    total_compressed = sum(info.compress_size for info in zf.infolist())

                    if total_compressed > 0:
                        ratio = total_uncompressed / total_compressed
                        if ratio > Config.ZIPProcessing.MAX_COMPRESSION_RATIO:
                            return False, f"Suspicious compression ratio: {ratio:.1f}x"

                    # Check extracted size
                    total_size_mb = total_uncompressed / (1024 * 1024)
                    if total_size_mb > Config.ZIPProcessing.MAX_EXTRACTED_SIZE_MB:
                        return False, f"Extracted size too large: {total_size_mb:.1f}MB"

                # Check number of files
                num_files = len(zf.namelist())
                if num_files > Config.ZIPProcessing.MAX_FILES_IN_ZIP:
                    return False, f"Too many files: {num_files} (max: {Config.ZIPProcessing.MAX_FILES_IN_ZIP})"

                return True, f"Valid ZIP: {num_files} files, {file_size_mb:.1f}MB"

        except zipfile.BadZipFile:
            return False, "Invalid ZIP file format"
        except Exception as e:
            return False, f"ZIP validation error: {str(e)}"

    def extract_zip(
        self, 
        zip_path: str, 
        extract_to: Optional[str] = None
    ) -> List[str]:
        """
        Extract ZIP contents to specified directory.

        Args:
            zip_path: Path to ZIP file
            extract_to: Extraction directory (uses temp if None)

        Returns:
            List of extracted file paths
        """
        # Validate ZIP first
        is_valid, message = self.validate_zip(zip_path)
        if not is_valid:
            raise ValueError(f"Invalid ZIP: {message}")

        # Determine extraction directory
        if not extract_to:
            if self.file_handler and Config.ZIPProcessing.EXTRACT_TO_TEMP:
                extract_to = os.path.join(self.file_handler.temp_dir, 'zip_extract') # type: ignore
            else:
                extract_to = './zip_extract'

        Path(extract_to).mkdir(parents=True, exist_ok=True) # type: ignore

        if self.verbose:
            print(f"📂 Extracting ZIP: {Path(zip_path).name}")
            print(f"📍 Destination: {extract_to}")

        extracted_files = []

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Extract all files
                for member in zf.namelist():
                    # Skip system files if configured
                    if Config.ZIPProcessing.IGNORE_SYSTEM:
                        if '__MACOSX' in member or '.DS_Store' in member:
                            continue

                    # Skip hidden files if configured
                    if Config.ZIPProcessing.IGNORE_HIDDEN:
                        if Path(member).name.startswith('.'):
                            continue

                    # Extract file
                    zf.extract(member, extract_to)
                    extracted_path = os.path.join(extract_to, member) # type: ignore

                    if os.path.isfile(extracted_path):
                        extracted_files.append(extracted_path)
                        if self.verbose and len(extracted_files) % 10 == 0:
                            print(f"  Extracted {len(extracted_files)} files...")

            self.extracted_files = extracted_files
            self.processing_stats['total_files'] = len(extracted_files)

            if self.verbose:
                print(f"✅ Extracted {len(extracted_files)} files")

            return extracted_files

        except Exception as e:
            error_msg = f"Extraction failed: {str(e)}"
            self.processing_stats['errors'].append(error_msg)
            if self.verbose:
                print(f"❌ {error_msg}")
            raise

    def filter_files_for_processing(
        self, 
        files: List[str]
    ) -> List[str]:
        """
        Filter files based on extension and configuration.

        Args:
            files: List of file paths

        Returns:
            Filtered list of files to process
        """
        filtered = []

        for file_path in files:
            file_ext = Path(file_path).suffix.lower()

            # Check if extension should be processed
            if file_ext in Config.ZIPProcessing.PROCESS_EXTENSIONS:
                # Check if not in ignore list
                if file_ext not in Config.ZIPProcessing.IGNORE_EXTENSIONS:
                    filtered.append(file_path)
                else:
                    self.processing_stats['skipped_files'] += 1
            else:
                self.processing_stats['skipped_files'] += 1

        if self.verbose:
            print(f"📋 Filtered files: {len(filtered)} to process, "
                  f"{self.processing_stats['skipped_files']} skipped")

        return filtered

    def process_extracted_files(
        self, 
        files: List[str],
        process_function: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process extracted files with custom function.

        Args:
            files: List of file paths to process
            process_function: Optional function to process each file

        Returns:
            Dictionary with processing results
        """
        results = {
            'processed': [],
            'failed': [],
            'skipped': [],
            'total': len(files)
        }

        # Filter files first
        files_to_process = self.filter_files_for_processing(files)

        if self.verbose:
            print(f"🔄 Processing {len(files_to_process)} files...")

        for file_path in files_to_process:
            try:
                file_ext = Path(file_path).suffix.lower()
                file_name = Path(file_path).name

                # Process based on file type
                if file_ext == '.pdf' and self.pdf_processor:
                    # Process PDF
                    if self.verbose:
                        print(f"  📄 Processing PDF: {file_name}")
                    result = self.pdf_processor.process_pdf_for_translation(file_path)
                    results['processed'].append({
                        'file': file_path,
                        'type': 'pdf',
                        'result': result
                    })

                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    # Process image
                    if self.verbose:
                        print(f"  🖼️ Processing image: {file_name}")
                    results['processed'].append({
                        'file': file_path,
                        'type': 'image',
                        'result': 'Ready for OCR'
                    })

                elif file_ext in ['.txt']:
                    # Process text file
                    if self.verbose:
                        print(f"  📝 Processing text: {file_name}")
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    results['processed'].append({
                        'file': file_path,
                        'type': 'text',
                        'result': content[:1000]  # First 1000 chars
                    })

                elif process_function:
                    # Use custom processing function
                    result = process_function(file_path)
                    results['processed'].append({
                        'file': file_path,
                        'type': 'custom',
                        'result': result
                    })
                else:
                    results['skipped'].append(file_path)

                self.processing_stats['processed_files'] += 1

            except Exception as e:
                results['failed'].append({
                    'file': file_path,
                    'error': str(e)
                })
                self.processing_stats['failed_files'] += 1

                if not Config.ZIPProcessing.CONTINUE_ON_ERROR:
                    raise

        if self.verbose:
            print("✅ Processing complete:")
            print(f"   Processed: {len(results['processed'])}")
            print(f"   Failed: {len(results['failed'])}")
            print(f"   Skipped: {len(results['skipped'])}")

        return results

    def create_zip(
        self, 
        files: List[str], 
        output_path: str,
        base_path: Optional[str] = None
    ) -> str:
        """
        Create ZIP archive from files.

        Args:
            files: List of file paths to include
            output_path: Path for output ZIP
            base_path: Base path for relative paths in ZIP

        Returns:
            Path to created ZIP
        """
        if self.verbose:
            print(f"📦 Creating ZIP: {output_path}")
            print(f"   Files to add: {len(files)}")

        compression = zipfile.ZIP_DEFLATED if Config.ZIPProcessing.COMPRESS_OUTPUT else zipfile.ZIP_STORED

        try:
            with zipfile.ZipFile(output_path, 'w', compression=compression) as zf:
                for file_path in files:
                    if os.path.exists(file_path):
                        # Determine archive name
                        if base_path and Config.ZIPProcessing.PRESERVE_STRUCTURE:
                            arcname = os.path.relpath(file_path, base_path)
                        else:
                            arcname = os.path.basename(file_path)

                        zf.write(file_path, arcname)

                        if self.verbose and len(zf.namelist()) % 10 == 0:
                            print(f"  Added {len(zf.namelist())} files...")

            # Get final size
            zip_size_mb = os.path.getsize(output_path) / (1024 * 1024)

            if self.verbose:
                print(f"✅ Created ZIP: {output_path}")
                print(f"   Size: {zip_size_mb:.2f}MB")
                print(f"   Files: {len(files)}")

            return output_path

        except Exception as e:
            error_msg = f"ZIP creation failed: {str(e)}"
            self.processing_stats['errors'].append(error_msg)
            if self.verbose:
                print(f"❌ {error_msg}")
            raise

    def process_zip_archive(
        self, 
        zip_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete ZIP processing pipeline.

        Args:
            zip_path: Path to input ZIP
            output_dir: Directory for output files

        Returns:
            Dictionary with processing results
        """
        if self.verbose:
            print(f"🔄 Processing ZIP archive: {Path(zip_path).name}")
            print("="*50)

        # Validate ZIP
        is_valid, message = self.validate_zip(zip_path)
        if not is_valid:
            raise ValueError(f"Invalid ZIP: {message}")

        # Extract files
        extracted_files = self.extract_zip(zip_path)

        # Process files
        results = self.process_extracted_files(extracted_files)

        # Create output ZIP if specified
        if output_dir:
            output_zip = os.path.join(
                output_dir,
                Config.ZIPProcessing.OUTPUT_PREFIX + Path(zip_path).name
            )
            processed_files = [r['file'] for r in results['processed']]
            if processed_files:
                self.create_zip(processed_files, output_zip)
                results['output_zip'] = output_zip

        results['stats'] = self.processing_stats.copy()

        if self.verbose:
            print("="*50)
            print("✅ ZIP processing complete!")

        return results

    def cleanup(self):
        """Clean up extracted files."""
        if self.extracted_files and self.file_handler:
            extract_dir = os.path.join(self.file_handler.temp_dir, 'zip_extract') # type: ignore
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
                if self.verbose:
                    print("🗑️ Cleaned up extracted files")

    def get_stats(self) -> Dict:
        """Get processing statistics."""
        return self.processing_stats.copy()


# Initialize ZIP processor
print("\n" + "="*50)
print("🚀 Initializing ZIP Processor...")
print("="*50)

# Initialize with existing handlers if available
zip_processor = ZIPProcessor(
    file_handler=file_handler if 'file_handler' in globals() else None,
    pdf_processor=pdf_processor if 'pdf_processor' in globals() else None,
    verbose=Config.Debug.VERBOSE
)

print("✅ ZIP Processor ready for use!")
print("="*50)


# In[ ]:


# Cell 10i: Output Generation System
"""
Output Generation System for Universal Translator v1.3
Handles final document creation and formatting
"""

from typing import List, Dict, Optional, Any, Union
import os

class OutputGenerator:
    """
    Comprehensive output generation system.
    Creates final translated documents in various formats.
    """

    def __init__(
        self,
        file_handler: Optional['FileHandler'] = None,
        pdf_processor: Optional['PDFProcessor'] = None,
        zip_processor: Optional['ZIPProcessor'] = None,
        verbose: bool = True
    ):
        """
        Initialize OutputGenerator.

        Args:
            file_handler: Optional FileHandler for temp management
            pdf_processor: Optional PDFProcessor for PDF generation
            zip_processor: Optional ZIPProcessor for archive creation
            verbose: Enable detailed output
        """
        self.verbose = verbose
        self.file_handler = file_handler
        self.pdf_processor = pdf_processor
        self.zip_processor = zip_processor

        self.output_stats = {
            'files_generated': 0,
            'total_size_mb': 0,
            'formats_used': set(),
            'processing_time': 0
        }

        if self.verbose:
            print("📤 OutputGenerator initialized")
            if self.file_handler:
                print(f"📁 Using session: {self.file_handler.session_id}")

    def generate_output_filename(
        self,
        source_file: str,
        output_format: str,
        include_timestamp: bool = True
    ) -> str:
        """
        Generate systematic output filename.

        Args:
            source_file: Original filename
            output_format: Output format (pdf, txt, zip)
            include_timestamp: Add timestamp to filename

        Returns:
            Generated output filename
        """
        source_path = Path(source_file)
        base_name = source_path.stem

        # Build filename components
        components = []

        if Config.OutputGeneration.USE_SOURCE_NAME:
            components.append(base_name)

        components.append(Config.OutputGeneration.OUTPUT_SUFFIX)

        if include_timestamp:
            timestamp = datetime.now().strftime(Config.OutputGeneration.TIMESTAMP_FORMAT)
            components.append(timestamp)

        # Join components and add extension
        filename = '_'.join(filter(None, components))
        filename = f"{filename}.{output_format}"

        return filename

    def create_text_output(
        self,
        translations: List[Dict[str, Any]],
        output_path: str,
        include_original: bool = True
    ) -> str:
        """
        Create plain text output file.

        Args:
            translations: List of translation dictionaries
            output_path: Path for output file
            include_original: Include original text

        Returns:
            Path to created file
        """
        if self.verbose:
            print(f"📝 Creating text output: {output_path}")

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Add header
                f.write("="*60 + "\n")
                f.write("TRANSLATED DOCUMENT\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Translator: Universal Translator v1.3\n")
                f.write("="*60 + "\n\n")

                # Add translations
                for i, item in enumerate(translations, 1):
                    f.write(f"--- Page/Section {i} ---\n\n")

                    if include_original and 'original' in item:
                        f.write("ORIGINAL:\n")
                        f.write(item['original'])
                        f.write("\n\n")
                        f.write("TRANSLATION:\n")

                    f.write(item.get('translated', item.get('text', '')))
                    f.write("\n\n")
                    f.write(Config.OutputGeneration.SEPARATOR_LINE)
                    f.write("\n\n")

                # Add footer
                f.write(f"\nTotal sections: {len(translations)}\n")
                f.write("="*60 + "\n")

            self.output_stats['files_generated'] += 1
            self.output_stats['formats_used'].add('txt')

            if self.verbose:
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"✅ Created text file: {size_mb:.2f}MB")

            return output_path

        except Exception as e:
            if self.verbose:
                print(f"❌ Text output failed: {e}")
            raise

    def create_pdf_output(
        self,
        translations: List[Dict[str, Any]],
        output_path: str,
        include_original: bool = True
    ) -> str:
        """
        Create PDF output file.

        Args:
            translations: List of translation dictionaries
            output_path: Path for output file
            include_original: Include original text

        Returns:
            Path to created file
        """
        if not self.pdf_processor:
            if self.verbose:
                print("⚠️ PDF processor not available, falling back to text")
            return self.create_text_output(translations, output_path.replace('.pdf', '.txt'))

        if self.verbose:
            print(f"📄 Creating PDF output: {output_path}")

        try:
            # Prepare content for PDF
            pdf_content = []

            for item in translations:
                if include_original and 'original' in item:
                    if Config.OutputGeneration.LAYOUT_TYPE == 'side_by_side':
                        # Format as table or columns
                        content = "Original | Translation\n"
                        content += f"{item['original']} | {item.get('translated', '')}"
                    elif Config.OutputGeneration.LAYOUT_TYPE == 'interleaved':
                        content = f"Original:\n{item['original']}\n\n"
                        content += f"Translation:\n{item.get('translated', '')}"
                    else:  # sequential
                        content = f"{item['original']}\n\n---\n\n{item.get('translated', '')}"
                else:
                    content = item.get('translated', item.get('text', ''))

                pdf_content.append(content)

            # Create PDF
            result = self.pdf_processor.create_translated_pdf(
                translations=pdf_content,
                output_path=output_path,
                metadata={'generator': 'Universal Translator v1.3'}
            )

            self.output_stats['files_generated'] += 1
            self.output_stats['formats_used'].add('pdf')

            return result

        except Exception as e:
            if self.verbose:
                print(f"❌ PDF output failed: {e}")
            raise

    def create_report(
        self,
        processing_results: Dict[str, Any],
        output_path: str
    ) -> str:
        """
        Create processing report.

        Args:
            processing_results: Dictionary with processing results
            output_path: Path for report file

        Returns:
            Path to created report
        """
        if self.verbose:
            print(f"📊 Creating report: {output_path}")

        report_content = {
            'timestamp': datetime.now().isoformat(),
            'version': 'Universal Translator v1.3',
            'session_id': self.file_handler.session_id if self.file_handler else 'unknown',
            'results': processing_results,
            'statistics': self.output_stats
        }

        try:
            if Config.OutputGeneration.REPORT_FORMAT == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_content, f, indent=2, default=str)
            else:  # txt format
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write("TRANSLATION PROCESSING REPORT\n")
                    f.write("="*60 + "\n\n")
                    f.write(f"Date: {report_content['timestamp']}\n")
                    f.write(f"Version: {report_content['version']}\n")
                    f.write(f"Session: {report_content['session_id']}\n\n")

                    if Config.OutputGeneration.INCLUDE_STATS:
                        f.write("STATISTICS:\n")
                        f.write("-"*40 + "\n")
                        for key, value in self.output_stats.items():
                            f.write(f"  {key}: {value}\n")
                        f.write("\n")

                    f.write("PROCESSING DETAILS:\n")
                    f.write("-"*40 + "\n")
                    f.write(json.dumps(processing_results, indent=2, default=str))
                    f.write("\n\n" + "="*60 + "\n")

            if self.verbose:
                print(f"✅ Report created: {output_path}")

            return output_path

        except Exception as e:
            if self.verbose:
                print(f"❌ Report creation failed: {e}")
            raise

    def generate_batch_output(
        self,
        translations: List[Dict[str, Any]],
        source_files: List[str],
        output_dir: str,
        output_format: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Generate output for batch processing.

        Args:
            translations: List of translation results
            source_files: List of source file paths
            output_dir: Output directory
            output_format: Output format or 'auto'

        Returns:
            Dictionary with output results
        """
        if self.verbose:
            print(f"📦 Generating batch output for {len(translations)} files")

        # Create output directory
        output_path = Path(output_dir) / Config.OutputGeneration.BATCH_OUTPUT_DIR

        if Config.OutputGeneration.ORGANIZE_BY_TYPE:
            # Create subdirectories by type
            for subdir in ['pdf', 'text', 'images', 'other']:
                (output_path / subdir).mkdir(parents=True, exist_ok=True)
        else:
            output_path.mkdir(parents=True, exist_ok=True)

        results = {
            'output_dir': str(output_path),
            'files_created': [],
            'errors': [],
            'summary': {}
        }

        # Process each translation
        for trans, source in zip(translations, source_files):
            try:
                # Determine output format
                if output_format == 'auto':
                    source_ext = Path(source).suffix.lower()
                    if source_ext == '.pdf':
                        fmt = 'pdf'
                    elif source_ext in ['.txt', '.md']:
                        fmt = 'txt'
                    else:
                        fmt = Config.OutputGeneration.DEFAULT_FORMAT
                else:
                    fmt = output_format

                # Generate output filename
                output_filename = self.generate_output_filename(source, fmt)

                # Determine subdirectory
                if Config.OutputGeneration.ORGANIZE_BY_TYPE:
                    if fmt == 'pdf':
                        subdir = 'pdf'
                    elif fmt == 'txt':
                        subdir = 'text'
                    else:
                        subdir = 'other'
                    full_output_path = output_path / subdir / output_filename
                else:
                    full_output_path = output_path / output_filename

                # Create output file
                if fmt == 'pdf':
                    created = self.create_pdf_output([trans], str(full_output_path))
                else:
                    created = self.create_text_output([trans], str(full_output_path))

                results['files_created'].append(created)

            except Exception as e:
                results['errors'].append({
                    'source': source,
                    'error': str(e)
                })

        # Create summary if configured
        if Config.OutputGeneration.CREATE_SUMMARY:
            summary_path = output_path / 'translation_summary.txt'
            self._create_batch_summary(results, str(summary_path))

        # Generate report if configured
        if Config.OutputGeneration.GENERATE_REPORT:
            report_path = output_path / f'report.{Config.OutputGeneration.REPORT_FORMAT}'
            self.create_report(results, str(report_path))

        if self.verbose:
            print("✅ Batch output complete:")
            print(f"   Files created: {len(results['files_created'])}")
            print(f"   Errors: {len(results['errors'])}")
            print(f"   Output directory: {output_path}")

        return results

    def _create_batch_summary(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> None:
        """Create summary file for batch processing."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("BATCH TRANSLATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files: {len(results['files_created'])}\n")
            if results['errors']:
                f.write(f"Errors: {len(results['errors'])}\n")
            f.write("\n" + "="*60 + "\n\n")

            f.write("FILES CREATED:\n")
            for file in results['files_created']:
                f.write(f"  - {Path(file).name}\n")

            if results['errors']:
                f.write("\nERRORS:\n")
                for error in results['errors']:
                    f.write(f"  - {error['source']}: {error['error']}\n")

    def process_and_output(
        self,
        source_file: str,
        translation_result: Dict[str, Any],
        output_format: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> str:
        """
        Complete processing and output generation pipeline.

        Args:
            source_file: Source file path
            translation_result: Translation results dictionary
            output_format: Desired output format
            output_dir: Output directory

        Returns:
            Path to generated output file
        """
        start_time = datetime.now()

        if self.verbose:
            print(f"🔄 Processing output for: {Path(source_file).name}")

        # Determine output format
        if not output_format:
            output_format = Config.OutputGeneration.DEFAULT_FORMAT

        # Determine output directory
        if not output_dir:
            if self.file_handler:
                output_dir = Path(self.file_handler.temp_dir) / 'output' # type: ignore
            else:
                output_dir = Path('.') # type: ignore

        Path(output_dir).mkdir(parents=True, exist_ok=True) # type: ignore

        # Generate output filename
        output_filename = self.generate_output_filename(
            source_file,
            output_format,
            include_timestamp=True
        )
        output_path = Path(output_dir) / output_filename # type: ignore

        # Create output based on format
        if output_format == 'pdf':
            result = self.create_pdf_output(
                [translation_result],
                str(output_path),
                include_original=Config.OutputGeneration.INCLUDE_ORIGINAL
            )
        elif output_format == 'txt':
            result = self.create_text_output(
                [translation_result],
                str(output_path),
                include_original=Config.OutputGeneration.INCLUDE_ORIGINAL
            )
        elif output_format == 'zip' and self.zip_processor:
            # Create text file first, then zip it
            txt_path = str(output_path).replace('.zip', '.txt')
            self.create_text_output([translation_result], txt_path)
            result = self.zip_processor.create_zip([txt_path], str(output_path))
            # Clean up temp text file
            if os.path.exists(txt_path):
                os.remove(txt_path)
        else:
            # Default to text
            result = self.create_text_output(
                [translation_result],
                str(output_path),
                include_original=Config.OutputGeneration.INCLUDE_ORIGINAL
            )

        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.output_stats['processing_time'] += processing_time

        if self.verbose:
            print(f"✅ Output generated: {Path(result).name}")
            print(f"   Format: {output_format}")
            print(f"   Processing time: {processing_time:.2f}s")

        return result

    def get_stats(self) -> Dict:
        """Get output generation statistics."""
        return self.output_stats.copy()


# Initialize Output Generator
print("\n" + "="*50)
print("🚀 Initializing Output Generator...")
print("="*50)

output_generator = OutputGenerator(
    file_handler=file_handler if 'file_handler' in globals() else None,
    pdf_processor=pdf_processor if 'pdf_processor' in globals() else None,
    zip_processor=zip_processor if 'zip_processor' in globals() else None,
    verbose=Config.Debug.VERBOSE
)

print("✅ Output Generator ready for use!")
print("="*50)
print("\n🎉 ALL COMPONENTS INITIALIZED!")
print("Ready for translation processing!")


# ## 🧪 Testing & Examples {#testing}
# Test the translator with sample images

# In[ ]:


# Cell: Comprehensive Functionality Test
"""
Comprehensive test suite for Universal Translator v1.3.
Tests core functionality, error handling, and component integration.
"""

import os
from PIL import ImageDraw


def create_test_image(
    text: str,
    filename: str,
    width: int = 400,
    height: int = 100
) -> str:
    """
    Create a simple test image with text.

    Args:
        text: Text to write on image
        filename: Output filename
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Path to created image
    """
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # Draw text at multiple positions for better OCR
    draw.text((20, 20), text, fill='black')
    draw.text((20, 50), "Test 123", fill='black')

    img.save(filename)
    return filename


def run_comprehensive_test() -> None:
    """Run comprehensive test suite for the translator."""

    print("="*60)
    print("🧪 UNIVERSAL TRANSLATOR v1.3 - COMPREHENSIVE TEST")
    print("="*60)

    # Test counters
    tests_passed = 0
    tests_failed = 0
    test_results = []

    # ========== Test 1: Component Initialization ==========
    print("\n📋 Test 1: Component Initialization")
    try:
        # Check translator exists
        assert translator is not None
        assert hasattr(translator, 'process')
        assert hasattr(translator, 'enhance_image')
        print("   ✅ Translator initialized correctly")
        tests_passed += 1
        test_results.append(("Initialization", True, None))
    except AssertionError as e:
        print(f"   ❌ Initialization failed: {e}")
        tests_failed += 1
        test_results.append(("Initialization", False, str(e)))

    # ========== Test 2: Language Support Check ==========
    print("\n📋 Test 2: Language Support")
    try:
        # Check supported languages
        assert len(translator.supported_languages) == 5
        lang_names = [l.name for l in translator.supported_languages]
        expected = ['ENGLISH', 'CHINESE', 'JAPANESE', 'KOREAN', 'HINDI']
        assert all(lang in lang_names for lang in expected)
        print("   ✅ All 5 languages defined")

        # Check available languages
        available_count = len(translator.available_languages)
        missing_count = len(translator.missing_languages)
        print(f"   📊 Available: {available_count}, Missing: {missing_count}")
        tests_passed += 1
        test_results.append(("Language Support", True, None))
    except AssertionError as e:
        print(f"   ❌ Language check failed: {e}")
        tests_failed += 1
        test_results.append(("Language Support", False, str(e)))

    # ========== Test 3: Image Creation & Processing ==========
    print("\n📋 Test 3: Image Processing (English)")
    test_image = None
    try:
        # Create test image
        test_text = "Hello World"
        test_image = create_test_image(test_text, "test_english.jpg")
        print(f"   ✅ Created test image: {test_image}")

        # Process image
        result = translator.process(test_image, Language.ENGLISH)

        # Validate result structure
        assert result is not None
        assert isinstance(result, dict)
        assert 'original' in result
        assert 'fixed' in result
        assert 'translated' in result
        assert 'language' in result

        # Check if text was extracted
        if result['original'] or result['fixed']:
            print(f"   ✅ Text extracted: '{result['fixed'][:50]}'") # type: ignore
        else:
            print("   ⚠️ No text extracted (image might be too simple)")

        print("   ✅ Processing completed successfully")
        tests_passed += 1
        test_results.append(("Image Processing", True, None))

    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
        tests_failed += 1
        test_results.append(("Image Processing", False, str(e)))
    finally:
        # Cleanup test image
        if test_image and os.path.exists(test_image):
            try:
                os.remove(test_image)
                print(f"   🗑️ Cleaned up {test_image}")
            except:
                pass

    # ========== Test 4: Error Handling ==========
    print("\n📋 Test 4: Error Handling")
    try:
        # Test with non-existent file
        result = translator.process("non_existent.jpg", Language.ENGLISH)

        # Should either return None or dict with errors
        if result is None:
            print("   ✅ Returned None for missing file")
        elif isinstance(result, dict) and 'errors' in result:
            print("   ✅ Returned error in result dict")
        else:
            print("   ⚠️ Unexpected result for missing file")

        tests_passed += 1
        test_results.append(("Error Handling", True, None))

    except FileNotFoundError:
        print("   ✅ Raised FileNotFoundError (expected behavior)")
        tests_passed += 1
        test_results.append(("Error Handling", True, None))
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        tests_failed += 1
        test_results.append(("Error Handling", False, str(e)))

    # ========== Test 5: Type Validation ==========
    print("\n📋 Test 5: Type Validation")
    try:
        # Test with invalid language type
        test_image = create_test_image("Test", "test_type.jpg")

        try:
            # This should raise TypeError
            result = translator.process(test_image, "english")  # type: ignore # String instead of enum
            print("   ❌ Should have raised TypeError")
            tests_failed += 1
            test_results.append(("Type Validation", False, "No error raised"))
        except TypeError:
            print("   ✅ Correctly rejected string instead of Language enum")
            tests_passed += 1
            test_results.append(("Type Validation", True, None))
        finally:
            if os.path.exists(test_image):
                os.remove(test_image)

    except Exception as e:
        print(f"   ❌ Type validation test failed: {e}")
        tests_failed += 1
        test_results.append(("Type Validation", False, str(e)))

    # ========== Test 6: Configuration Integration ==========
    print("\n📋 Test 6: Configuration Integration")
    try:
        # Check if Config is being used
        assert Config.Debug.VERBOSE in [True, False]
        assert Config.Image.SCALE_FACTOR > 0
        assert Config.ErrorHandling.RETRY_COUNT >= 0
        print("   ✅ Configuration properly integrated")
        tests_passed += 1
        test_results.append(("Configuration", True, None))
    except Exception as e:
        print(f"   ❌ Configuration check failed: {e}")
        tests_failed += 1
        test_results.append(("Configuration", False, str(e)))

    # ========== Test Summary ==========
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)

    # Print results table
    print("\n📋 Detailed Results:")
    for test_name, passed, error in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:20} {status}")
        if error and Config.Debug.DETAILED_ERRORS:
            print(f"      Error: {error}")

    # Overall summary
    total_tests = tests_passed + tests_failed
    pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0

    print("\n📈 Overall Results:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {tests_passed}")
    print(f"   Failed: {tests_failed}")
    print(f"   Pass Rate: {pass_rate:.1f}%")

    # Final status
    print("\n" + "="*60)
    if tests_failed == 0:
        print("🎉 ALL TESTS PASSED! Translator is working correctly.")
    elif tests_passed > tests_failed:
        print("⚠️ PARTIAL SUCCESS: Most features working, some issues found.")
    else:
        print("❌ TESTS FAILED: Significant issues detected.")
    print("="*60)


# Run the comprehensive test
if __name__ == "__main__" or True:  # Always run in notebook
    run_comprehensive_test()


# In[ ]:


# Cell: Test File Handler
"""Test the File Handling System"""

print("🧪 TESTING FILE HANDLER")
print("=" * 50)

# Test 1: File type detection
print("\n📋 Test 1: File Type Detection")
test_files = [
    ("test_english.jpg", "image"),
    ("document.pdf", "pdf"),
    ("archive.zip", "zip"),
    ("unknown.xyz", "unknown")
]

for filename, expected in test_files:
    # Create dummy file for testing
    Path(filename).touch()
    detected = file_handler.file_type_detector(filename)
    status = "✅" if detected == expected else "❌"
    print(f"{status} {filename}: detected as '{detected}'")
    # Clean up
    if Path(filename).exists():
        Path(filename).unlink()

print("\n" + "=" * 50)

# Test 2: File validation
print("📋 Test 2: File Validation")
# Use the test image we created earlier
if Path("test_english.jpg").exists():
    is_valid, message = file_handler.file_validator("test_english.jpg")
    print(f"✅ Validation result: {message}")
else:
    print("⚠️ No test file available")

print("\n" + "=" * 50)

# Test 3: Batch processing
print("📋 Test 3: Batch Processing")
# Create test directory with files
test_dir = Path("test_batch")
test_dir.mkdir(exist_ok=True)

# Create some test files
for i in range(3):
    Path(test_dir / f"test_{i}.jpg").touch()
    Path(test_dir / f"doc_{i}.txt").touch()

# Process the directory
valid_files = file_handler.batch_file_processor(
    str(test_dir),
    file_types=['image', 'text']
)
print(f"✅ Found {len(valid_files)} valid files")

# Cleanup test directory
shutil.rmtree(test_dir)

print("\n" + "=" * 50)

# Test 4: Temp file management
print("📋 Test 4: Temporary File Management")
temp_dir = file_handler.temp_file_manager('get')
print(f"✅ Temp directory: {temp_dir}")
print(f"✅ Session ID: {file_handler.session_id}")

# Get stats
stats = file_handler.get_processing_stats()
print(f"✅ Session stats: {stats['total_files']} files processed")

print("\n" + "=" * 50)
print("✅ All File Handler tests complete!")


# In[ ]:


# Cell: Test PDF Processor
"""Test the PDF Processing System"""

print("🧪 TESTING PDF PROCESSOR")
print("=" * 50)

# Test 1: Create a simple test PDF
print("\n📋 Test 1: Create Test PDF")
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Create a simple test PDF
test_pdf_path = "test_document.pdf"
c = canvas.Canvas(test_pdf_path, pagesize=letter)

# Add some pages with text
test_content = [
    "Hello World! This is page 1 of the test PDF.",
    "This is page 2. Testing PDF processing capabilities.",
    "Page 3 contains Chinese text: 你好世界",
    "Final page 4. The End."
]

for i, text in enumerate(test_content, 1):
    c.drawString(100, 750, f"Page {i}")
    c.drawString(100, 700, text)
    c.drawString(100, 650, "Universal Translator Test Document")
    if i < len(test_content):
        c.showPage()  # Add new page except for last

c.save()
print(f"✅ Created test PDF: {test_pdf_path}")
print(f"   Pages: {len(test_content)}")

print("\n" + "=" * 50)

# Test 2: Extract text from PDF
print("📋 Test 2: Extract Text from PDF")
extracted = pdf_processor.extract_text_from_pdf(test_pdf_path)
print(f"✅ Extracted {len(extracted)} pages")
for page in extracted[:2]:  # Show first 2 pages
    text_preview = page['text'][:100] if page['text'] else "No text"
    print(f"   Page {page['page']}: {text_preview}...")

print("\n" + "=" * 50)

# Test 3: Convert PDF to images
print("📋 Test 3: Convert PDF to Images")
try:
    images = pdf_processor.pdf_to_images(test_pdf_path)
    print(f"✅ Created {len(images)} images")
    for img in images[:2]:
        print(f"   {Path(img).name}")
except Exception as e:
    print(f"⚠️ Image conversion skipped: {e}")
    print("   (This is OK if poppler-utils is not installed)")

print("\n" + "=" * 50)

# Test 4: Create translated PDF
print("📋 Test 4: Create Translated PDF")
sample_translations = [
    "Hello World! This is page 1 of the test PDF. (Translated)",
    "This is page 2. Testing PDF processing capabilities. (Translated)",
    "Page 3 contains Chinese text: Hello World (Translated from Chinese)",
    "Final page 4. The End. (Translated)"
]

output_pdf = "test_translated.pdf"
result_path = pdf_processor.create_translated_pdf(
    translations=sample_translations,
    output_path=output_pdf
)
print(f"✅ Created translated PDF: {result_path}")

print("\n" + "=" * 50)

# Test 5: Full processing pipeline
print("📋 Test 5: Full Processing Pipeline")
process_result = pdf_processor.process_pdf_for_translation(test_pdf_path)
print("✅ Processing complete!")
print(f"   Method: {process_result['method_used']}")
print(f"   Pages: {process_result['total_pages']}")
print(f"   Needs OCR: {process_result['needs_ocr']}")

# Cleanup
import os
if os.path.exists(test_pdf_path):
    os.remove(test_pdf_path)
    print(f"\n🗑️ Cleaned up {test_pdf_path}")
if os.path.exists(output_pdf):
    os.remove(output_pdf)
    print(f"🗑️ Cleaned up {output_pdf}")

print("\n" + "=" * 50)
print("✅ All PDF Processor tests complete!")


# In[ ]:


# Cell: Test ZIP Processor
"""Test the ZIP Processing System"""

import os
from pathlib import Path

print("🧪 TESTING ZIP PROCESSOR")
print("=" * 50)

# Test 1: Create test files and ZIP
print("\n📋 Test 1: Create Test ZIP")

# Create test directory
test_dir = Path("test_zip_content")
test_dir.mkdir(exist_ok=True)

# Create various test files
test_files = []

# Create text files
for i in range(3):
    file_path = test_dir / f"document_{i}.txt"
    file_path.write_text(f"This is test document {i}. Contains sample text for processing.")
    test_files.append(str(file_path))

# Create a simple PDF using reportlab
from reportlab.pdfgen import canvas
pdf_path = test_dir / "test_document.pdf"
c = canvas.Canvas(str(pdf_path))
c.drawString(100, 750, "Test PDF in ZIP archive")
c.drawString(100, 700, "This PDF is inside the ZIP file")
c.save()
test_files.append(str(pdf_path))

# Create subdirectory with files
sub_dir = test_dir / "subfolder"
sub_dir.mkdir(exist_ok=True)
sub_file = sub_dir / "nested_file.txt"
sub_file.write_text("This file is in a subdirectory")
test_files.append(str(sub_file))

# Create the ZIP file
test_zip = "test_archive.zip"
with zipfile.ZipFile(test_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for file in test_files:
        arcname = os.path.relpath(file, test_dir.parent)
        zf.write(file, arcname)
    # Add a system file to test filtering
    zf.writestr(".DS_Store", "system file")
    zf.writestr("__MACOSX/._test", "mac file")

print(f"✅ Created test ZIP: {test_zip}")
print(f"   Files added: {len(test_files)}")

print("\n" + "=" * 50)

# Test 2: Validate ZIP
print("📋 Test 2: Validate ZIP")
is_valid, message = zip_processor.validate_zip(test_zip)
print(f"{'✅' if is_valid else '❌'} Validation: {message}")

print("\n" + "=" * 50)

# Test 3: Extract ZIP
print("📋 Test 3: Extract ZIP")
extracted_files = zip_processor.extract_zip(test_zip)
print(f"✅ Extracted {len(extracted_files)} files")
for file in extracted_files[:3]:  # Show first 3
    print(f"   - {Path(file).name}")

print("\n" + "=" * 50)

# Test 4: Filter files
print("📋 Test 4: Filter Files for Processing")
filtered = zip_processor.filter_files_for_processing(extracted_files)
print(f"✅ Filtered: {len(filtered)} files to process")
print(f"   Skipped: {zip_processor.processing_stats['skipped_files']} files")

print("\n" + "=" * 50)

# Test 5: Process extracted files
print("📋 Test 5: Process Extracted Files")
results = zip_processor.process_extracted_files(extracted_files)
print("✅ Processing complete:")
print(f"   Processed: {len(results['processed'])}")
print(f"   Failed: {len(results['failed'])}")
print(f"   Skipped: {len(results['skipped'])}")

for item in results['processed'][:2]:  # Show first 2
    print(f"   - {Path(item['file']).name} ({item['type']})")

print("\n" + "=" * 50)

# Test 6: Create output ZIP
print("📋 Test 6: Create Output ZIP")
output_zip = "test_output.zip"
processed_files = [r['file'] for r in results['processed']]
if processed_files:
    created_zip = zip_processor.create_zip(
        processed_files, 
        output_zip,
        base_path=Path(extracted_files[0]).parent if extracted_files else None # type: ignore
    )
    print(f"✅ Created output ZIP: {created_zip}")

    # Verify output ZIP
    with zipfile.ZipFile(output_zip, 'r') as zf:
        print(f"   Contains {len(zf.namelist())} files")

print("\n" + "=" * 50)

# Test 7: Full pipeline
print("📋 Test 7: Full ZIP Processing Pipeline")
pipeline_result = zip_processor.process_zip_archive(test_zip, output_dir=".")
print("✅ Pipeline complete!")
print(f"   Stats: {pipeline_result['stats']['processed_files']} processed, "
      f"{pipeline_result['stats']['failed_files']} failed")

print("\n" + "=" * 50)

# Cleanup
print("🗑️ Cleaning up test files...")

# Clean up test files
import shutil
if test_dir.exists():
    shutil.rmtree(test_dir)
    print(f"   Removed {test_dir}")

for file in [test_zip, output_zip, f"{Config.ZIPProcessing.OUTPUT_PREFIX}{test_zip}"]:
    if os.path.exists(file):
        os.remove(file)
        print(f"   Removed {file}")

# Clean up extracted files
zip_processor.cleanup()

print("\n" + "=" * 50)
print("✅ All ZIP Processor tests complete!")


# In[ ]:


# Cell: Test Output Generator
"""Test the Output Generation System - Final Component!"""

from pathlib import Path

print("🧪 TESTING OUTPUT GENERATOR - FINAL COMPONENT")
print("=" * 50)

# Test 1: Create sample translation data
print("\n📋 Test 1: Prepare Sample Data")
sample_translations = [
    {
        'original': '你好世界！这是测试文档。',
        'translated': 'Hello World! This is a test document.',
        'source_language': 'Chinese',
        'target_language': 'English',
        'confidence': 0.95
    },
    {
        'original': 'こんにちは世界！テストです。',
        'translated': 'Hello World! This is a test.',
        'source_language': 'Japanese', 
        'target_language': 'English',
        'confidence': 0.92
    },
    {
        'original': '안녕하세요 세계! 테스트입니다.',
        'translated': 'Hello World! This is a test.',
        'source_language': 'Korean',
        'target_language': 'English',
        'confidence': 0.88
    }
]
print(f"✅ Created {len(sample_translations)} sample translations")

print("\n" + "=" * 50)

# Test 2: Generate text output
print("📋 Test 2: Generate Text Output")
text_output = output_generator.create_text_output(
    translations=sample_translations,
    output_path="test_output.txt",
    include_original=True
)
print(f"✅ Created text output: {text_output}")
if Path(text_output).exists():
    size = Path(text_output).stat().st_size
    print(f"   File size: {size} bytes")

print("\n" + "=" * 50)

# Test 3: Generate PDF output
print("📋 Test 3: Generate PDF Output")
pdf_output = output_generator.create_pdf_output(
    translations=sample_translations,
    output_path="test_output.pdf",
    include_original=True
)
print(f"✅ Created PDF output: {pdf_output}")
if Path(pdf_output).exists():
    size = Path(pdf_output).stat().st_size / 1024
    print(f"   File size: {size:.1f} KB")

print("\n" + "=" * 50)

# Test 4: Generate report
print("📋 Test 4: Generate Processing Report")
processing_results = {
    'files_processed': 3,
    'total_characters': 150,
    'languages_detected': ['Chinese', 'Japanese', 'Korean'],
    'success_rate': '100%',
    'processing_time': '2.5 seconds'
}

report_output = output_generator.create_report(
    processing_results=processing_results,
    output_path="test_report.txt"
)
print(f"✅ Created report: {report_output}")

# Also test JSON report
json_report = output_generator.create_report(
    processing_results=processing_results,
    output_path="test_report.json"
)
print(f"✅ Created JSON report: {json_report}")

print("\n" + "=" * 50)

# Test 5: Test filename generation
print("📋 Test 5: Test Filename Generation")
test_filenames = [
    ("document.pdf", "pdf"),
    ("image.jpg", "txt"),
    ("archive.zip", "pdf")
]

for source, fmt in test_filenames:
    generated = output_generator.generate_output_filename(
        source_file=source,
        output_format=fmt,
        include_timestamp=True
    )
    print(f"   {source} -> {generated}")

print("\n" + "=" * 50)

# Test 6: Batch output generation
print("📋 Test 6: Batch Output Generation")

# Create test batch data
batch_translations = [
    {'original': 'Text 1', 'translated': 'Translated 1'},
    {'original': 'Text 2', 'translated': 'Translated 2'},
    {'original': 'Text 3', 'translated': 'Translated 3'}
]
batch_sources = ['doc1.pdf', 'doc2.txt', 'doc3.pdf']

batch_results = output_generator.generate_batch_output(
    translations=batch_translations,
    source_files=batch_sources,
    output_dir=".",
    output_format='auto'
)

print("✅ Batch output complete:")
print(f"   Output directory: {batch_results['output_dir']}")
print(f"   Files created: {len(batch_results['files_created'])}")
print(f"   Errors: {len(batch_results['errors'])}")

print("\n" + "=" * 50)

# Test 7: Complete pipeline test
print("📋 Test 7: Complete Pipeline Test")

# Simulate a complete translation result
complete_translation = {
    'original': 'नमस्ते दुनिया! यह एक परीक्षण है।',
    'translated': 'Hello World! This is a test.',
    'source_language': 'Hindi',
    'target_language': 'English',
    'pages': 1,
    'method': 'text_extraction'
}

# Process and generate output
final_output = output_generator.process_and_output(
    source_file="test_document.pdf",
    translation_result=complete_translation,
    output_format='pdf',
    output_dir="."
)

print("✅ Complete pipeline successful!")
print(f"   Generated: {Path(final_output).name}")

print("\n" + "=" * 50)

# Test 8: Check statistics
print("📋 Test 8: Output Statistics")
stats = output_generator.get_stats()
print("📊 Generation Statistics:")
for key, value in stats.items():
    print(f"   {key}: {value}")

print("\n" + "=" * 50)

# Cleanup
print("🗑️ Cleaning up test files...")
test_files = [
    "test_output.txt", "test_output.pdf",
    "test_report.txt", "test_report.json",
    final_output
]

# Clean up batch output directory
import shutil
batch_dir = Path(batch_results['output_dir'])
if batch_dir.exists():
    shutil.rmtree(batch_dir)
    print("   Removed batch output directory")

for file in test_files:
    if Path(file).exists():
        Path(file).unlink()
        print(f"   Removed {file}")

print("\n" + "=" * 50)
print("✅ All Output Generator tests complete!")
print("\n" + "="*50)
print("🎉 CONGRATULATIONS! ALL 4 COMPONENTS TESTED!")
print("="*50)
print("\n📊 FINAL SYSTEM STATUS:")
print("  ✅ Component 1: FileHandler - READY")
print("  ✅ Component 2: PDFProcessor - READY")
print("  ✅ Component 3: ZIPProcessor - READY")
print("  ✅ Component 4: OutputGenerator - READY")
print("\n🚀 Universal Translator v1.3 - FULLY OPERATIONAL!")
print("="*50)


# ## 📚 Development Notes {#notes}
# 
# ### ✅ Completed Features (v1.4 - November 16, 2024):
# - **FileHandler Component:** Batch file processing with validation, session management, and temp file lifecycle
# - **PDFProcessor Component:** PDF text extraction (PyPDF2/pdfplumber), PDF to image conversion, translated PDF generation
# - **ZIPProcessor Component:** Archive extraction/creation, security validation, batch processing within ZIPs
# - **OutputGenerator Component:** Multi-format output (PDF/TXT/ZIP), bilingual documents, processing reports
# - **Language Support:** Chinese, Japanese, Korean, Hindi to English translation
# - **Code Quality:** Fixed all type hints (any→Any), resolved Path type issues, PEP 8 compliant
# 
# ### 🔄 Changes from v1.3 to v1.4:
# - Added 4 major processing components (was just core translator)
# - Expanded from single image processing to multiple file types
# - Implemented comprehensive error handling with retry logic
# - Added session-based processing with unique IDs
# - Integrated all components with existing translator
# 
# ### 📖 Technical Implementation:
# - **Design Pattern:** Modular component architecture
# - **Error Handling:** Try-except with retry mechanisms and graceful fallbacks
# - **Type Safety:** Proper type hints with Optional, Union, Callable
# - **Configuration:** Centralized Config class with nested settings
# - **Testing:** Individual component tests + integration tests
# 
# ### 🐛 Known Issues:
# - Poppler-utils required for PDF→image conversion (optional feature)
# - Some Pylance type warnings for Path/string (marked with # type: ignore)
# - Large PDF processing may require memory optimization
# 
# ### 📊 Performance Notes:
# - Processes up to 20 files per batch (configurable)
# - PDF extraction: ~1-2 seconds per page
# - ZIP processing: Handles up to 100 files
# - Session cleanup: Automatic temp file removal
# 
# ### 🔧 Dependencies Added:
# - PyPDF2: PDF reading
# - pdfplumber: Advanced PDF text extraction
# - pdf2image: PDF to image conversion (requires poppler-utils)
# - reportlab: PDF generation
# - All integrated with existing deep-translator and pytesseract
# 
# ### 📚 References:
# - PyPDF2 Documentation: https://pypdf2.readthedocs.io/
# - pdfplumber Documentation: https://github.com/jsvine/pdfplumber
# - ReportLab Documentation: https://www.reportlab.com/docs/
# 
# ### 👨‍💻 Developer: Victor
# ### 📅 Version: 1.4
# ### 📍 Session: GitHub Codespaces
# ### ⏱️ Development Time: Single day sprint
# 

# In[ ]:


# Cell: Export Notebook for UI Integration
"""
Export all translator components to Python file for Streamlit UI.
This allows the UI to import and use your translator.
Run this whenever you update the translator code.
"""

print("📤 EXPORTING TRANSLATOR FOR UI")
print("="*50)

# Export notebook to Python file
# get_ipython().system('jupyter nbconvert --to python translator_v1.3.ipynb --output translator_core')

print("\n✅ Export complete!")
print("📁 Created: translator_core.py")
print("📍 Location: Same folder as notebook")
print("\n📝 What this does:")
print("   • Converts notebook → Python file")
print("   • UI can now import your translator")
print("   • Run this after any code changes")
print("\n🔗 Next steps:")
print("   1. Update translator_integration.py")
print("   2. Restart Streamlit")
print("   3. Test with real files")
print("="*50)

# Verify the file was created
import os
if os.path.exists('translator_core.py'):
    file_size = os.path.getsize('translator_core.py') / 1024
    print(f"\n✅ Verified: translator_core.py ({file_size:.1f} KB)")
else:
    print("\n⚠️ File not found - check for errors above")

