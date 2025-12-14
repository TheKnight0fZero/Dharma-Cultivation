# app.py - Universal Translator v1.5 UI with Visual Translation
"""
Universal Translator Web Interface.
Now with VISUAL TRANSLATION - replaces text in images!

Version: 1.5
Developer: Victor
"""

import os
import tempfile
# import zipfile # REMOVED: Redundant top-level import
import json
import base64
from datetime import datetime
from pathlib import Path
import shutil

import streamlit as st
from PIL import Image

# Additional imports for downloads
# REMOVED: reportlab availability check, as app.py doesn't directly use it for UI logic.
# The backend translator_integration.py handles its own reportlab dependencies.

# Import translator service
try:
    from translator_integration import translator_service
    TRANSLATOR_READY = True
except ImportError as e:
    TRANSLATOR_READY = False
    print(f"⚠️ Translator not available: {e}")

# Constants
# MAX_LINE_LENGTH = 79 # REMOVED: Unused constant
SUPPORTED_TYPES = ['jpg', 'jpeg', 'png', 'pdf', 'txt', 'zip']
LANGUAGES = ["Auto-detect", "Chinese", "Japanese", "Korean", "Hindi"]
OUTPUT_FORMATS = ["Visual (Images/PDF)", "Text Only", "Both"] # This constant is now unused, but kept for context if output_format is re-added

# Page configuration
st.set_page_config(
    page_title="Universal Translator v1.5 - Visual",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
css_style = """
<style>
.main {padding: 0rem 1rem;}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
    width: 100%;
}
.stButton>button:hover {
    background-color: #45a049;
}
.image-container {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 10px;
    margin: 10px 0;
}
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
# if 'translation_history' not in st.session_state: # REMOVED: Unused session state variable
#     st.session_state.translation_history = []
if 'translated_files_paths' not in st.session_state:
    st.session_state.translated_files_paths = []


def get_file_icon(filename):
    """Get appropriate icon for file type."""
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        return "🖼️"
    elif filename.endswith('.pdf'):
        return "📄"
    elif filename.endswith('.txt'):
        return "📝"
    elif filename.endswith('.zip'):
        return "📦"
    return "📎"


def format_file_size(size_bytes):
    """Format file size for display."""
    size_kb = size_bytes / 1024
    if size_kb > 1024:
        return f"{size_kb/1024:.2f} MB"
    return f"{size_kb:.2f} KB"


# --- CHANGE 3: Modified function signature to accept idx ---
def display_image_comparison(original_path, translated_path, filename, idx):
    """Display original and translated images side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Original")
        if original_path and os.path.exists(original_path):
            original_img = Image.open(original_path)
            st.image(original_img, caption=f"Original: {filename}", use_column_width=True)
        else:
            st.info("Original image not available")
    
    with col2:
        st.subheader("📤 Translated")
        if translated_path and os.path.exists(translated_path):
            translated_img = Image.open(translated_path)
            st.image(translated_img, caption=f"Translated: {filename}", use_column_width=True)
            
            # Download button for translated image
            with open(translated_path, 'rb') as f:
                img_data = f.read()
            st.download_button(
            label=f"💾 Download Translated Image",
            data=img_data,
            file_name=f"translated_{filename}",
            mime="image/jpeg",
             key=f"download_{filename}_{idx}"  # --- CHANGE 3: Used idx in key ---
        )
        else:
            st.info("Translation in progress...")


def create_download_zip(file_paths):
    """Create a ZIP file with all translated files."""
    import zipfile # Local import is kept here, making top-level import redundant
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    
    with zipfile.ZipFile(temp_zip.name, 'w') as zf:
        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                arcname = Path(file_path).name
                zf.write(file_path, arcname)
    
    return temp_zip.name


# Header with visual translation emphasis
st.title("🌍 Universal Translator v1.5 - Visual Translation")
header_text = (
    "**Translate text IN images and PDFs from Chinese, Japanese, "
    "Korean, and Hindi to English**\n\n"
    "🎨 **NEW: Visual translation replaces text directly in images!**"
)
st.markdown(header_text)
st.markdown("---")

# Check translator status
if not TRANSLATOR_READY:
    st.error(
        "⚠️ Translator components not loaded. "
        "Please check translator_integration.py"
    )
else:
    # Check if visual translation is available
    if hasattr(translator_service, 'image_translator') and translator_service.image_translator:
        st.success("✅ Visual Translation System Active - Text will be replaced in images!")
    else:
        st.warning("⚠️ Visual translation not available - Text extraction mode only")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Language selection
    source_language = st.selectbox(
        "Source Language",
        LANGUAGES,
        help="Select source language or auto-detect"
    )
    
    # REMOVED: output_format selectbox as its functionality was not yet implemented in the backend
    # output_format = st.selectbox(
    #     "Output Format",
    #     OUTPUT_FORMATS,
    #     help="Visual: Replace text in images\nText: Extract text only"
    # )
    
    # Processing options
    st.subheader("📋 Processing Options")
    show_original = st.checkbox(
        "Show original images",
        value=True,
        help="Display original images alongside translations"
    )
    
    # REMOVED: save_intermediates checkbox as it was unused
    # save_intermediates = st.checkbox(
    #     "Save intermediate steps",
    #     value=False,
    #     help="Save cleaned images before text addition"
    # )
    
    # Info section
    st.markdown("---")
    st.subheader("📚 How It Works")
    info_text = (
        "**Visual Translation Process:**\n"
        "1. 🔍 Detect text locations\n"
        "2. 🌐 Translate to English\n"
        "3. 🎨 Remove original text\n"
        "4. ✍️ Add English text\n\n"
        "**Supported:**\n"
        "• Images → New images\n"
        "• PDFs → New PDFs\n"
        "• ZIPs → New ZIPs"
    )
    st.info(info_text)
    
    # Stats
    st.markdown("---")
    processed_count = len(st.session_state.processed_files)
    st.metric("Files Processed", processed_count)
    
    # System status
    st.markdown("---")
    if TRANSLATOR_READY:
        if hasattr(translator_service, 'image_translator') and translator_service.image_translator:
            st.success("✅ Visual Mode Ready")
        else:
            st.warning("⚠️ Text Mode Only")
    else:
        st.error("❌ System Not Ready")

# Main content area - using tabs
tabs = ["📤 Upload", "🎨 Process", "🖼️ Visual Results", "📊 Text Results", "📜 History"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tabs)

# Upload tab
with tab1:
    st.header("Upload Files for Visual Translation")
    
    # Show sample before/after if available
    col1, col2 = st.columns(2)
    with col1:
        st.info("📥 **Upload files with foreign text**")
    with col2:
        st.success("📤 **Get images with English text!**")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to translate visually",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Upload images/PDFs with Chinese, Japanese, Korean, or Hindi text"
    )
    
    if uploaded_files:
        file_count = len(uploaded_files)
        st.success(f"✅ {file_count} file(s) uploaded!")
        
        # Display uploaded files info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Uploaded Files:")
            for file in uploaded_files:
                icon = get_file_icon(file.name)
                size_str = format_file_size(file.size)
                st.write(f"{icon} {file.name} ({size_str})")
        
        with col2:
            st.subheader("📊 Processing Info:")
            st.write(f"**Total files:** {file_count}")
            
            # Count file types
            image_count = sum(1 for f in uploaded_files if f.name.lower().endswith(('.jpg', '.jpeg', '.png')))
            pdf_count = sum(1 for f in uploaded_files if f.name.lower().endswith('.pdf'))
            
            if image_count > 0:
                st.write(f"🖼️ Images: {image_count} (will be visually translated)")
            if pdf_count > 0:
                st.write(f"📄 PDFs: {pdf_count} (will be visually translated)")
            
            # REMOVED: Estimated time calculation and display
            # est_seconds = image_count * 5 + pdf_count * 15
            # st.write(f"⏱️ Estimated time: {est_seconds} seconds")

# Process tab
with tab2:
    st.header("🎨 Visual Translation Processing")
    
    if not TRANSLATOR_READY:
        st.error("❌ Translator not initialized. Check console for errors.")
    elif uploaded_files:
        
        # Show processing info
        st.info(
            "**Visual Translation will:**\n"
            "• Detect text in your images\n"
            "• Translate to English\n"
            "• Replace text visually in the image\n"
            "• Return new images/PDFs with English text\n\n"
            "*(Note: For images with transparent backgrounds, the output will have a white background.)*" # Added note
        )
        
        # Process button
        if st.button("🚀 Start Visual Translation", key="process_btn"):
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            with st.spinner("Initializing visual translation system..."):
                status_text.text("Loading translation models...")
                
                # Clear previous results
                st.session_state.translated_files_paths = []
                
                # Process files
                total_files = len(uploaded_files)
                processed = []
                errors = []
                
                for idx, file in enumerate(uploaded_files):
                    # Update progress
                    progress = (idx + 1) / total_files
                    progress_bar.progress(progress)
                    
                    msg = f"🎨 Visually translating {file.name}... ({idx+1}/{total_files})"
                    status_text.text(msg)
                    
                    # Save uploaded file temporarily
                    temp_dir = tempfile.mkdtemp()
                    temp_path = Path(temp_dir) / file.name
                    
                    try:
                        with open(temp_path, 'wb') as f:
                            f.write(file.getvalue())
                        
                        # Call visual translator
                        if TRANSLATOR_READY:
                            trans_result = translator_service.translate_file(
                                str(temp_path),
                                source_language
                            )
                            
                            # Store result with paths
                            result = {
                                'filename': file.name,
                                'status': trans_result.get('status', 'error'),
                                'original_path': str(temp_path),
                                'translated_path': trans_result.get('output_path'),
                                'original': trans_result.get('original', 'Processing...'),
                                'translated': trans_result.get('translated', 'Visual translation'),
                                'method': trans_result.get('method', 'Visual'),
                                'file_type': trans_result.get('file_type', 'unknown'),
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Keep track of translated file paths
                            if result['translated_path']:
                                st.session_state.translated_files_paths.append(result['translated_path'])
                        else:
                            result = {
                                'filename': file.name,
                                'status': 'error',
                                'original_path': str(temp_path),
                                'translated_path': None,
                                'original': 'Translator not available',
                                'translated': 'Please check setup',
                                'method': 'None',
                                'file_type': 'unknown',
                                'timestamp': datetime.now().isoformat()
                            }
                        
                        processed.append(result)
                        st.session_state.processed_files.append(result)
                        
                    except Exception as e:
                        error_result = {
                            'filename': file.name,
                            'status': 'error',
                            'original_path': None,
                            'translated_path': None,
                            'original': f'Error: {str(e)}',
                            'translated': 'Processing failed',
                            'method': 'Error',
                            'file_type': 'unknown',
                            'timestamp': datetime.now().isoformat()
                        }
                        errors.append(error_result)
                        st.session_state.processed_files.append(error_result)
                
                # Complete
                progress_bar.progress(1.0)
                
                if errors:
                    status_text.text(f"⚠️ Completed with {len(errors)} errors")
                else:
                    status_text.text("✅ Visual translation complete!")
                
                # Show results summary
                with results_container:
                    st.success(f"🎉 Processed {len(processed)} files!")
                    
                    # Quick stats
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        success_count = len([p for p in processed if p['status'] == 'success'])
                        st.metric("Successful", success_count)
                    
                    with col2:
                        images_translated = len([p for p in processed if p['file_type'] == 'image'])
                        st.metric("Images Translated", images_translated)
                    
                    with col3:
                        if st.session_state.translated_files_paths:
                            st.metric("Files Ready", len(st.session_state.translated_files_paths))
                    
                    st.info("👉 Go to 'Visual Results' tab to see and download your translated images!")
    else:
        st.warning("⚠️ Please upload files in the Upload tab first")

# Visual Results tab - NEW!
with tab3:
    st.header("🖼️ Visual Translation Results")
    
    if st.session_state.processed_files:
        # Filter for visual results (images and PDFs)
        visual_results = [
            r for r in st.session_state.processed_files 
            if r.get('translated_path') and r['file_type'] in ['image', 'pdf', 'zip']
        ]
        
        if visual_results:
            # Download all button
            if st.session_state.translated_files_paths:
                st.subheader("📥 Download All Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Create ZIP of all translated files
                    if st.button("💾 Download All as ZIP"):
                        with st.spinner("Creating ZIP..."):
                            zip_path = create_download_zip(st.session_state.translated_files_paths)
                            with open(zip_path, 'rb') as f:
                                zip_data = f.read()
                            
                            st.download_button(
                                label="📥 Download ZIP",
                                data=zip_data,
                                file_name=f"translated_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip"
                            )
                            os.unlink(zip_path)
                
                st.markdown("---")
            
            # Display each result
            st.subheader("🎨 Individual Results")
            
            for idx, result in enumerate(visual_results):
                with st.expander(f"{get_file_icon(result['filename'])} {result['filename']}", expanded=(idx == 0)):
                    
                    if result['file_type'] == 'image':
                        # Show image comparison
                        # --- CHANGE 3: Pass idx to display_image_comparison ---
                        display_image_comparison(
                            result.get('original_path'),
                            result.get('translated_path'),
                            result['filename'],
                            idx # Passed idx here
                        )
                        
                    elif result['file_type'] == 'pdf':
                        st.info(f"📄 PDF translated: {result['filename']}")
                        st.write(f"Method: {result['method']}")
                        
                        # Download button for PDF
                        if result['translated_path'] and os.path.exists(result['translated_path']):
                            with open(result['translated_path'], 'rb') as f:
                                pdf_data = f.read()
                            st.download_button(
                                label=f"💾 Download Translated PDF",
                                data=pdf_data,
                                file_name=f"translated_{result['filename']}",
                                mime="application/pdf",
                                key=f"download_pdf_{result['filename']}_{idx}" # Added unique key
                            )
                    
                    elif result['file_type'] == 'zip':
                        st.info(f"📦 ZIP translated: {result['filename']}")
                        st.write(f"Method: {result['method']}")
                        
                        # Download button for ZIP
                        if result['translated_path'] and os.path.exists(result['translated_path']):
                            with open(result['translated_path'], 'rb') as f:
                                zip_data = f.read()
                            st.download_button(
                                label=f"💾 Download Translated ZIP",
                                data=zip_data,
                                file_name=f"translated_{result['filename']}",
                                mime="application/zip",
                                key=f"download_zip_{result['filename']}_{idx}" # Added unique key
                            )
        else:
            st.info("No visual translation results yet. Process some images or PDFs first!")
    else:
        st.info("📭 No results yet. Process some files to see visual translations here!")

# Text Results tab (original functionality)
with tab4:
    st.header("📝 Text Extraction Results")
    
    if st.session_state.processed_files:
        # Show text results
        for result in st.session_state.processed_files:
            if result['status'] == 'success':
                with st.expander(f"{result['filename']} - {result['method']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.text_area(
                            "Original Text:",
                            result['original'][:500],
                            height=150,
                            key=f"orig_{result['filename']}"
                        )
                    
                    with col2:
                        st.text_area(
                            "Translated Text:",
                            result['translated'][:500],
                            height=150,
                            key=f"trans_{result['filename']}"
                        )
    else:
        st.info("📭 No text results yet")

# History tab
with tab5:
    st.header("📜 Translation History")
    
    if st.session_state.processed_files:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        files = st.session_state.processed_files
        
        with col1:
            st.metric("Total Processed", len(files))
        
        with col2:
            visual_count = len([f for f in files if f.get('translated_path')])
            st.metric("Visual Translations", visual_count)
        
        with col3:
            success = sum(1 for f in files if f['status'] == 'success')
            st.metric("Successful", success)
        
        with col4:
            errors = sum(1 for f in files if f['status'] == 'error')
            st.metric("Errors", errors)
        
        # Clear history button
        if st.button("🗑️ Clear History", key="clear_history"):
            # Clean up translated files
            for file_path in st.session_state.translated_files_paths:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
            
            # Clear session state
            st.session_state.processed_files = []
            # st.session_state.translation_history = [] # REMOVED: Unused
            st.session_state.translated_files_paths = []
            
            # Clean up translator temp files
            if TRANSLATOR_READY:
                translator_service.cleanup()
            
            st.rerun()
    else:
        st.info("📭 No history yet")

# Footer
st.markdown("---")
footer_html = """
<div style='text-align: center'>
    <p><strong>Universal Translator v1.5 - Visual Translation</strong></p>
    <p>🎨 Now with visual text replacement in images!</p>
    <p style='font-size: 0.9em'>Translates: Chinese, Japanese, Korean, Hindi → English</p>
    <p style='font-size: 0.8em'>Built with ❤️ by Victor</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
