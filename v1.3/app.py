# app.py - Universal Translator v1.5 UI
"""
Universal Translator Web Interface.

Version: 1.5
Developer: Victor
"""

import os
import tempfile
import zipfile
import json
from datetime import datetime
from pathlib import Path

import streamlit as st

# Additional imports for downloads
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️ ReportLab not available for PDF generation")

# Import translator service
try:
    from translator_integration import translator_service
    TRANSLATOR_READY = True
except ImportError as e:
    TRANSLATOR_READY = False
    print(f"⚠️ Translator not available: {e}")

# Constants
MAX_LINE_LENGTH = 79
SUPPORTED_TYPES = ['jpg', 'jpeg', 'png', 'pdf', 'txt', 'zip']
LANGUAGES = ["Auto-detect", "Chinese", "Japanese", "Korean", "Hindi"]
OUTPUT_FORMATS = ["PDF", "Text", "Both"]

# Page configuration
st.set_page_config(
    page_title="Universal Translator v1.5",
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
</style>
"""
st.markdown(css_style, unsafe_allow_html=True)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []


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


# Header
st.title("🌍 Universal Translator v1.5")
header_text = (
    "**Translate documents from Chinese, Japanese, "
    "Korean, and Hindi to English**"
)
st.markdown(header_text)
st.markdown("---")

# Check translator status
if not TRANSLATOR_READY:
    st.error(
        "⚠️ Translator components not loaded. "
        "Please check translator_integration.py"
    )

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Language selection
    source_language = st.selectbox(
        "Source Language",
        LANGUAGES,
        help="Select source language or auto-detect"
    )
    
    # Output format
    output_format = st.selectbox(
        "Output Format",
        OUTPUT_FORMATS,
        help="Choose your preferred output format"
    )
    
    # Processing options
    st.subheader("📋 Processing Options")
    include_original = st.checkbox(
        "Include original text",
        value=True
    )
    create_report = st.checkbox(
        "Generate processing report",
        value=False
    )
    
    # Info section
    st.markdown("---")
    info_text = (
        "**Supported Files:**\n"
        "• Images (JPG, PNG)\n"
        "• PDFs\n"
        "• Text files\n"
        "• ZIP archives"
    )
    st.info(info_text)
    
    # Stats
    st.markdown("---")
    processed_count = len(st.session_state.processed_files)
    st.metric("Files Processed", processed_count)
    
    # Translator status indicator
    st.markdown("---")
    if TRANSLATOR_READY:
        st.success("✅ Translator Ready")
    else:
        st.error("❌ Translator Not Loaded")

# Main content area - using tabs
tabs = ["📤 Upload", "📝 Process", "📊 Results", "📜 History"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)

# Upload tab
with tab1:
    st.header("Upload Files")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files to translate",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        help="Upload one or more files for translation"
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
            st.subheader("📊 Summary:")
            st.write(f"**Total files:** {file_count}")
            
            total_bytes = sum(f.size for f in uploaded_files)
            total_mb = total_bytes / (1024 * 1024)
            st.write(f"**Total size:** {total_mb:.2f} MB")
            
            # Estimate processing time
            est_seconds = file_count * 3
            st.write(f"**Estimated time:** {est_seconds} seconds")

# Process tab
with tab2:
    st.header("Process Files")
    
    if not TRANSLATOR_READY:
        st.error(
            "❌ Translator not initialized. "
            "Check console for errors."
        )
    elif uploaded_files:
        # Process button
        if st.button("🚀 Start Translation", key="process_btn"):
            
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            with st.spinner("Initializing translator..."):
                status_msg = "Loading translation modules..."
                status_text.text(status_msg)
                
                # Process files
                total_files = len(uploaded_files)
                processed = []
                errors = []
                
                for idx, file in enumerate(uploaded_files):
                    # Update progress
                    progress = (idx + 1) / total_files
                    progress_bar.progress(progress)
                    
                    msg = (
                        f"Processing {file.name}... "
                        f"({idx+1}/{total_files})"
                    )
                    status_text.text(msg)
                    
                    # Save uploaded file temporarily
                    temp_dir = tempfile.mkdtemp()
                    temp_path = Path(temp_dir) / file.name
                    
                    try:
                        with open(temp_path, 'wb') as f:
                            f.write(file.getvalue())
                        
                        # Call REAL translator
                        if TRANSLATOR_READY:
                            trans_result = translator_service.translate_file(
                                str(temp_path),
                                source_language
                            )
                            
                            result = {
                                'filename': file.name,
                                'status': trans_result.get(
                                    'status', 'error'
                                ),
                                'original': trans_result.get(
                                    'original', 'No text extracted'
                                ),
                                'translated': trans_result.get(
                                    'translated', 'Translation pending'
                                ),
                                'method': trans_result.get(
                                    'method', 'Unknown'
                                ),
                                'timestamp': datetime.now().isoformat()
                            }
                        else:
                            # Fallback if translator not ready
                            result = {
                                'filename': file.name,
                                'status': 'error',
                                'original': 'Translator not available',
                                'translated': 'Please check setup',
                                'method': 'None',
                                'timestamp': datetime.now().isoformat()
                            }
                        
                        processed.append(result)
                        st.session_state.processed_files.append(result)
                        
                    except Exception as e:
                        error_result = {
                            'filename': file.name,
                            'status': 'error',
                            'original': f'Error: {str(e)}',
                            'translated': 'Processing failed',
                            'method': 'Error',
                            'timestamp': datetime.now().isoformat()
                        }
                        errors.append(error_result)
                        st.session_state.processed_files.append(
                            error_result
                        )
                    
                    finally:
                        # Clean up temp file
                        try:
                            if temp_path.exists():
                                temp_path.unlink()
                            if Path(temp_dir).exists():
                                Path(temp_dir).rmdir()
                        except:
                            pass
                
                # Complete
                progress_bar.progress(1.0)
                
                # Clean up translator temp files
                if TRANSLATOR_READY:
                    try:
                        translator_service.cleanup()
                    except:
                        pass
                
                if errors:
                    status_text.text(
                        f"⚠️ Completed with {len(errors)} errors"
                    )
                else:
                    status_text.text("✅ Translation complete!")
                
                # Show results summary
                with results_container:
                    if processed:
                        success_count = len([
                            p for p in processed 
                            if p['status'] == 'success'
                        ])
                        msg = (
                            f"🎉 Processed {len(processed)} files "
                            f"({success_count} successful)"
                        )
                        st.success(msg)
                    
                    if errors:
                        st.error(f"❌ {len(errors)} files failed")
                    
                    # Quick preview of first successful result
                    successful = [
                        p for p in processed 
                        if p['status'] == 'success'
                    ]
                    if successful:
                        st.subheader("Preview (First Success):")
                        first = successful[0]
                        
                        # Show method used
                        st.caption(
                            f"Method: {first.get('method', 'OCR')}"
                        )
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.text_area(
                                "Original:",
                                first['original'][:500],
                                height=150,
                                key="preview_orig"
                            )
                        
                        with col2:
                            st.text_area(
                                "Translation:",
                                first['translated'][:500],
                                height=150,
                                key="preview_trans"
                            )
    else:
        warning_msg = "⚠️ Please upload files in the Upload tab first"
        st.warning(warning_msg)

# Results tab - WITH WORKING DOWNLOADS
with tab3:
    st.header("Translation Results")
    
    if st.session_state.processed_files:
        # Download section
        st.subheader("📥 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Download All (ZIP)", key="download_zip"):
                with st.spinner("Creating ZIP file..."):
                    try:
                        # Create ZIP filename
                        zip_filename = f"translations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                        
                        # Create temporary ZIP file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                            with zipfile.ZipFile(tmp_zip.name, 'w') as zf:
                                # Add each result as a text file
                                for idx, result in enumerate(st.session_state.processed_files):
                                    # Create filename
                                    base_name = Path(result['filename']).stem
                                    txt_name = f"{base_name}_translated.txt"
                                    
                                    # Create content
                                    content = f"File: {result['filename']}\n"
                                    content += f"Status: {result['status']}\n"
                                    content += f"Method: {result.get('method', 'N/A')}\n"
                                    content += "="*50 + "\n\n"
                                    content += "ORIGINAL:\n"
                                    content += result['original'] + "\n\n"
                                    content += "TRANSLATION:\n"
                                    content += result['translated'] + "\n"
                                    
                                    # Add to ZIP
                                    zf.writestr(txt_name, content)
                                
                                # Add summary file
                                summary = {
                                    'total_files': len(st.session_state.processed_files),
                                    'timestamp': datetime.now().isoformat(),
                                    'results': [
                                        {
                                            'file': r['filename'],
                                            'status': r['status']
                                        } for r in st.session_state.processed_files
                                    ]
                                }
                                zf.writestr('summary.json', json.dumps(summary, indent=2))
                            
                            # Read ZIP file
                            tmp_zip.seek(0)
                            with open(tmp_zip.name, 'rb') as f:
                                zip_data = f.read()
                            
                            # Offer download
                            st.download_button(
                                label="📥 Download ZIP",
                                data=zip_data,
                                file_name=zip_filename,
                                mime="application/zip"
                            )
                            st.success("✅ ZIP file ready!")
                            
                    except Exception as e:
                        st.error(f"Failed to create ZIP: {str(e)}")
        
        with col2:
            if st.button("📄 Download PDF Report", key="download_pdf"):
                with st.spinner("Generating PDF..."):
                    try:
                        if REPORTLAB_AVAILABLE:
                            # Create PDF with reportlab
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                                c = canvas.Canvas(tmp_pdf.name, pagesize=letter)
                                
                                # Add title
                                c.setFont("Helvetica-Bold", 16)
                                c.drawString(50, 750, "Translation Report")
                                c.setFont("Helvetica", 10)
                                c.drawString(50, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
                                
                                # Add results
                                y_position = 700
                                for idx, result in enumerate(st.session_state.processed_files[:10]):  # First 10
                                    if y_position < 100:  # New page if needed
                                        c.showPage()
                                        y_position = 750
                                    
                                    c.setFont("Helvetica-Bold", 12)
                                    c.drawString(50, y_position, f"File: {result['filename']}")
                                    y_position -= 20
                                    
                                    c.setFont("Helvetica", 10)
                                    # Status
                                    status_text = f"Status: {result['status']}"
                                    c.drawString(50, y_position, status_text)
                                    y_position -= 20
                                    
                                    # Original (truncated)
                                    c.drawString(50, y_position, "Original:")
                                    y_position -= 15
                                    original_text = result['original'][:100] + "..."
                                    c.drawString(70, y_position, original_text[:80])
                                    y_position -= 20
                                    
                                    # Translation (truncated)
                                    c.drawString(50, y_position, "Translation:")
                                    y_position -= 15
                                    trans_text = result['translated'][:100] + "..."
                                    c.drawString(70, y_position, trans_text[:80])
                                    y_position -= 30
                                
                                c.save()
                                
                                # Read PDF
                                with open(tmp_pdf.name, 'rb') as f:
                                    pdf_data = f.read()
                                
                                # Offer download
                                st.download_button(
                                    label="📥 Download PDF Report",
                                    data=pdf_data,
                                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                                st.success("✅ PDF report ready!")
                        else:
                            st.error("PDF generation requires reportlab. Install with: pip install reportlab")
                            
                    except Exception as e:
                        st.error(f"Failed to create PDF: {str(e)}")
        
        with col3:
            if st.button("📝 Download Text", key="download_txt"):
                # Create text content
                text_content = "TRANSLATION RESULTS\n"
                text_content += "="*50 + "\n\n"
                
                for result in st.session_state.processed_files:
                    text_content += f"File: {result['filename']}\n"
                    text_content += f"Status: {result['status']}\n"
                    text_content += f"Method: {result.get('method', 'N/A')}\n"
                    text_content += "-"*30 + "\n"
                    text_content += "Original:\n"
                    text_content += result['original'][:500] + "\n\n"
                    text_content += "Translation:\n"
                    text_content += result['translated'][:500] + "\n"
                    text_content += "="*50 + "\n\n"
                
                st.download_button(
                    label="📥 Download Text File",
                    data=text_content,
                    file_name=f"translations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                st.success("✅ Text file ready!")
        
        st.markdown("---")
        
        # Results details
        st.subheader("📋 Detailed Results")
        
        # Filter options
        show_status = st.selectbox(
            "Filter by status:",
            ["All", "Success", "Error"],
            key="filter_status"
        )
        
        # Filter results
        if show_status == "Success":
            filtered = [
                r for r in st.session_state.processed_files 
                if r['status'] == 'success'
            ]
        elif show_status == "Error":
            filtered = [
                r for r in st.session_state.processed_files 
                if r['status'] == 'error'
            ]
        else:
            filtered = st.session_state.processed_files
        
        # Display filtered results
        for idx, result in enumerate(filtered):
            status_icon = "✅" if result['status'] == 'success' else "❌"
            label = (
                f"{status_icon} {result['filename']} - "
                f"{result['status'].upper()}"
            )
            
            with st.expander(label):
                # Show method if available
                if 'method' in result:
                    st.caption(f"Processing method: {result['method']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Original Text:**")
                    st.text(result['original'][:500])
                
                with col2:
                    st.write("**Translated Text:**")
                    st.text(result['translated'][:500])
                
                caption = f"Processed at: {result['timestamp']}"
                st.caption(caption)
    else:
        info_msg = (
            "📭 No results yet. "
            "Process some files to see results here!"
        )
        st.info(info_msg)

# History tab
with tab4:
    st.header("Translation History")
    
    if st.session_state.processed_files:
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        files = st.session_state.processed_files
        
        with col1:
            st.metric("Total Translations", len(files))
        
        with col2:
            success = sum(1 for f in files if f['status'] == 'success')
            st.metric("Successful", success)
        
        with col3:
            errors = sum(1 for f in files if f['status'] == 'error')
            st.metric("Errors", errors)
        
        with col4:
            # Get unique dates
            dates = set(
                f['timestamp'].split('T')[0]
                for f in files
            )
            st.metric("Active Days", len(dates))
        
        st.markdown("---")
        
        # History table
        st.subheader("📜 Recent Translations")
        
        # Create table data
        history_data = []
        recent = st.session_state.processed_files[-10:]
        
        for file in recent:
            status_icon = "✅" if file['status'] == 'success' else "❌"
            history_data.append({
                "File": file['filename'],
                "Status": f"{status_icon} {file['status']}",
                "Method": file.get('method', 'N/A'),
                "Time": file['timestamp'].split('T')[0]
            })
        
        st.table(history_data)
        
        # Clear history button
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state.processed_files = []
            st.session_state.translation_history = []
            st.rerun()
    else:
        st.info("📭 No translation history yet")

# Footer
st.markdown("---")
footer_lines = [
    "<div style='text-align: center'>",
    "<p>Universal Translator v1.5 | Built with ❤️ by Victor</p>",
    "<p style='font-size: 0.8em'>",
    "Supports: Chinese, Japanese, Korean, Hindi → English",
    "</p>",
    "</div>"
]
footer_html = "".join(footer_lines)
st.markdown(footer_html, unsafe_allow_html=True)

# Show debug info in sidebar if translator not ready
if not TRANSLATOR_READY:
    with st.sidebar:
        st.error("Debug: Check console for import errors")