#!/usr/bin/env python3
"""
INSTANT Demo Compressor - Shows UI workflow without slow compression
Uses simple gzip for demo purposes, with a mock "AI compression" interface
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import gradio as gr
import gzip
import tempfile
from datetime import datetime
import time

# ==================== INSTANT COMPRESSION (for demo) ====================

def compress_text_instant(text):
    """Instant compression using gzip (for demo/testing)"""
    if not text or len(text.strip()) == 0:
        return "❌ Please enter some text to compress", None
    
    if len(text) > 50000:
        return f"❌ Text too long ({len(text)} chars). Max 50,000 characters.", None
    
    try:
        # Simulate AI processing (instant!)
        print(f"⚡ Compressing {len(text)} characters instantly...")
        start_time = datetime.now()
        
        # Use gzip for instant compression (demo mode)
        text_bytes = text.encode('utf-8')
        compressed_bytes = gzip.compress(text_bytes, compresslevel=9)
        
        compression_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate statistics
        original_size = len(text_bytes)
        compressed_size = len(compressed_bytes)
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        # Save compressed file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compressed_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=".gz",
            prefix=f"compressed_{timestamp}_"
        )
        
        with open(compressed_file.name, 'wb') as f:
            f.write(compressed_bytes)
        
        # Create result message
        result = f"""**Compression Complete**

**Statistics:**
- Original: {original_size:,} bytes ({len(text)} chars)
- Compressed: {compressed_size:,} bytes
- Ratio: {ratio:.2f}x
- Saved: {100 * (1 - compressed_size/original_size):.1f}%
- Time: {compression_time:.3f}s

Download compressed file below
"""
        
        print(f"✅ Done! Ratio: {ratio:.2f}x in {compression_time:.3f}s")
        return result, compressed_file.name
        
    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}"
        print(error_msg)
        return error_msg, None


def decompress_text_instant(file):
    """Instant decompression"""
    if file is None:
        return "❌ Please upload a compressed file"
    
    try:
        with open(file.name, 'rb') as f:
            compressed_bytes = f.read()
        
        decompressed_bytes = gzip.decompress(compressed_bytes)
        text = decompressed_bytes.decode('utf-8')
        
        result = f"""**Decompression Complete**

**Statistics:**
- Compressed: {len(compressed_bytes):,} bytes
- Decompressed: {len(decompressed_bytes):,} bytes
- Text length: {len(text)} characters

**Decompressed Text:**
```
{text[:500]}{'...' if len(text) > 500 else ''}
```
"""
        return result
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}"


def compress_file_instant(file):
    """Compress uploaded file"""
    if file is None:
        return "❌ Please upload a file", None
    
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if len(text) > 50000:
            text = text[:50000]
            
        result, compressed_file = compress_text_instant(text)
        return result, compressed_file
        
    except UnicodeDecodeError:
        return "❌ File must be a text file (UTF-8)", None
    except Exception as e:
        return f"❌ Error: {str(e)}", None


# ==================== Gradio Interface ====================

css = """
* {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.gradio-container {
    max-width: 1400px;
    margin: auto;
    background-color: #1a1a1a !important;
}
body {
    background-color: #1a1a1a !important;
}
.main-header {
    color: #ffffff;
    font-weight: 600;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 20px;
}
.compress-btn {
    background: #d97634 !important;
    border: 1px solid #d97634 !important;
    color: #ffffff !important;
    font-size: 1.1em !important;
    padding: 12px !important;
}
.compress-btn:hover {
    background: #c2651e !important;
    border: 1px solid #c2651e !important;
}

/* Override ALL backgrounds to dark grey */
.gr-box, .gr-form, .gr-panel, .gr-padded, div[class*="wrap"], 
div[class*="container"], section, .block, .gr-button-secondary,
.gr-input-label, .markdown, .prose {
    background-color: #1a1a1a !important;
    background: #1a1a1a !important;
}

/* Text areas and inputs - dark grey background with white text */
.gr-input, .gr-textarea, textarea, input[type="text"],
textarea.scroll-hide, .gr-text-input {
    background-color: #2a2a2a !important;
    background: #2a2a2a !important;
    border-color: #3a3a3a !important;
    color: #ffffff !important;
}

/* Markdown/Results areas - dark grey with white text */
.markdown-text, .gr-markdown, .prose, .markdown,
div[class*="markdown"], div[class*="prose"] {
    background-color: #1a1a1a !important;
    background: #1a1a1a !important;
    color: #ffffff !important;
}

/* All text should be white */
label, p, span, div, h1, h2, h3, h4, h5, h6, 
.markdown-text, .markdown-text *, .prose *, 
pre, code, strong, em, li {
    color: #ffffff !important;
}

/* Buttons - orange primary, dark grey secondary */
.gr-button-primary {
    background-color: #d97634 !important;
    border-color: #d97634 !important;
    color: #ffffff !important;
}
.gr-button-primary:hover {
    background-color: #c2651e !important;
    border-color: #c2651e !important;
}
.gr-button-secondary {
    background-color: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    color: #ffffff !important;
}
.gr-button-secondary:hover {
    background-color: #3a3a3a !important;
}

/* File upload areas */
.file-preview, .upload-container {
    background-color: #2a2a2a !important;
    border-color: #3a3a3a !important;
}

/* Tabs */
.tab-nav {
    background-color: #1a1a1a !important;
}
"""

with gr.Blocks(css=css, title="Text Compressor", theme=gr.themes.Base()) as app:
    
    gr.HTML("""
    <div class="main-header">
        RAG-LLM Lossless Data Compressor
    </div>
    """)
    
    gr.Markdown("""
    This is a CPU tool. I didn't have money 😔 to buy cloud GPU resources for better compression.
    
    1. <a href="https://drive.google.com/file/d/1eQlX4rz9dsl4EnZzbYY4YbNJJ0WTW1t-/view?usp=sharing" target="_blank">Read the Research Proposal</a>
    2. Didn't understand? <a href="https://drive.google.com/file/d/1qPx-EZFVsWTZV0G1WR0KioYj5HepXD5h/view?usp=sharing" target="_blank">See a video</a>
    """)
    
    with gr.Tabs():
        # Tab 1: Text Compression
        with gr.Tab("Compress"):
            gr.Markdown("### Paste your text and compress instantly")
            
            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="Input Text",
                        placeholder="Paste your text here (up to 50,000 characters)...",
                        lines=15,
                        max_lines=20
                    )
                    
                    with gr.Row():
                        compress_btn = gr.Button(
                            "Compress",
                            variant="primary",
                            elem_classes="compress-btn",
                            size="lg"
                        )
                        clear_btn = gr.ClearButton([text_input], value="Clear")
                    
                    gr.Examples(
                        examples=[
                            "The quick brown fox jumps over the lazy dog. " * 10,
                            "Hello, World! Welcome to instant compression technology. " * 20,
                            "Python is an interpreted, high-level, general-purpose programming language. " * 15,
                        ],
                        inputs=text_input,
                        label="Quick Examples"
                    )
                
                with gr.Column(scale=1):
                    result_output = gr.Markdown(
                        value="### Enter text and click compress"
                    )
                    compressed_download = gr.File(
                        label="Download Compressed File",
                        visible=True
                    )
        
        # Tab 2: Decompress
        with gr.Tab("Decompress"):
            gr.Markdown("### Upload a compressed file to decompress")
            
            with gr.Row():
                with gr.Column():
                    decompress_file_input = gr.File(
                        label="Upload Compressed File (.gz)",
                        file_types=[".gz"]
                    )
                    
                    decompress_btn = gr.Button(
                        "Decompress",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column():
                    decompress_result = gr.Markdown(value="### Upload a compressed file")
        
        # Tab 3: About
        with gr.Tab("About"):
            gr.Markdown("""
            ## About This Tool
            
            This is a research project on **RAG-LLM Lossless Data Compression**. 
            
            All mathematical details, algorithms, and implementation specifics are available in the GitHub repository.
            
            **GitHub Repository:** [https://github.com/nrpatel18/RAG-compressor](https://github.com/nrpatel18/RAG-compressor)
            
            ### Features:
            
            - Instant compression - Results in milliseconds  
            - Lossless compression - Perfect reconstruction  
            - Decompression - Restore original text  
            - Statistics - See compression ratios  
            
            ### Performance:
            
            - Speed: < 1 second for most texts
            - Ratio: Typically 2-4x for text
            - Method: gzip (industry standard)
            - Max size: 50,000 characters
            
            ### For LLM Compression:
            
            This tool uses gzip for instant results. For LLM-based compression with potentially better ratios on natural language:
            
            ```bash
            # Use the CLI version
            python LLMCompress.py input.txt output.llmzip
            ```
            
            The CLI version uses GPT-2 or DistilGPT2 for compression, which can achieve better ratios on natural language but takes longer.
            
            ### Compression Methods:
            
            | Method | Speed | Ratio (text) | Use Case |
            |--------|-------|--------------|----------|
            | gzip (this tool) | Instant | 2-4x | Quick testing, general use |
            | LLM (CLI) | Slow (5-30s) | 2-5x | Natural language, better ratios |
            | BGPT (image/audio) | Very slow | Varies | Multimedia compression |
            
            ### Tips:
            
            - Longer texts compress better
            - Repetitive text compresses very well
            - Natural language compresses well
            - Random data compresses poorly
            """)
    
    # Event handlers
    compress_btn.click(
        fn=compress_text_instant,
        inputs=[text_input],
        outputs=[result_output, compressed_download]
    )
    
    decompress_btn.click(
        fn=decompress_text_instant,
        inputs=[decompress_file_input],
        outputs=[decompress_result]
    )


if __name__ == "__main__":
    print("=" * 60)
    print("⚡ INSTANT COMPRESSION DEMO")
    print("=" * 60)
    print("Method: gzip (instant)")
    print("Speed: < 1 second")
    print("Perfect for: Testing, demos, quick compression")
    print("=" * 60)
    print()
    print("🚀 Starting web interface...")
    print("   No model downloads needed!")
    print()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True,
        quiet=False
    )
