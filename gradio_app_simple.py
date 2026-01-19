#!/usr/bin/env python3
"""
Simple Gradio Web Interface for RAG-based Lossless Compressor
Loads models on-demand for faster startup
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import gradio as gr
import torch
import tempfile
from datetime import datetime

# Try to import compression modules
try:
    from LLMCompress import Metric, compress, decode, write_padded_bytes, read_padded_bytes
    COMPRESS_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import LLMCompress: {e}")
    COMPRESS_AVAILABLE = False


# ==================== Simple Compression Functions ====================

def compress_text_simple(text, model_name="gpt2", precision=32):
    """Compress text using LLM-based compression"""
    if not COMPRESS_AVAILABLE:
        return "❌ LLMCompress module not available", None, None
    
    if not text or len(text.strip()) == 0:
        return "❌ Please enter some text to compress", None, None
    
    try:
        # Limit text length
        if len(text) > 10000:
            return f"❌ Text too long ({len(text)} chars). Please use less than 10,000 characters.", None, None
        
        # Import model on-demand
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        status = "🔄 Loading model... This may take a moment on first run."
        print(status)
        
        # Load model (will be cached by transformers)
        device = "cpu"  # Force CPU for compatibility
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True
        )
        model.eval()
        
        status = "🔄 Compressing text..."
        print(status)
        
        # Compress
        start_time = datetime.now()
        _, compressed_bits = compress(
            model=model,
            tokenizer=tokenizer,
            input_text=text,
            prefix_len=1,
            device=device,
            precision=precision
        )
        compression_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate statistics
        original_size = len(text.encode('utf-8'))
        compressed_size = len(compressed_bits) // 8
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        # Save compressed file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        compressed_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=".llmzip",
            prefix=f"compressed_{timestamp}_"
        )
        
        # Write compressed data
        write_padded_bytes(compressed_bits, compressed_file.name)
        
        # Create result message
        result = f"""✅ **Compression Successful!**

📊 **Statistics:**
- Original size: {original_size:,} bytes ({len(text)} characters)
- Compressed size: {compressed_size:,} bytes
- Compression ratio: {ratio:.2f}x
- Space saved: {100 * (1 - compressed_size/original_size):.1f}%
- Time taken: {compression_time:.2f} seconds

💾 **Download your compressed file below**
"""
        
        return result, compressed_file.name, compressed_bits
        
    except Exception as e:
        return f"❌ **Compression Error:**\n\n{str(e)}", None, None


def compress_file_simple(file, model_name="gpt2", precision=32):
    """Compress text from uploaded file"""
    if file is None:
        return "❌ Please upload a file", None
    
    try:
        # Read file
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Compress
        result, compressed_file, _ = compress_text_simple(text, model_name, precision)
        return result, compressed_file
        
    except UnicodeDecodeError:
        return "❌ File must be a text file (UTF-8 encoded)", None
    except Exception as e:
        return f"❌ Error reading file: {str(e)}", None


def get_file_info(file):
    """Get information about a compressed file"""
    if file is None:
        return "❌ Please upload a compressed file"
    
    try:
        compressed_bits = read_padded_bytes(file.name)
        size_bytes = len(compressed_bits) // 8
        size_bits = len(compressed_bits)
        
        info = f"""📦 **Compressed File Information**

- File size: {size_bytes:,} bytes
- Size in bits: {size_bits:,} bits
- File path: {file.name}

ℹ️ **Note:** Full decompression requires the same model used for compression.
"""
        return info
        
    except Exception as e:
        return f"❌ Error reading compressed file: {str(e)}"


# ==================== Gradio Interface ====================

def create_simple_interface():
    """Create a simple Gradio interface"""
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .output-text {
        font-size: 14px;
    }
    #component-0 {
        max-width: 1200px;
        margin: auto;
    }
    """
    
    with gr.Blocks(css=css, title="LLM Compressor") as app:
        gr.Markdown("""
        # 🗜️ LLM-Based Lossless Text Compressor
        
        Compress text using language models for **lossless compression**.
        
        ⚡ **Fast Start:** Models load on-demand when you compress.
        """)
        
        with gr.Tabs():
            # Tab 1: Text Compression
            with gr.Tab("💬 Text Compression"):
                gr.Markdown("### Compress text directly")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text to compress (up to 10,000 characters)...",
                            lines=10
                        )
                        
                        with gr.Row():
                            model_select = gr.Dropdown(
                                choices=["gpt2", "distilgpt2"],
                                value="gpt2",
                                label="Model (gpt2 recommended)"
                            )
                            precision_select = gr.Slider(
                                minimum=16,
                                maximum=64,
                                value=32,
                                step=8,
                                label="Precision (higher = better compression, slower)"
                            )
                        
                        compress_btn = gr.Button("🗜️ Compress", variant="primary")
                        
                        gr.Markdown("**Examples:**")
                        gr.Examples(
                            examples=[
                                "The quick brown fox jumps over the lazy dog.",
                                "To be or not to be, that is the question.",
                                "In the beginning God created the heaven and the earth."
                            ],
                            inputs=text_input
                        )
                    
                    with gr.Column(scale=2):
                        result_output = gr.Markdown(label="Result")
                        compressed_download = gr.File(label="📥 Download Compressed File")
                
            # Tab 2: File Compression
            with gr.Tab("📁 File Compression"):
                gr.Markdown("### Upload a text file to compress")
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            label="Upload Text File",
                            file_types=[".txt", ".md", ".json", ".csv"]
                        )
                        
                        with gr.Row():
                            model_select_file = gr.Dropdown(
                                choices=["gpt2", "distilgpt2"],
                                value="gpt2",
                                label="Model"
                            )
                            precision_select_file = gr.Slider(
                                minimum=16,
                                maximum=64,
                                value=32,
                                step=8,
                                label="Precision"
                            )
                        
                        compress_file_btn = gr.Button("🗜️ Compress File", variant="primary")
                    
                    with gr.Column():
                        file_result_output = gr.Markdown(label="Result")
                        file_compressed_download = gr.File(label="📥 Download Compressed File")
            
            # Tab 3: File Info
            with gr.Tab("ℹ️ File Info"):
                gr.Markdown("### View compressed file information")
                
                compressed_file_input = gr.File(label="Upload Compressed File (.llmzip)")
                info_btn = gr.Button("📊 Get Info")
                info_output = gr.Markdown()
        
        # Event handlers
        compress_btn.click(
            fn=compress_text_simple,
            inputs=[text_input, model_select, precision_select],
            outputs=[result_output, compressed_download]
        )
        
        compress_file_btn.click(
            fn=compress_file_simple,
            inputs=[file_input, model_select_file, precision_select_file],
            outputs=[file_result_output, file_compressed_download]
        )
        
        info_btn.click(
            fn=get_file_info,
            inputs=[compressed_file_input],
            outputs=[info_output]
        )
        
        gr.Markdown("""
        ---
        ### 📚 About
        
        This tool uses **language models** (GPT-2) to achieve lossless text compression.
        
        **Features:**
        - ✅ Truly lossless - perfect reconstruction
        - ✅ Better compression for natural language text
        - ✅ Works on CPU (no GPU required)
        - ✅ Fast startup - models load on-demand
        
        **Tips:**
        - Use `gpt2` for better compression
        - Higher precision = better compression but slower
        - Works best on English text
        """)
    
    return app


# ==================== Main ====================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG-Based Lossless Compressor - Simple Web Interface")
    print("=" * 60)
    print("Device: CPU (optimized for compatibility)")
    print("Models: Load on-demand (faster startup)")
    print("=" * 60)
    print()
    
    if not COMPRESS_AVAILABLE:
        print("⚠️  WARNING: LLMCompress module not available!")
        print("    Make sure LLMCompress.py is in the same directory.")
        print()
    
    # Create and launch interface
    app = create_simple_interface()
    
    print("🚀 Launching web interface...")
    print("   Models will download on first use (cached for future use)")
    print()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
