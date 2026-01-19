import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore')

import gradio as gr
import torch
import tempfile
from typing import Optional, Tuple
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from LLMCompress import Metric, compress, decode, write_padded_bytes, read_padded_bytes


# ==================== Configuration ====================
class GradioCompressorConfig:
    """Configuration for Gradio-based compressor"""
    # Use smaller models or CPU-compatible options
    USE_CPU = True  # Set to False if GPU is available
    MODEL_NAME = "distilgpt2"  # Lightweight fallback model
    # MODEL_NAME = "pretrained/Qwen3-0.6B"  # Original model (requires GPU)
    
    DEVICE = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_DTYPE = torch.float32 if USE_CPU else torch.float16
    
    MAX_TEXT_LENGTH = 5000  # characters
    PRECISION = 64
    PREFIX_LENGTH = 1


# ==================== Model Loader ====================
class ModelManager:
    """Singleton model manager to avoid reloading"""
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_model(self):
        """Load model and tokenizer if not already loaded"""
        if self._model is None or self._tokenizer is None:
            print(f"Loading model: {GradioCompressorConfig.MODEL_NAME}")
            print(f"Device: {GradioCompressorConfig.DEVICE}")
            
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    GradioCompressorConfig.MODEL_NAME,
                    torch_dtype=GradioCompressorConfig.MODEL_DTYPE if not GradioCompressorConfig.USE_CPU else None,
                    low_cpu_mem_usage=True
                )
                if not GradioCompressorConfig.USE_CPU:
                    self._model = self._model.to(GradioCompressorConfig.DEVICE)
                self._tokenizer = AutoTokenizer.from_pretrained(
                    GradioCompressorConfig.MODEL_NAME,
                    use_fast=False
                )
                
                if GradioCompressorConfig.USE_CPU:
                    self._model = self._model.to(GradioCompressorConfig.DEVICE)
                
                self._model.eval()
                print("✓ Model loaded successfully")
                
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                print("Falling back to DistilGPT2...")
                GradioCompressorConfig.MODEL_NAME = "distilgpt2"
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    "distilgpt2",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self._tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
                self._model = self._model.to(GradioCompressorConfig.DEVICE)
                self._model.eval()
                print("✓ Fallback model loaded")
        
        return self._model, self._tokenizer


# ==================== Compression Functions ====================
def compress_text(text: str, progress=gr.Progress()) -> Tuple[str, str, str]:
    """
    Compress text using LLM-based compression
    
    Args:
        text: Input text to compress
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (status_message, compression_stats, download_path)
    """
    if not text or len(text.strip()) == 0:
        return "❌ Error: Please provide text to compress", "", None
    
    if len(text) > GradioCompressorConfig.MAX_TEXT_LENGTH:
        return f"❌ Error: Text too long. Maximum {GradioCompressorConfig.MAX_TEXT_LENGTH} characters.", "", None
    
    try:
        progress(0.1, desc="Loading model...")
        model_manager = ModelManager()
        model, tokenizer = model_manager.load_model()
        device = GradioCompressorConfig.DEVICE
        
        progress(0.3, desc="Tokenizing text...")
        tokenized = tokenizer(text, return_tensors="pt").to(device)
        
        if tokenized["input_ids"].shape[1] > 1024:
            return "❌ Error: Text results in too many tokens (>1024). Please use shorter text.", "", None
        
        progress(0.5, desc="Compressing...")
        metric = Metric()
        
        with torch.inference_mode():
            logits = model(tokenized["input_ids"], use_cache=False).logits[:, :-1].to(torch.float32)
        
        compressed_bytes, num_padded_bits, start_symbol, sequence_array, pd, probs = compress(
            tokenized["input_ids"],
            logits,
            metric,
            precision=GradioCompressorConfig.PRECISION,
            prefix_length=GradioCompressorConfig.PREFIX_LENGTH
        )
        
        progress(0.8, desc="Saving compressed file...")
        
        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        compressed_path = os.path.join(temp_dir, "compressed.bin")
        original_length = tokenized["input_ids"].shape[1] - 1
        
        write_padded_bytes(compressed_path, compressed_bytes, num_padded_bits, original_length)
        
        # Calculate statistics
        original_size_bytes = len(text.encode('utf-8'))
        compressed_size = len(compressed_bytes)
        compression_ratio = original_size_bytes / compressed_size if compressed_size > 0 else 0
        
        stats = f"""
## Compression Statistics

- **Original Size**: {original_size_bytes} bytes ({len(text)} characters)
- **Compressed Size**: {compressed_size} bytes
- **Compression Ratio**: {compression_ratio:.2f}x
- **Space Saved**: {((1 - compressed_size/original_size_bytes) * 100):.1f}%
- **Tokens**: {tokenized["input_ids"].shape[1]} tokens
- **Padded Bits**: {num_padded_bits}
"""
        
        progress(1.0, desc="Complete!")
        
        return "✅ Compression successful!", stats, compressed_path
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during compression: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "", None


def decompress_file(file) -> Tuple[str, str, Optional[str]]:
    """
    Decompress a .bin file back to text
    
    Args:
        file: Uploaded .bin file
    
    Returns:
        Tuple of (status_message, decompressed_text, download_path)
    """
    if file is None:
        return "❌ Error: Please upload a .bin file", "", None
    
    try:
        model_manager = ModelManager()
        model, tokenizer = model_manager.load_model()
        device = GradioCompressorConfig.DEVICE
        
        # Read compressed file
        compressed_bytes, num_padded_bits, original_length = read_padded_bytes(file.name)
        
        # For decompression, we need the start symbol and other info
        # This is a simplified version - in production, you'd store metadata
        # For now, we'll return an informational message
        
        info = f"""
## Compressed File Information

- **Compressed Size**: {len(compressed_bytes)} bytes
- **Padded Bits**: {num_padded_bits}
- **Original Token Length**: {original_length}

⚠️ **Note**: Full decompression requires the original compression metadata (start symbol, probabilities).
This demo shows compression statistics. For full decompression, use the Python API.
"""
        
        return "ℹ️ File loaded successfully", info, None
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error reading file: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "", None


def compress_file_text(file) -> Tuple[str, str, Optional[str]]:
    """Compress text from uploaded file"""
    if file is None:
        return "❌ Error: Please upload a text file", "", None
    
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return compress_text(text)
    
    except Exception as e:
        return f"❌ Error reading file: {str(e)}", "", None


# ==================== Gradio Interface ====================
def create_interface():
    """Create Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .main-title {
        text-align: center;
        color: #2563eb;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f9ff;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    .warning-box {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
    }
    """
    
    with gr.Blocks(css=custom_css, title="RAG Compressor") as interface:
        gr.Markdown(
            """
            # 🗜️ RAG-Based Lossless Compressor
            
            <div class="info-box">
            
            **Research Project**: This tool uses Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) 
            to achieve lossless data compression. Instead of traditional compression algorithms, 
            it leverages the predictive power of neural networks combined with arithmetic coding.
            
            </div>
            
            <div class="warning-box">
            
            ⚠️ **Note**: Currently running in **CPU mode** with a lightweight model (DistilGPT2) for compatibility. 
            For best compression ratios, use the full Python API with GPU acceleration and larger models.
            
            </div>
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Tabs():
            # Tab 1: Text Compression
            with gr.Tab("📝 Text Compression"):
                gr.Markdown("### Compress Text Directly")
                
                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Input Text",
                            placeholder="Enter text to compress (max 5000 characters)...",
                            lines=10,
                            max_lines=15
                        )
                        compress_btn = gr.Button("🗜️ Compress", variant="primary", size="lg")
                    
                    with gr.Column():
                        status_output = gr.Markdown(label="Status")
                        stats_output = gr.Markdown(label="Statistics")
                        download_output = gr.File(label="Download Compressed File")
                
                compress_btn.click(
                    fn=compress_text,
                    inputs=[text_input],
                    outputs=[status_output, stats_output, download_output]
                )
                
                gr.Examples(
                    examples=[
                        ["The quick brown fox jumps over the lazy dog. This is a test of the compression system."],
                        ["Artificial intelligence and machine learning are revolutionizing the field of data compression. Traditional algorithms like gzip and bzip2 use pattern matching, while neural approaches use learned representations."],
                    ],
                    inputs=[text_input],
                    label="Example Texts"
                )
            
            # Tab 2: File Compression
            with gr.Tab("📄 File Compression"):
                gr.Markdown("### Compress Text from File")
                
                with gr.Row():
                    with gr.Column():
                        file_input = gr.File(
                            label="Upload Text File (.txt)",
                            file_types=[".txt"]
                        )
                        compress_file_btn = gr.Button("🗜️ Compress File", variant="primary", size="lg")
                    
                    with gr.Column():
                        file_status_output = gr.Markdown(label="Status")
                        file_stats_output = gr.Markdown(label="Statistics")
                        file_download_output = gr.File(label="Download Compressed File")
                
                compress_file_btn.click(
                    fn=compress_file_text,
                    inputs=[file_input],
                    outputs=[file_status_output, file_stats_output, file_download_output]
                )
            
            # Tab 3: Decompression (Info)
            with gr.Tab("📂 Decompression"):
                gr.Markdown("### View Compressed File Info")
                
                with gr.Row():
                    with gr.Column():
                        decompress_file_input = gr.File(
                            label="Upload Compressed File (.bin)",
                            file_types=[".bin"]
                        )
                        decompress_btn = gr.Button("📂 View Info", variant="primary", size="lg")
                    
                    with gr.Column():
                        decompress_status = gr.Markdown(label="Status")
                        decompress_info = gr.Markdown(label="File Information")
                        decompress_output = gr.File(label="Decompressed Output", visible=False)
                
                decompress_btn.click(
                    fn=decompress_file,
                    inputs=[decompress_file_input],
                    outputs=[decompress_status, decompress_info, decompress_output]
                )
            
            # Tab 4: About & API
            with gr.Tab("ℹ️ About"):
                gr.Markdown(
                    """
                    ## About This Project
                    
                    This is a research project exploring **neural compression** using:
                    - **Large Language Models (LLMs)** for predictive modeling
                    - **Retrieval-Augmented Generation (RAG)** for context-aware compression
                    - **Arithmetic Coding** for entropy encoding
                    
                    ### How It Works
                    
                    1. **Model Prediction**: An LLM predicts probability distributions for each token
                    2. **Arithmetic Coding**: These probabilities are used to compress the sequence
                    3. **RAG Enhancement**: Retrieved context improves prediction accuracy
                    4. **Lossless**: Perfect reconstruction of original data
                    
                    ### Performance Notes
                    
                    - **CPU Mode**: This demo runs on CPU with a small model (DistilGPT2)
                    - **GPU Mode**: Full version supports larger models (Qwen3-0.6B, etc.)
                    - **RAG Mode**: Requires embedding model and document index
                    
                    ### Python API Usage
                    
                    ```python
                    from LLMCompress import compress, decode, Metric
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    
                    # Load model
                    model = AutoModelForCausalLM.from_pretrained("pretrained/Qwen3-0.6B")
                    tokenizer = AutoTokenizer.from_pretrained("pretrained/Qwen3-0.6B")
                    
                    # Compress
                    tokenized = tokenizer(text, return_tensors="pt")
                    logits = model(tokenized["input_ids"]).logits[:, :-1]
                    compressed_bytes, num_padded_bits, start_symbol, _, _, _ = compress(
                        tokenized["input_ids"], logits, Metric()
                    )
                    
                    # Decompress
                    decompressed = decode(compressed_bytes, num_padded_bits, model, 
                                        start_symbol, device, original_length)
                    ```
                    
                    ### Supported Modes
                    
                    - ✅ **Text Compression** (This demo)
                    - ⚡ **RAG-Enhanced Compression** (Python API)
                    - 🖼️ **Image Compression** (bGPT - Python API)
                    - 🎵 **Audio Compression** (bGPT - Python API)
                    
                    ### System Requirements
                    
                    - **Minimum**: CPU, 4GB RAM (This demo)
                    - **Recommended**: GPU with 8GB+ VRAM, 16GB+ RAM
                    - **Optimal**: GPU with 24GB+ VRAM, 32GB+ RAM
                    
                    ### Citation
                    
                    If you use this research in your work, please cite:
                    ```
                    @misc{rag-compressor,
                      title={RAG-Based Lossless Data Compression},
                      author={Your Name},
                      year={2026},
                      note={Research Project}
                    }
                    ```
                    
                    ### License
                    
                    This project is for research purposes.
                    """
                )
        
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
            Built with ❤️ using Gradio | Research Project | 2026
            </div>
            """
        )
    
    return interface


# ==================== Main ====================
if __name__ == "__main__":
    print("="*60)
    print("RAG-Based Lossless Compressor - Web Interface")
    print("="*60)
    print(f"Device: {GradioCompressorConfig.DEVICE}")
    print(f"Model: {GradioCompressorConfig.MODEL_NAME}")
    print("="*60)
    
    # Pre-load model to avoid first-time delay
    print("\nPre-loading model...")
    model_manager = ModelManager()
    model_manager.load_model()
    print("✓ Model ready!\n")
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        share=False,  # Set to True to create public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        show_error=True
    )
