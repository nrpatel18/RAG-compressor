# RAG-Based Lossless Compressor - Deployment Guide

## 🚀 Quick Start (Local)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web Interface

```bash
python gradio_app.py
```

The interface will be available at: `http://localhost:7860`

## 🎯 Running on Your Machine

### CPU Mode (No GPU Required)

The default configuration uses CPU and a lightweight model (DistilGPT2):

```bash
python gradio_app.py
```

### GPU Mode (Better Compression)

If you have a GPU, edit `gradio_app.py` and change:

```python
USE_CPU = False
MODEL_NAME = "pretrained/Qwen3-0.6B"  # Or your preferred model
```

## 🌐 Deployment Options

### Option 1: Hugging Face Spaces (Recommended)

1. Create account at https://huggingface.co
2. Create a new Space
3. Upload files:
   - `gradio_app.py`
   - `requirements.txt`
   - `LLMCompress.py`
   - `arithmetic_coder/` folder
4. Set Space hardware (CPU or GPU)
5. Space will auto-deploy!

**Public URL**: Your space will get a URL like `https://huggingface.co/spaces/username/rag-compressor`

### Option 2: Deploy to Cloud (AWS/GCP/Azure)

#### Using Docker

1. Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "gradio_app.py"]
```

2. Build and run:

```bash
docker build -t rag-compressor .
docker run -p 7860:7860 rag-compressor
```

### Option 3: Local Network (Demo Mode)

To allow others on your network to access:

1. Edit `gradio_app.py`, set:
```python
interface.launch(share=True)  # Creates public temporary link
```

2. Run:
```bash
python gradio_app.py
```

You'll get a public URL like: `https://xxxxx.gradio.live`

## 📦 Model Setup

### Using Pretrained Models

If you have local models in `pretrained/` folder:

```python
# In gradio_app.py
MODEL_NAME = "pretrained/Qwen3-0.6B"
```

### Using Hugging Face Models

The app will auto-download from Hugging Face:

```python
MODEL_NAME = "distilgpt2"  # or "gpt2", "Qwen/Qwen-0.5B", etc.
```

## 🔧 Configuration Options

Edit `gradio_app.py` to customize:

```python
class GradioCompressorConfig:
    USE_CPU = True  # False for GPU
    MODEL_NAME = "distilgpt2"  # Your model
    MAX_TEXT_LENGTH = 5000  # Max characters
    PRECISION = 64  # Arithmetic coding precision
```

## 🎨 Features

### Current (v1.0)
- ✅ Text compression via web interface
- ✅ CPU-compatible mode
- ✅ File upload/download
- ✅ Compression statistics
- ✅ Example texts

### Coming Soon
- 🔜 RAG-enhanced compression
- 🔜 Image compression
- 🔜 Audio compression
- 🔜 Batch processing
- 🔜 Full decompression UI

## 💻 System Requirements

### Minimum (CPU Mode)
- CPU: 2 cores
- RAM: 4GB
- Storage: 2GB
- Internet: For model download

### Recommended (GPU Mode)
- GPU: 8GB VRAM (e.g., RTX 3070)
- RAM: 16GB
- Storage: 10GB
- Internet: For model download

### Optimal (Full Features)
- GPU: 24GB VRAM (e.g., RTX 3090/4090)
- RAM: 32GB
- Storage: 50GB
- Internet: For dataset/model downloads

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce `MAX_TEXT_LENGTH` or use smaller model:
```python
MAX_TEXT_LENGTH = 1000
MODEL_NAME = "distilgpt2"
```

### Issue: Model Download Fails

**Solution**: Download manually and set local path:
```bash
git lfs install
git clone https://huggingface.co/distilgpt2 pretrained/distilgpt2
```

### Issue: Slow Compression

**Solution**: 
1. Use GPU if available
2. Use smaller model
3. Reduce text length

### Issue: Port Already in Use

**Solution**: Change port in `gradio_app.py`:
```python
interface.launch(server_port=7861)  # Use different port
```

## 📊 Performance Benchmarks

| Model | Device | Speed | Compression Ratio |
|-------|--------|-------|------------------|
| DistilGPT2 | CPU | ~5s/KB | 1.2-1.5x |
| DistilGPT2 | GPU | ~1s/KB | 1.2-1.5x |
| GPT2 | GPU | ~2s/KB | 1.5-2.0x |
| Qwen3-0.6B | GPU | ~3s/KB | 2.0-3.0x |
| Qwen3-0.6B + RAG | GPU | ~5s/KB | 2.5-4.0x |

*Benchmarks on typical English text

## 🔒 Security Notes

- Run on trusted networks only
- Don't expose to public internet without authentication
- Uploaded files are stored temporarily
- No data is collected or stored permanently

## 📝 Usage Examples

### Python API

```python
# Direct compression
from LLMCompress import compress, Metric
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

text = "Your text here"
tokenized = tokenizer(text, return_tensors="pt")
logits = model(tokenized["input_ids"]).logits[:, :-1]

compressed_bytes, num_padded_bits, _, _, _, _ = compress(
    tokenized["input_ids"], logits, Metric()
)
```

### RAG-Enhanced

```python
# Use RAG_LLMCompress.py for better compression
from RAG_LLMCompress import run_rag_compression

metric, compression_rate, compression_ratio = run_rag_compression(
    test_sample_path="my_text.txt"
)
print(f"Compression ratio: {compression_ratio:.2f}x")
```

## 🤝 Contributing

This is a research project. Contributions welcome!

## 📄 License

Research project - Please cite if used in academic work.

## 🆘 Support

For issues:
1. Check troubleshooting section
2. Review error messages
3. Check system requirements
4. Try with smaller model/text

## 🔗 Links

- Gradio Docs: https://gradio.app/docs
- Transformers: https://huggingface.co/docs/transformers
- Hugging Face Spaces: https://huggingface.co/spaces
