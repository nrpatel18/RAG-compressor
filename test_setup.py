#!/usr/bin/env python3
"""
Quick test script to verify the RAG compressor setup
"""

import sys
import torch

print("="*60)
print("RAG Compressor - System Check")
print("="*60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("   ❌ Python 3.8+ required")
    sys.exit(1)
else:
    print("   ✅ Python version OK")

# Check PyTorch
try:
    print(f"\n2. PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print("   ✅ PyTorch installed")
except Exception as e:
    print(f"   ❌ PyTorch error: {e}")

# Check core dependencies
dependencies = [
    'transformers',
    'datasets',
    'numpy',
    'tqdm',
    'gradio'
]

print("\n3. Dependencies:")
all_ok = True
for dep in dependencies:
    try:
        __import__(dep)
        print(f"   ✅ {dep}")
    except ImportError:
        print(f"   ❌ {dep} - Not installed")
        all_ok = False

# Check custom modules
print("\n4. Custom Modules:")
custom_modules = [
    'LLMCompress',
    'arithmetic_coder.arithmetic_coder',
    'arithmetic_coder.ac_utils'
]

for module in custom_modules:
    try:
        __import__(module)
        print(f"   ✅ {module}")
    except ImportError as e:
        print(f"   ❌ {module} - {e}")
        all_ok = False

# Test basic compression
print("\n5. Testing Basic Compression:")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from LLMCompress import Metric, compress
    
    print("   Loading model (this may take a moment)...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model.eval()
    
    print("   ✅ Model loaded")
    
    # Test compression
    test_text = "Hello, this is a test of the compression system."
    tokenized = tokenizer(test_text, return_tensors="pt")
    
    with torch.inference_mode():
        logits = model(tokenized["input_ids"], use_cache=False).logits[:, :-1]
    
    metric = Metric()
    compressed_bytes, num_padded_bits, _, _, _, _ = compress(
        tokenized["input_ids"],
        logits,
        metric
    )
    
    original_size = len(test_text.encode('utf-8'))
    compressed_size = len(compressed_bytes)
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    print(f"   ✅ Compression test successful!")
    print(f"      Original: {original_size} bytes")
    print(f"      Compressed: {compressed_size} bytes")
    print(f"      Ratio: {ratio:.2f}x")
    
except Exception as e:
    import traceback
    print(f"   ❌ Compression test failed: {e}")
    print(traceback.format_exc())
    all_ok = False

# Final summary
print("\n" + "="*60)
if all_ok:
    print("✅ All checks passed! System is ready.")
    print("\nYou can now run:")
    print("  python gradio_app.py")
    print("\nor use the Python API directly.")
else:
    print("❌ Some checks failed. Please install missing dependencies:")
    print("  pip install -r requirements.txt")
print("="*60)
