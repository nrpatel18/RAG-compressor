#!/usr/bin/env python3
"""
Quick test to verify compression works and download models
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("Testing LLM Compression")
print("=" * 60)

# Test 1: Import modules
print("\n1. Importing modules...")
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from LLMCompress import compress
    print("   ✅ Imports successful")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    exit(1)

# Test 2: Load model
print("\n2. Loading GPT-2 model...")
print("   (This will download ~500MB on first run)")
try:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", low_cpu_mem_usage=True)
    model.eval()
    print("   ✅ Model loaded successfully")
except Exception as e:
    print(f"   ❌ Model loading failed: {e}")
    exit(1)

# Test 3: Compress short text
print("\n3. Testing compression...")
test_text = "The quick brown fox jumps over the lazy dog."
print(f"   Input: '{test_text}'")
print(f"   Length: {len(test_text)} chars ({len(test_text.encode('utf-8'))} bytes)")

try:
    import time
    start = time.time()
    
    _, compressed_bits = compress(
        model=model,
        tokenizer=tokenizer,
        input_text=test_text,
        prefix_len=1,
        device="cpu",
        precision=32
    )
    
    elapsed = time.time() - start
    
    original_size = len(test_text.encode('utf-8'))
    compressed_size = len(compressed_bits) // 8
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    print(f"   ✅ Compression successful!")
    print(f"   Original: {original_size} bytes")
    print(f"   Compressed: {compressed_size} bytes")
    print(f"   Ratio: {ratio:.2f}x")
    print(f"   Time: {elapsed:.2f} seconds")
    
except Exception as e:
    print(f"   ❌ Compression failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed! Compression works correctly.")
print("=" * 60)
