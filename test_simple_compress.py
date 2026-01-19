#!/usr/bin/env python3
"""
Dead simple test - does compression even work?
"""

print("Testing basic compression...")
print("=" * 60)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from LLMCompress import compress
    import torch
    
    # Tiny test
    text = "Hello world! This is a test."
    print(f"Text: {text}")
    print(f"Size: {len(text)} chars\n")
    
    # Load smallest model
    print("Loading distilgpt2...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    model.eval()
    print("✅ Model loaded!\n")
    
    # Compress with minimal settings
    print("Compressing...")
    _, compressed = compress(
        model=model,
        tokenizer=tokenizer,
        input_text=text,
        prefix_len=1,
        device="cpu",
        precision=16  # Fast mode
    )
    
    original_size = len(text.encode('utf-8'))
    compressed_size = len(compressed) // 8
    ratio = original_size / compressed_size
    
    print(f"\n✅ SUCCESS!")
    print(f"Original: {original_size} bytes")
    print(f"Compressed: {compressed_size} bytes")
    print(f"Ratio: {ratio:.2f}x")
    
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
