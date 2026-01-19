#!/usr/bin/env python3
"""Simple test to verify model loading works"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

print("Testing model loading...")
print("Step 1: Importing transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Step 2: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
print(f"✅ Tokenizer loaded! Vocab size: {len(tokenizer)}")

print("Step 3: Loading model...")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print(f"✅ Model loaded! Parameters: {model.num_parameters():,}")

print("\n✅ All tests passed! Model loading works correctly.")
