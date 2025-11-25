"""
Test script to verify the cache system is working correctly.
"""

import sys
import os
sys.path.append('.')

from t2ebm.cache import get_graph_cache, clear_graph_cache
from t2ebm.graphs import extract_graph, graph_to_text
from interpret.glassbox import ExplainableBoostingClassifier
import numpy as np
from sklearn.model_selection import train_test_split

# Create a simple test dataset
np.random.seed(42)
X = np.random.randn(100, 3)
y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(100) * 0.1 > 0).astype(int)

# Train a simple EBM
ebm = ExplainableBoostingClassifier()
ebm.fit(X, y)

# Clear any existing cache
clear_graph_cache()
print("Cache cleared")

# Test cache functionality
print("\n=== Testing Cache System ===")

# First call - should process and cache
print("\n1. First call (should process and cache):")
graph = extract_graph(ebm, 0)
text1 = graph_to_text(graph, ebm=ebm, feature_index=0)
print(f"First call completed. Text length: {len(text1)}")

# Second call - should use cache
print("\n2. Second call (should use cache):")
text2 = graph_to_text(graph, ebm=ebm, feature_index=0)
print(f"Second call completed. Text length: {len(text2)}")

# Verify they are the same
print(f"\nTexts are identical: {text1 == text2}")

# Test with different feature
print("\n3. Different feature (should process and cache):")
graph_diff = extract_graph(ebm, 1)
text3 = graph_to_text(graph_diff, ebm=ebm, feature_index=1)
print(f"Different feature completed. Text length: {len(text3)}")

# Check cache directory
cache_dir = ".cache"
if os.path.exists(cache_dir):
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    print(f"\nCache files created: {len(cache_files)}")
    for file in cache_files:
        print(f"  - {file}")
else:
    print("\nNo cache directory found")

print("\n=== Cache Test Complete ===")
