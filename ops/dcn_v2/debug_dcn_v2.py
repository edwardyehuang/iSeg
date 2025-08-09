#!/usr/bin/env python3
"""
Debug script to understand the differences between original and custom DCNv2.
"""

import os
import sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, '/home/edwardyehuang/research/segmenty')

# Import both implementations
from dcn_v2_layer import DCNv2Optimized
from iseg.layers.dcn_v2 import DCNv2

def compare_implementations():
    print("🔍 Detailed DCNv2 Implementation Comparison")
    print("=" * 60)
    
    # Create very small test case for detailed analysis
    batch_size, height, width, input_channels = 1, 4, 4, 2
    output_channels = 2
    kernel_size = 3
    
    # Create deterministic test data
    np.random.seed(42)
    tf.random.set_seed(42)
    
    input_data = tf.constant(np.random.randn(batch_size, height, width, input_channels).astype(np.float32))
    
    print(f"Input shape: {input_data.shape}")
    print(f"Input data sample:\n{input_data[0, :, :, 0].numpy()}")
    
    # Create both layers with same initialization
    original_layer = DCNv2(
        filters=output_channels,
        kernel_size=(kernel_size, kernel_size),
        use_bias=True,
        use_custom_offset=False,
        kernel_initializer='ones',  # Use deterministic initialization
        bias_initializer='zeros',
        name='original'
    )
    
    custom_layer = DCNv2Optimized(
        filters=output_channels,
        kernel_size=kernel_size,
        deformable_groups=1,
        use_bias=True,
        use_custom_offset=False,
        kernel_initializer='ones',  # Use deterministic initialization
        bias_initializer='zeros',
        name='custom'
    )
    
    # Initialize layers
    original_output = original_layer(input_data)
    custom_output = custom_layer(input_data)
    
    print(f"\nOriginal output shape: {original_output.shape}")
    print(f"Custom output shape: {custom_output.shape}")
    
    print(f"\nOriginal output sample:\n{original_output[0, :, :, 0].numpy()}")
    print(f"\nCustom output sample:\n{custom_output[0, :, :, 0].numpy()}")
    
    # Check offset and mask generation
    print("\n🔍 Analyzing offset and mask generation...")
    
    # Get the offset and mask from original implementation
    # We need to manually generate them to compare
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        original_output = original_layer(input_data)
    
    # Extract learned weights
    print("\n📊 Layer weights comparison:")
    print("Original layer weights:")
    for i, weight in enumerate(original_layer.weights):
        print(f"  Weight {i}: {weight.name} - shape: {weight.shape}")
        if 'offset' in weight.name and weight.shape.as_list()[-1] <= 10:  # Print small weights
            print(f"    Sample values: {weight.numpy().flatten()[:10]}")
    
    print("\nCustom layer weights:")
    for i, weight in enumerate(custom_layer.weights):
        print(f"  Weight {i}: {weight.name} - shape: {weight.shape}")
        if hasattr(weight, 'numpy') and weight.shape.as_list()[-1] <= 10:  # Print small weights
            print(f"    Sample values: {weight.numpy().flatten()[:10]}")
    
    # Calculate differences
    diff = tf.abs(original_output - custom_output)
    max_diff = tf.reduce_max(diff)
    mean_diff = tf.reduce_mean(diff)
    
    print(f"\n📈 Output differences:")
    print(f"  Max absolute difference: {max_diff.numpy():.6f}")
    print(f"  Mean absolute difference: {mean_diff.numpy():.6f}")
    print(f"  Original output range: [{tf.reduce_min(original_output).numpy():.6f}, {tf.reduce_max(original_output).numpy():.6f}]")
    print(f"  Custom output range: [{tf.reduce_min(custom_output).numpy():.6f}, {tf.reduce_max(custom_output).numpy():.6f}]")

if __name__ == "__main__":
    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    compare_implementations()
