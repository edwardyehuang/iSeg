#!/usr/bin/env python3
"""
Test script to compare implementations with identical weights.
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

def test_with_identical_weights():
    print("🔧 Testing DCNv2 with Identical Weights")
    print("=" * 50)
    
    # Create very small test case
    batch_size, height, width, input_channels = 1, 4, 4, 2
    output_channels = 2
    kernel_size = 3
    
    # Create deterministic test data
    np.random.seed(42)
    tf.random.set_seed(42)
    
    input_data = tf.constant(np.random.randn(batch_size, height, width, input_channels).astype(np.float32))
    
    print(f"Input shape: {input_data.shape}")
    
    # Create original layer first  
    original_layer = DCNv2(
        filters=output_channels,
        kernel_size=(kernel_size, kernel_size),
        use_bias=True,
        use_custom_offset=False,
        name='original'
    )
    
    # Initialize with dummy forward pass
    _ = original_layer(input_data)
    
    print("Original layer initialized!")
    
    # Create custom layer
    custom_layer = DCNv2Optimized(
        filters=output_channels,
        kernel_size=kernel_size,
        deformable_groups=1,
        use_bias=True,
        use_custom_offset=False,
        name='custom'
    )
    
    # Initialize with dummy forward pass
    _ = custom_layer(input_data)
    
    print("Custom layer initialized!")
    
    # Copy weights from original to custom
    print("\n🔄 Copying weights from original to custom layer...")
    
    # Map corresponding weights
    weight_mapping = [
        ('kernel', 'kernel'),
        ('bias', 'bias'), 
        ('offset_kernel', 'offset_kernel'),
        ('offset_bias', 'offset_bias')
    ]
    
    for orig_name, custom_name in weight_mapping:
        orig_weight = None
        custom_weight = None
        
        # Find original weight
        for w in original_layer.weights:
            if orig_name in w.name:
                orig_weight = w
                break
                
        # Find custom weight  
        for w in custom_layer.weights:
            if custom_name in w.name:
                custom_weight = w
                break
                
        if orig_weight is not None and custom_weight is not None:
            print(f"  Copying {orig_name}: {orig_weight.shape} -> {custom_weight.shape}")
            custom_weight.assign(orig_weight)
        else:
            print(f"  ⚠️  Could not find matching weights for {orig_name}")
    
    print("\n✅ Weight copying complete!")
    
    # Now test with identical weights
    print("\n🧪 Testing with identical weights...")
    
    original_output = original_layer(input_data)
    custom_output = custom_layer(input_data)
    
    print(f"\nOriginal output:\n{original_output[0, :, :, 0].numpy()}")
    print(f"\nCustom output:\n{custom_output[0, :, :, 0].numpy()}")
    
    # Calculate differences
    diff = tf.abs(original_output - custom_output)
    max_diff = tf.reduce_max(diff)
    mean_diff = tf.reduce_mean(diff)
    
    # Calculate mismatch percentage
    tolerance = 1e-4
    mismatched = tf.reduce_sum(tf.cast(diff > tolerance, tf.float32))
    total_elements = tf.cast(tf.size(diff), tf.float32)
    mismatch_percentage = (mismatched / total_elements) * 100.0
    
    print(f"\n📊 Results with identical weights:")
    print(f"  Max absolute difference: {max_diff.numpy():.8f}")
    print(f"  Mean absolute difference: {mean_diff.numpy():.8f}")
    print(f"  Mismatched elements (>{tolerance}): {mismatched.numpy():.0f} / {total_elements.numpy():.0f} ({mismatch_percentage.numpy():.2f}%)")
    
    if mismatch_percentage.numpy() < 1.0:
        print("✅ SUCCESS: Implementations match with identical weights!")
    else:
        print("❌ FAILURE: Still significant differences with identical weights")
        print("   This indicates algorithmic differences in the implementation")

if __name__ == "__main__":
    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    test_with_identical_weights()
