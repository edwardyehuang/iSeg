#!/usr/bin/env python3
"""
Simple test script to verify DCNv2 custom op functionality.
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add current directory to path to import our custom op
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_custom_op_loading():
    """Test if the custom op library can be loaded."""
    print("Testing custom op loading...")
    try:
        lib_path = os.path.join(os.path.dirname(__file__), 'dcn_v2_op.so')
        module = tf.load_op_library(lib_path)
        print("✓ Custom op library loaded successfully!")
        return True
    except Exception as e:
        print(f"✗ Failed to load custom op: {e}")
        return False

def test_dcn_v2_op():
    """Test the DCNv2 custom op."""
    print("\nTesting DCNv2 custom op...")
    try:
        # Import our DCNv2 Python interface - avoid the .so file conflict
        import importlib.util
        import sys
        
        # Load the dcn_v2_layer.py module directly
        spec = importlib.util.spec_from_file_location("dcn_v2_layer_module", 
                                                     os.path.join(os.path.dirname(__file__), "dcn_v2_layer.py"))
        dcn_module = importlib.util.module_from_spec(spec)
        sys.modules["dcn_v2_layer_module"] = dcn_module
        spec.loader.exec_module(dcn_module)
        
        # Check if the module loaded successfully
        if dcn_module._dcn_v2_module is None:
            print("✗ DCNv2 op module not loaded properly")
            return False
            
        DCNv2Optimized = dcn_module.DCNv2Optimized
        
        # Create small test data
        batch_size, height, width = 1, 8, 8
        input_channels, output_channels = 4, 8
        kernel_size = 3
        deformable_groups = 1
        
        # Create test tensors
        input_tensor = tf.random.normal([batch_size, height, width, input_channels])
        offset_channels = 2 * kernel_size * kernel_size * deformable_groups
        offset_tensor = tf.random.normal([batch_size, height, width, offset_channels]) * 0.1
        mask_channels = kernel_size * kernel_size * deformable_groups
        mask_tensor = tf.nn.sigmoid(tf.random.normal([batch_size, height, width, mask_channels]))
        
        # Create DCNv2 layer
        dcn_layer = DCNv2Optimized(
            filters=output_channels,
            kernel_size=kernel_size,
            deformable_groups=deformable_groups,
            use_bias=True,
            use_custom_offset=True
        )
        
        # Forward pass
        output = dcn_layer([input_tensor, offset_tensor, mask_tensor])
        
        print(f"✓ DCNv2 forward pass successful!")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{tf.reduce_min(output):.4f}, {tf.reduce_max(output):.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ DCNv2 op test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("DCNv2 Custom Op Test")
    print("=" * 40)
    
    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    # Test custom op loading
    if not test_custom_op_loading():
        return
    
    # Test DCNv2 functionality
    if test_dcn_v2_op():
        print("\n🎉 All tests passed! Custom DCNv2 op is working correctly.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
