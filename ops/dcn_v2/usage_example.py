#!/usr/bin/env python3
"""
Usage example for DCNv2 custom TensorFlow op.

This script demonstrates two ways to use the DCNv2Optimized layer:
1. With automatically generated offset and mask (use_custom_offset=False)
2. With custom provided offset and mask (use_custom_offset=True)
"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our DCNv2 layer
from dcn_v2_layer import DCNv2Optimized

def example_automatic_offset_mask():
    """Example using DCNv2 with automatically generated offset and mask."""
    print("\n=== Example 1: Automatic Offset/Mask Generation ===")
    
    # Create input
    batch_size, height, width, input_channels = 1, 16, 16, 32
    output_channels = 64
    kernel_size = 3
    
    input_tensor = tf.random.normal([batch_size, height, width, input_channels])
    
    # Create DCNv2 layer with automatic offset/mask generation
    dcn_layer = DCNv2Optimized(
        filters=output_channels,
        kernel_size=kernel_size,
        deformable_groups=1,
        use_bias=True,
        use_custom_offset=False  # Automatic generation
    )
    
    # Forward pass - just pass the input tensor
    output = dcn_layer(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{tf.reduce_min(output):.4f}, {tf.reduce_max(output):.4f}]")
    print("✓ Automatic offset/mask generation working!")

def example_custom_offset_mask():
    """Example using DCNv2 with custom provided offset and mask."""
    print("\n=== Example 2: Custom Offset/Mask ===")
    
    # Create input
    batch_size, height, width, input_channels = 1, 16, 16, 32
    output_channels = 64
    kernel_size = 3
    deformable_groups = 1
    
    input_tensor = tf.random.normal([batch_size, height, width, input_channels])
    
    # Create custom offset and mask
    offset_channels = 2 * kernel_size * kernel_size * deformable_groups
    offset_tensor = tf.random.normal([batch_size, height, width, offset_channels]) * 0.1
    
    mask_channels = kernel_size * kernel_size * deformable_groups
    mask_tensor = tf.nn.sigmoid(tf.random.normal([batch_size, height, width, mask_channels]))
    
    # Create DCNv2 layer with custom offset/mask
    dcn_layer = DCNv2Optimized(
        filters=output_channels,
        kernel_size=kernel_size,
        deformable_groups=deformable_groups,
        use_bias=True,
        use_custom_offset=True  # Use custom offset/mask
    )
    
    # Forward pass - pass [input, offset, mask] as a list
    output = dcn_layer([input_tensor, offset_tensor, mask_tensor])
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Offset shape: {offset_tensor.shape}")
    print(f"Mask shape: {mask_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{tf.reduce_min(output):.4f}, {tf.reduce_max(output):.4f}]")
    print("✓ Custom offset/mask working!")

def example_in_model():
    """Example of using DCNv2 in a Keras model."""
    print("\n=== Example 3: DCNv2 in Keras Model ===")
    
    # Define a simple model using DCNv2
    input_shape = (32, 32, 64)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        DCNv2Optimized(
            filters=128,
            kernel_size=3,
            deformable_groups=1,
            activation='relu',
            use_bias=True
        ),
        tf.keras.layers.BatchNormalization(),
        DCNv2Optimized(
            filters=256,
            kernel_size=3,
            deformable_groups=2,
            activation='relu',
            use_bias=True
        ),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Build the model
    model.build(input_shape=(None,) + input_shape)
    
    # Print model summary
    print("Model Summary:")
    model.summary()
    
    # Test forward pass
    test_input = tf.random.normal([2] + list(input_shape))
    output = model(test_input)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Model output shape: {output.shape}")
    print("✓ DCNv2 in Keras model working!")

def main():
    """Main function to run all examples."""
    print("DCNv2 Custom Op Usage Examples")
    print("=" * 50)
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    try:
        # Run examples
        example_automatic_offset_mask()
        example_custom_offset_mask()
        example_in_model()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
