#!/usr/bin/env python3
"""
Benchmark script to compare DCNv2 custom op with original TensorFlow implementation.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Add current directory to path to import our custom op
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the custom DCNv2 op
try:
    from dcn_v2_layer import DCNv2Optimized
    CUSTOM_OP_AVAILABLE = True
    print("✓ Custom DCNv2 op loaded successfully!")
except ImportError as e:
    print(f"✗ Failed to import custom DCNv2 op: {e}")
    CUSTOM_OP_AVAILABLE = False

# Import the original implementation
sys.path.insert(0, '/home/edwardyehuang/research/segmenty')
try:
    from iseg.layers.dcn_v2 import DCNv2
    ORIGINAL_AVAILABLE = True
    print("✓ Original DCNv2 implementation loaded successfully!")
except ImportError as e:
    print(f"✗ Failed to import original DCNv2: {e}")
    ORIGINAL_AVAILABLE = False

def create_test_data(batch_size, height, width, input_channels, output_channels, kernel_size=3, deformable_groups=1):
    """Create test data for DCNv2 benchmark."""
    # Input tensor
    input_tensor = tf.random.normal([batch_size, height, width, input_channels], dtype=tf.float32)
    
    # Offset tensor (2 * kernel_size^2 * deformable_groups channels)
    offset_channels = 2 * kernel_size * kernel_size * deformable_groups
    offset_tensor = tf.random.normal([batch_size, height, width, offset_channels], dtype=tf.float32) * 0.1
    
    # Mask tensor (kernel_size^2 * deformable_groups channels)
    mask_channels = kernel_size * kernel_size * deformable_groups
    mask_tensor = tf.nn.sigmoid(tf.random.normal([batch_size, height, width, mask_channels], dtype=tf.float32))
    
    return input_tensor, offset_tensor, mask_tensor

def benchmark_layer(layer, input_data, num_runs=10, warmup_runs=2):
    """Benchmark a DCNv2 layer implementation."""
    times = []
    
    # Warmup runs
    for _ in range(warmup_runs):
        _ = layer(input_data)
    
    # Benchmark runs
    for i in range(num_runs):
        start_time = time.time()
        
        output = layer(input_data)
        
        # Ensure computation is completed
        tf.reduce_sum(output).numpy()
        
        end_time = time.time()
        times.append(end_time - start_time)
        
        if (i + 1) % 5 == 0:
            print(f"  Run {i+1}/{num_runs} completed")
    
    return times

def main():
    print("DCNv2 Benchmark")
    print("=" * 50)
    
    if not CUSTOM_OP_AVAILABLE or not ORIGINAL_AVAILABLE:
        print("Cannot run benchmark - missing implementations")
        return
    
    # Test configurations
    configs = [
        {"name": "Small", "batch_size": 2, "height": 32, "width": 32, "input_channels": 64, "output_channels": 64},
        {"name": "Medium", "batch_size": 4, "height": 64, "width": 64, "input_channels": 128, "output_channels": 128},
        {"name": "Large", "batch_size": 2, "height": 128, "width": 128, "input_channels": 256, "output_channels": 256},
    ]
    
    kernel_size = 3
    deformable_groups = 1  # Original implementation only supports 1 deformable group
    num_benchmark_runs = 10
    
    print(f"\nNote: Three implementations will be tested with IDENTICAL WEIGHTS for fair comparison")
    print(f"This benchmark compares performance of algorithmically-equivalent DCNv2 implementations:")
    print(f"  - Original: Pure TensorFlow implementation (DCNv2 with use_custom_op=False)")
    print(f"  - Custom Op: DCNv2 with use_custom_op=True (uses GPU-optimized CUDA implementation)")
    print(f"  - Direct Custom: Direct DCNv2Optimized layer")
    print(f"All implementations will be tested with deformable_groups={deformable_groups}")
    
    for config in configs:
        print(f"\n--- {config['name']} Configuration ---")
        print(f"Batch size: {config['batch_size']}")
        print(f"Input shape: [{config['batch_size']}, {config['height']}, {config['width']}, {config['input_channels']}]")
        print(f"Output channels: {config['output_channels']}")
        print(f"Kernel size: {kernel_size}")
        print(f"Deformable groups: {deformable_groups}")
        
        # Create test data
        input_data, offset_data, mask_data = create_test_data(
            config['batch_size'], config['height'], config['width'], 
            config['input_channels'], config['output_channels'], 
            kernel_size, deformable_groups
        )
        
        # Create layers - use automatic offset/mask generation for fair comparison
        kernel_size_tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        
        # Original implementation
        original_layer = DCNv2(
            filters=config['output_channels'],
            kernel_size=kernel_size_tuple,
            use_bias=True,
            use_custom_offset=False,  # Use automatic offset/mask generation
            use_custom_op=False       # Use pure TensorFlow implementation
        )
        
        # DCNv2 with custom op enabled
        custom_op_layer = DCNv2(
            filters=config['output_channels'],
            kernel_size=kernel_size_tuple,
            use_bias=True,
            use_custom_offset=False,  # Use automatic offset/mask generation
            use_custom_op=True        # Use optimized custom op
        )
        
        # Direct custom implementation
        direct_custom_layer = DCNv2Optimized(
            filters=config['output_channels'],
            kernel_size=kernel_size,
            deformable_groups=deformable_groups,
            use_bias=True,
            use_custom_offset=False  # Use automatic offset/mask generation for fair comparison
        )
        
        # Initialize layers by calling them once with just input data
        _ = original_layer(input_data)
        _ = custom_op_layer(input_data)
        _ = direct_custom_layer(input_data)
        
        # Copy weights from original to other layers for fair comparison
        print("🔄 Copying weights from original to other layers for fair comparison...")
        
        def copy_weights(source_layer, target_layer, layer_name):
            weight_mapping = [
                ('kernel', 'kernel'),
                ('bias', 'bias'), 
                ('offset_kernel', 'offset_kernel'),
                ('offset_bias', 'offset_bias')
            ]
            
            for orig_name, target_name in weight_mapping:
                source_weight = None
                target_weight = None
                
                # Find source weight
                for w in source_layer.weights:
                    if orig_name in w.name:
                        source_weight = w
                        break
                        
                # For custom_op_layer, weights are in the _custom_layer
                weights_to_search = target_layer._custom_layer.weights if hasattr(target_layer, '_custom_layer') else target_layer.weights
                
                # Find target weight  
                for w in weights_to_search:
                    if target_name in w.name:
                        target_weight = w
                        break
                        
                if source_weight is not None and target_weight is not None:
                    target_weight.assign(source_weight)
        
        copy_weights(original_layer, custom_op_layer, "custom_op")
        copy_weights(original_layer, direct_custom_layer, "direct_custom")
        
        print("✅ Identical weights applied!")
        
        # Test correctness with identical weights
        print("\nTesting correctness between all implementations...")
        
        original_output = original_layer(input_data)
        custom_op_output = custom_op_layer(input_data)
        direct_custom_output = direct_custom_layer(input_data)
        
        # Test original vs custom_op
        correctness_custom_op = test_correctness_outputs(original_output, custom_op_output, "Original vs Custom Op", rtol=1e-4, atol=1e-4)
        
        # Test original vs direct custom
        correctness_direct = test_correctness_outputs(original_output, direct_custom_output, "Original vs Direct Custom", rtol=1e-4, atol=1e-4)
        
        # Test custom_op vs direct custom
        correctness_custom_vs_direct = test_correctness_outputs(custom_op_output, direct_custom_output, "Custom Op vs Direct Custom", rtol=1e-4, atol=1e-4)
        
        if not (correctness_custom_op and correctness_direct and correctness_custom_vs_direct):
            print("❌ UNEXPECTED: Outputs differ despite identical weights - this indicates a bug!")
            return False
        else:
            print("✅ All implementations produce identical outputs!")
        
        # Benchmark all implementations
        print(f"\nBenchmarking Original DCNv2 (TensorFlow)...")
        original_times = benchmark_layer(original_layer, input_data, num_benchmark_runs)
        original_mean = np.mean(original_times)
        original_std = np.std(original_times)
        
        print(f"Original DCNv2: {original_mean:.4f} ± {original_std:.4f} seconds")
        
        print(f"\nBenchmarking DCNv2 with Custom Op...")
        custom_op_times = benchmark_layer(custom_op_layer, input_data, num_benchmark_runs)
        custom_op_mean = np.mean(custom_op_times)
        custom_op_std = np.std(custom_op_times)
        
        print(f"DCNv2 w/ Custom Op: {custom_op_mean:.4f} ± {custom_op_std:.4f} seconds")
        
        print(f"\nBenchmarking Direct Custom DCNv2...")
        direct_custom_times = benchmark_layer(direct_custom_layer, input_data, num_benchmark_runs)
        direct_custom_mean = np.mean(direct_custom_times)
        direct_custom_std = np.std(direct_custom_times)
        
        print(f"Direct Custom DCNv2: {direct_custom_mean:.4f} ± {direct_custom_std:.4f} seconds")
        
        # Calculate speedups
        speedup_custom_op = original_mean / custom_op_mean
        speedup_direct = original_mean / direct_custom_mean
        
        print(f"\n🚀 Speedup Results:")
        print(f"  Custom Op vs Original: {speedup_custom_op:.2f}x")
        print(f"  Direct Custom vs Original: {speedup_direct:.2f}x")
        
        if speedup_custom_op > 1.0:
            print(f"✓ Custom Op implementation is {speedup_custom_op:.2f}x faster!")
        else:
            print(f"⚠️  Custom Op implementation is {1/speedup_custom_op:.2f}x slower")
            
        if speedup_direct > 1.0:
            print(f"✓ Direct Custom implementation is {speedup_direct:.2f}x faster!")
        else:
            print(f"⚠️  Direct Custom implementation is {1/speedup_direct:.2f}x slower")
        
        print("-" * 50)


def test_correctness_outputs(output1, output2, comparison_name, rtol=1e-3, atol=1e-3):
    """Test correctness by comparing two outputs."""
    print(f"\nTesting {comparison_name}...")
    
    # Compare shapes
    if output1.shape != output2.shape:
        print(f"✗ Shape mismatch: {output1.shape} vs {output2.shape}")
        return False
    
    # Compare values
    try:
        np.testing.assert_allclose(output1.numpy(), output2.numpy(), 
                                 rtol=rtol, atol=atol)
        print(f"✓ Outputs match within tolerance (rtol={rtol}, atol={atol})")
        
        # Print some statistics
        diff = np.abs(output1.numpy() - output2.numpy())
        print(f"  Max absolute difference: {np.max(diff):.6f}")
        print(f"  Mean absolute difference: {np.mean(diff):.6f}")
        print(f"  Relative difference: {np.max(diff) / (np.max(np.abs(output1.numpy())) + 1e-8):.6f}")
        
        return True
    except AssertionError as e:
        print(f"✗ Outputs don't match: {e}")
        return False

if __name__ == "__main__":
    # Set up GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    
    main()
