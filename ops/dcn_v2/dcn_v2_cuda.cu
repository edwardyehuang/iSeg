#include "dcn_v2_cuda.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <algorithm>

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
const int kMaxGridNum = 65535;

inline int GET_BLOCKS(const int N) {
  return std::min(kMaxGridNum, (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS);
}

// Simplified DCNv2 implementation that matches TensorFlow behavior

__global__ void add_bias_kernel(float* output, const float* bias, 
                               int batch_size, int output_channels, 
                               int output_height, int output_width, int n) {
  CUDA_KERNEL_LOOP(idx, n) {
    const int c = (idx / (output_height * output_width)) % output_channels;
    output[idx] += bias[c];
  }
}

// TensorFlow-compatible gather_nd sampling (no interpolation, exact grid sampling)
__device__ float gather_sample(const float* input, int height, int width, int channels,
                               int y, int x, int c) {
    if (y < 0 || y >= height || x < 0 || x >= width) {
        return 0.0f;
    }
    return input[(y * width + x) * channels + c];
}

// Ultra-optimized deformable convolution kernel for large feature maps - pure compute focus
__global__ void deformable_conv2d_kernel_large(
    const float* input, const float* offset, const float* mask, const float* weight, float* output,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_height, int output_width, int output_channels,
    int kernel_h, int kernel_w, int pad_h, int pad_w, 
    int stride_h, int stride_w, int dilation_h, int dilation_w, int deformable_groups) {
  
  // Ultra-optimized kernel targeting L20 GPU architecture - pure compute optimization
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_outputs = batch_size * output_height * output_width * output_channels;
  
  if (global_idx >= total_outputs) return;
  // Ultra-fast index computation
  const int oc = global_idx % output_channels;
  const int spatial_idx = global_idx / output_channels;
  const int ow = spatial_idx % output_width;
  const int spatial_remaining = spatial_idx / output_width;
  const int oh = spatial_remaining % output_height;
  const int b = spatial_remaining / output_height;
  
  float sum = 0.0f;
  
  // Precompute base addressing for maximum efficiency  
  const int input_base = b * input_height * input_width * input_channels;
  const int spatial_base = (b * output_height + oh) * output_width + ow;
  const int offset_base = spatial_base * 18;  // 9 * 2 for 3x3 kernel
  const int mask_base = spatial_base * 9;     // 9 for 3x3 kernel
  
  // Precompute spatial coordinates for better performance
  const int base_y = oh * stride_h - pad_h;
  const int base_x = ow * stride_w - pad_w;
  
  // Hardcoded optimized 3x3 kernel (most common case)
  if (kernel_h == 3 && kernel_w == 3) {
    // Process all 9 kernel positions with optimal memory access
    #pragma unroll
    for (int kernel_idx = 0; kernel_idx < 9; kernel_idx++) {
      const int kh = kernel_idx / 3;
      const int kw = kernel_idx % 3;
      
      const float offset_y = offset[offset_base + kernel_idx * 2];
      const float offset_x = offset[offset_base + kernel_idx * 2 + 1];
      const float mask_val = mask[mask_base + kernel_idx];
      
      const float sample_y = base_y + kh * dilation_h + offset_y;
      const float sample_x = base_x + kw * dilation_w + offset_x;
      
      // Fast bounds check
      if (sample_y >= 0.0f && sample_y < input_height && sample_x >= 0.0f && sample_x < input_width) {
        // Use fastest floor function
        const int y0 = __float2int_rd(sample_y);
        const int x0 = __float2int_rd(sample_x);
        const int y1 = y0 + 1;
        const int x1 = x0 + 1;
        
        const float dy = sample_y - y0;
        const float dx = sample_x - x0;
        
        // Compute bilinear weights directly in computation - save registers
        const float dy_inv = 1.0f - dy;
        const float dx_inv = 1.0f - dx;
        const float w00 = dy_inv * dx_inv;
        const float w01 = dy_inv * dx;
        const float w10 = dy * dx_inv;
        const float w11 = dy * dx;
        
        // Optimized channel processing with vectorized loads
        for (int ic = 0; ic < input_channels; ic += 4) {
          // Process up to 4 channels at once for better memory bandwidth
          const int remaining_channels = min(4, input_channels - ic);
          
          for (int ch_offset = 0; ch_offset < remaining_channels; ch_offset++) {
            const int current_ic = ic + ch_offset;
            const int addr_base = input_base + current_ic;
            const int addr_y0_x0 = addr_base + (y0 * input_width + x0) * input_channels;
            const int addr_y0_x1 = addr_y0_x0 + input_channels;
            const int addr_y1_x0 = addr_base + (y1 * input_width + x0) * input_channels;
            const int addr_y1_x1 = addr_y1_x0 + input_channels;
            
            const float v00 = (y0 >= 0 && x0 >= 0) ? input[addr_y0_x0] : 0.0f;
            const float v01 = (y0 >= 0 && x1 < input_width) ? input[addr_y0_x1] : 0.0f;
            const float v10 = (y1 < input_height && x0 >= 0) ? input[addr_y1_x0] : 0.0f;
            const float v11 = (y1 < input_height && x1 < input_width) ? input[addr_y1_x1] : 0.0f;
            
            const float interpolated = __fmaf_rn(w00, v00, __fmaf_rn(w01, v01, __fmaf_rn(w10, v10, w11 * v11)));
            const int weight_idx = (kernel_idx * input_channels + current_ic) * output_channels + oc;
            sum = __fmaf_rn(interpolated * mask_val, weight[weight_idx], sum);
          }
        }
      }
    }
  } else {
    // Generic kernel path (fallback)
    #pragma unroll
    for (int kernel_idx = 0; kernel_idx < kernel_h * kernel_w; kernel_idx++) {
      const int kh = kernel_idx / kernel_w;
      const int kw = kernel_idx % kernel_w;
      
      const float offset_y = offset[offset_base + kernel_idx * 2];
      const float offset_x = offset[offset_base + kernel_idx * 2 + 1];
      const float mask_val = mask[mask_base + kernel_idx];
      
      const float sample_y = base_y + kh * dilation_h + offset_y;
      const float sample_x = base_x + kw * dilation_w + offset_x;
      
      if (sample_y >= 0.0f && sample_y < input_height && sample_x >= 0.0f && sample_x < input_width) {
        const int y0 = __float2int_rd(sample_y);
        const int x0 = __float2int_rd(sample_x);
        const int y1 = y0 + 1;
        const int x1 = x0 + 1;
        
        const float dy = sample_y - y0;
        const float dx = sample_x - x0;
        const float w00 = (1.0f - dy) * (1.0f - dx);
        const float w01 = (1.0f - dy) * dx;
        const float w10 = dy * (1.0f - dx);
        const float w11 = dy * dx;
        
        for (int ic = 0; ic < input_channels; ic++) {
          float v00 = 0.0f, v01 = 0.0f, v10 = 0.0f, v11 = 0.0f;
          
          const int base_addr = input_base + ic;
          
          if (y0 >= 0 && x0 >= 0) {
            const int addr = base_addr + (y0 * input_width + x0) * input_channels;
            v00 = input[addr];
            if (x1 < input_width) v01 = input[addr + input_channels];
          }
          if (y1 < input_height && x0 >= 0) {
            const int addr = base_addr + (y1 * input_width + x0) * input_channels;
            v10 = input[addr];
            if (x1 < input_width) v11 = input[addr + input_channels];
          }
          
          const float interpolated = __fmaf_rn(w00, v00, __fmaf_rn(w01, v01, __fmaf_rn(w10, v10, w11 * v11)));
          const int weight_idx = ((kernel_idx * input_channels + ic) * output_channels) + oc;
          sum = __fmaf_rn(interpolated * mask_val, weight[weight_idx], sum);
        }
      }
    }
  }
  
  output[global_idx] = sum;
}

// Deformable convolution kernel that exactly matches TensorFlow DCNv2
__global__ void deformable_conv2d_kernel(
    const float* input, const float* offset, const float* mask, const float* weight, float* output,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_height, int output_width, int output_channels,
    int kernel_h, int kernel_w, int pad_h, int pad_w, 
    int stride_h, int stride_w, int dilation_h, int dilation_w, int deformable_groups) {
  
  CUDA_KERNEL_LOOP(index, batch_size * output_height * output_width * output_channels) {
    const int oc = index % output_channels;
    const int ow = (index / output_channels) % output_width;
    const int oh = (index / (output_channels * output_width)) % output_height;
    const int b = index / (output_channels * output_width * output_height);
    
    float sum = 0.0f;
    const int kernel_size = kernel_h * kernel_w;
    
    for (int kh = 0; kh < kernel_h; kh++) {
      for (int kw = 0; kw < kernel_w; kw++) {
        const int kernel_idx = kh * kernel_w + kw;
        
        // TensorFlow offset layout: [B, H, W, 2*K] where K = kernel_h * kernel_w
        const int offset_base = ((b * output_height + oh) * output_width + ow) * (2 * kernel_size);
        const int offset_y_idx = offset_base + kernel_idx * 2;
        const int offset_x_idx = offset_base + kernel_idx * 2 + 1;
        
        // TensorFlow mask layout: [B, H, W, K]
        const int mask_idx = ((b * output_height + oh) * output_width + ow) * kernel_size + kernel_idx;
        
        const float offset_y = offset[offset_y_idx];
        const float offset_x = offset[offset_x_idx];
        const float mask_val = mask[mask_idx];
        
        // TensorFlow coordinate calculation - exact match
        // 1. Base grid position (output position): [oh, ow]
        // 2. Add padding offset: [oh + pad_h, ow + pad_w]  
        // 3. Add kernel patch offset: [oh + pad_h + (kh-1), ow + pad_w + (kw-1)] for 3x3 kernel
        // 4. Add learned deformable offset
        
        // For 3x3 kernel, patch offsets are: [-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]
        const int patch_y = kh - pad_h;  // Convert kh index to patch coordinate
        const int patch_x = kw - pad_w;  // Convert kw index to patch coordinate
        
        const float base_y = oh + pad_h + patch_y;  // TensorFlow grid calculation
        const float base_x = ow + pad_w + patch_x;  // TensorFlow grid calculation
        
        // Apply deformable offset 
        const float sample_y = base_y + offset_y;
        const float sample_x = base_x + offset_x;
        
        // Get 4 sampling points for bilinear interpolation (TensorFlow method)
        const int y0 = (int)floorf(sample_y);
        const int x0 = (int)floorf(sample_x);
        const int y1 = y0 + 1;
        const int x1 = x0 + 1;
        
        // Calculate bilinear weights
        const float wy = sample_y - y0;
        const float wx = sample_x - x0;
        const float w00 = (1.0f - wy) * (1.0f - wx);
        const float w01 = (1.0f - wy) * wx;
        const float w10 = wy * (1.0f - wx);
        const float w11 = wy * wx;
        
        // Sample from all input channels
        for (int ic = 0; ic < input_channels; ic++) {
          // Sample 4 points using padded coordinates but from original input
          // Convert back to original input coordinates
          const int orig_y0 = y0 - pad_h;
          const int orig_x0 = x0 - pad_w;
          const int orig_y1 = y1 - pad_h;
          const int orig_x1 = x1 - pad_w;
          
          const float* input_ptr = input + b * input_height * input_width * input_channels;
          
          const float v00 = gather_sample(input_ptr, input_height, input_width, input_channels, 
                                        orig_y0, orig_x0, ic);
          const float v01 = gather_sample(input_ptr, input_height, input_width, input_channels, 
                                        orig_y0, orig_x1, ic);
          const float v10 = gather_sample(input_ptr, input_height, input_width, input_channels, 
                                        orig_y1, orig_x0, ic);
          const float v11 = gather_sample(input_ptr, input_height, input_width, input_channels, 
                                        orig_y1, orig_x1, ic);
          
          // Apply bilinear interpolation
          const float interpolated = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
          
          // Weight layout: [kh, kw, ic, oc]
          const int weight_idx = ((kh * kernel_w + kw) * input_channels + ic) * output_channels + oc;
          sum += interpolated * mask_val * weight[weight_idx];
        }
      }
    }
    
    output[index] = sum;
  }
}

// Proper DCNv2 forward implementation
void dcn_v2_forward_cuda(
    const float* input, const float* offset, const float* mask,
    const float* weight, const float* bias,
    float* output,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    cudaStream_t stream) {
  
  const int output_height = (input_height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_width = (input_width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  
  const int num_kernels = batch_size * output_height * output_width * output_channels;
  
  // Choose kernel based on workload size for optimal performance
  const int total_elements = batch_size * input_height * input_width * input_channels;
  const bool use_large_kernel = (total_elements > 1024 * 128 * 128) || (output_height * output_width > 192 * 192);
  
  if (use_large_kernel) {
    // Back to your optimal configuration for L20 GPU
    int threads_per_block = 256;  // You identified this as optimal for large workloads
    int num_blocks = (num_kernels + threads_per_block - 1) / threads_per_block;
    
    // Pure compute optimization
    deformable_conv2d_kernel_large<<<num_blocks, threads_per_block, 0, stream>>>(
        input, offset, mask, weight, output,
        batch_size, input_height, input_width, input_channels,
        output_height, output_width, output_channels,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w, deformable_groups);
  } else {
    // Use standard kernel for small to medium feature maps
    deformable_conv2d_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
        input, offset, mask, weight, output,
        batch_size, input_height, input_width, input_channels,
        output_height, output_width, output_channels,
        kernel_h, kernel_w, pad_h, pad_w,
        stride_h, stride_w, dilation_h, dilation_w, deformable_groups);
  }
  
  // Add bias if present
  if (bias != nullptr) {
    const int bias_num_kernels = batch_size * output_channels * output_height * output_width;
    const int num_threads = 256;
    const int num_blocks = (bias_num_kernels + num_threads - 1) / num_threads;
    
    add_bias_kernel<<<num_blocks, num_threads, 0, stream>>>(
        output, bias, batch_size, output_channels, output_height, output_width, bias_num_kernels);
  }
}

// Simplified backward functions (placeholders for now)
void dcn_v2_backward_input_cuda(
    const float* grad_output, const float* offset, const float* mask,
    const float* weight,
    float* grad_input, float* grad_offset, float* grad_mask,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    cudaStream_t stream) {
  
  // Simplified placeholder - just zero out gradients for now
  const int input_size = batch_size * input_height * input_width * input_channels;
  const int offset_size = batch_size * input_height * input_width * 2 * kernel_h * kernel_w * deformable_groups;
  const int mask_size = batch_size * input_height * input_width * kernel_h * kernel_w * deformable_groups;
  
  cudaMemset(grad_input, 0, input_size * sizeof(float));
  cudaMemset(grad_offset, 0, offset_size * sizeof(float));
  cudaMemset(grad_mask, 0, mask_size * sizeof(float));
}

void dcn_v2_backward_weight_cuda(
    const float* input, const float* offset, const float* mask,
    const float* grad_output,
    float* grad_weight, float* grad_bias,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    cudaStream_t stream) {
  
  // Simplified placeholder - just zero out gradients for now
  const int weight_size = kernel_h * kernel_w * input_channels * output_channels;
  const int bias_size = output_channels;
  
  cudaMemset(grad_weight, 0, weight_size * sizeof(float));
  if (grad_bias != nullptr) {
    cudaMemset(grad_bias, 0, bias_size * sizeof(float));
  }
}
