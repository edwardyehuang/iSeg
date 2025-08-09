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

__global__ void add_bias_kernel(float* output, const float* bias, 
                               int batch_size, int output_channels, 
                               int output_height, int output_width, int n) {
  CUDA_KERNEL_LOOP(idx, n) {
    const int c = (idx / (output_height * output_width)) % output_channels;
    output[idx] += bias[c];
  }
}

__global__ void fill_ones_kernel(float* data, int n) {
  CUDA_KERNEL_LOOP(idx, n) {
    data[idx] = 1.0f;
  }
}

__device__ float dmcn_im2col_bilinear(const float* bottom_data, const int data_width,
                                      const int height, const int width, float h, float w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = bottom_data[h_low * data_width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = bottom_data[h_low * data_width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = bottom_data[h_high * data_width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = bottom_data[h_high * data_width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__device__ float dmcn_get_gradient_weight(float argmax_h, float argmax_w,
                                          const int h, const int w, const int height, const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (argmax_h_high - argmax_h) * (argmax_w_high - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (argmax_h_high - argmax_h) * (argmax_w - argmax_w_low);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h - argmax_h_low) * (argmax_w_high - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h - argmax_h_low) * (argmax_w - argmax_w_low);
  return weight;
}

__device__ float dmcn_get_coordinate_weight(float argmax_h, float argmax_w,
                                            const int h, const int w, const int height, const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 || argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  float weight = 0;
  if (h == argmax_h_low && w == argmax_w_low)
    weight = (argmax_h_high - argmax_h) * (argmax_w_high - argmax_w);
  if (h == argmax_h_low && w == argmax_w_high)
    weight = (argmax_h_high - argmax_h) * (argmax_w - argmax_w_low);
  if (h == argmax_h_high && w == argmax_w_low)
    weight = (argmax_h - argmax_h_low) * (argmax_w_high - argmax_w);
  if (h == argmax_h_high && w == argmax_w_high)
    weight = (argmax_h - argmax_h_low) * (argmax_w - argmax_w_low);
  return weight;
}

__global__ void dcn_v2_im2col_gpu_kernel(const int n,
                                          const float* data_im, const float* data_offset, const float* data_mask,
                                          const int height, const int width, const int kernel_h, const int kernel_w,
                                          const int pad_h, const int pad_w,
                                          const int stride_h, const int stride_w,
                                          const int dilation_h, const int dilation_w,
                                          const int channel_per_deformable_group,
                                          const int batch_size, const int num_channels, const int deformable_group,
                                          const int height_col, const int width_col,
                                          float* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int w_col = index % width_col;
    const int h_col = (index / width_col) % height_col;
    const int b_col = (index / width_col / height_col) % batch_size;
    const int c_im = (index / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    float* data_col_ptr = data_col + ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const float* data_im_ptr = data_im + (b_col * num_channels + c_im) * height * width;
    const float* data_offset_ptr = data_offset + (b_col * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float* data_mask_ptr = data_mask + (b_col * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col + w_col;
        const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;

        const float offset_h = data_offset_ptr[data_offset_h_ptr];
        const float offset_w = data_offset_ptr[data_offset_w_ptr];
        const float mask = data_mask_ptr[data_mask_hw_ptr];
        float val = static_cast<float>(0);
        const float h_im = h_in + i * dilation_h + offset_h;
        const float w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val = dmcn_im2col_bilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

__global__ void dcn_v2_col2im_gpu_kernel(const int n,
                                          const float* data_col, const float* data_offset, const float* data_mask,
                                          const int channels, const int height, const int width,
                                          const int kernel_h, const int kernel_w,
                                          const int pad_h, const int pad_w,
                                          const int stride_h, const int stride_w,
                                          const int dilation_h, const int dilation_w,
                                          const int channel_per_deformable_group,
                                          const int batch_size, const int deformable_group,
                                          const int height_col, const int width_col,
                                          float* grad_im) {
  CUDA_KERNEL_LOOP(index, n) {
    const int j = (index / width_col / height_col / batch_size) % kernel_w;
    const int i = (index / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c = index / width_col / height_col / batch_size / kernel_w / kernel_h;

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = index % width_col;
    int h_out = (index / width_col) % height_col;
    int b = (index / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const float* data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float* data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr = ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr = ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr = ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const float offset_h = data_offset_ptr[data_offset_h_ptr];
    const float offset_w = data_offset_ptr[data_offset_w_ptr];
    const float mask = data_mask_ptr[data_mask_hw_ptr];
    const float cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const float cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const float cur_top_grad = data_col[index] * mask;
    const int cur_h = (int)cur_inv_h_data;
    const int cur_w = (int)cur_inv_w_data;
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height &&
            cur_w + dx >= 0 && cur_w + dx < width &&
            abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos = ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          float weight = dmcn_get_gradient_weight(cur_inv_h_data, cur_inv_w_data, cur_h + dy, cur_w + dx, height, width);
          atomicAdd(grad_im + cur_bottom_grad_pos, weight * cur_top_grad);
        }
      }
    }
  }
}

__global__ void dcn_v2_col2im_coord_gpu_kernel(const int n,
                                                const float* data_col, const float* data_im, const float* data_offset, const float* data_mask,
                                                const int channels, const int height, const int width,
                                                const int kernel_h, const int kernel_w,
                                                const int pad_h, const int pad_w,
                                                const int stride_h, const int stride_w,
                                                const int dilation_h, const int dilation_w,
                                                const int channel_per_deformable_group,
                                                const int batch_size, const int offset_channels, const int deformable_group,
                                                const int height_col, const int width_col,
                                                float* grad_offset, float* grad_mask) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0, mval = 0;
    int w = index % width_col;
    int h = (index / width_col) % height_col;
    int c = (index / width_col / height_col) % offset_channels;
    int b = (index / width_col / height_col) / offset_channels;

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const float* data_col_ptr = data_col + deformable_group_index * channel_per_deformable_group * batch_size * width_col * height_col;
    const float* data_im_ptr = data_im + (b * deformable_group + deformable_group_index) * channel_per_deformable_group * height * width;
    const float* data_offset_ptr = data_offset + (b * deformable_group + deformable_group_index) * 2 * kernel_h * kernel_w * height_col * width_col;
    const float* data_mask_ptr = data_mask + (b * deformable_group + deformable_group_index) * kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = (offset_c / 2); col_c < channel_per_deformable_group; col_c += col_step) {
      const int col_pos = (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (offset_c / 2) % kernel_w;
      int i = (offset_c / 2) / kernel_w;
      int w_out = w;
      int h_out = h;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr = (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr = (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out);
      const int data_mask_hw_ptr = (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const float offset_h = data_offset_ptr[data_offset_h_ptr];
      const float offset_w = data_offset_ptr[data_offset_w_ptr];
      const float mask = data_mask_ptr[data_mask_hw_ptr];
      float inv_h = h_in + i * dilation_h + offset_h;
      float inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      }
      
      // Simplified gradient computation
      float weight = 1.0f;
      if (inv_h > -1 && inv_w > -1 && inv_h < height && inv_w < width) {
        weight = dmcn_get_coordinate_weight(inv_h, inv_w, 0, 0, height, width);
      }
      
      val += weight * data_col_ptr[col_pos] * mask;
      mval += weight * data_col_ptr[col_pos];
      cnt += 1;
    }

    grad_offset[index] = val;
    if (offset_c % 2 == 0)
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h * kernel_w + offset_c / 2) * height_col + h) * width_col + w] = mval;
  }
}

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
  
  const int num_kernels = input_channels * batch_size * output_height * output_width;
  const int channel_per_deformable_group = input_channels / deformable_groups;
  
  // Allocate workspace for column buffer
  size_t col_buffer_size = input_channels * kernel_h * kernel_w * batch_size * output_height * output_width * sizeof(float);
  float* col_buffer;
  cudaMalloc(&col_buffer, col_buffer_size);
  
  // im2col with deformable convolution
  dcn_v2_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, input, offset, mask,
      input_height, input_width, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, batch_size, input_channels, deformable_groups,
      output_height, output_width, col_buffer);
  
  // Matrix multiplication (GEMM)
  // output = weight * col_buffer + bias
  const int M = output_channels;
  const int N = batch_size * output_height * output_width;
  const int K = input_channels * kernel_h * kernel_w;
  
  // Use cuBLAS for optimized GEMM
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  
  const float alpha = 1.0f, beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K,
              &alpha,
              col_buffer, N,
              weight, K,
              &beta,
              output, N);
  
  // Add bias if present
  if (bias != nullptr) {
    const int bias_num_kernels = batch_size * output_channels * output_height * output_width;
    const int num_threads = 256;
    const int num_blocks = (bias_num_kernels + num_threads - 1) / num_threads;
    
    add_bias_kernel<<<num_blocks, num_threads, 0, stream>>>(
        output, bias, batch_size, output_channels, output_height, output_width, bias_num_kernels);
  }
  
  cublasDestroy(handle);
  cudaFree(col_buffer);
}

void dcn_v2_backward_input_cuda(
    const float* grad_output, const float* offset, const float* mask,
    const float* weight,
    float* grad_input, float* grad_offset, float* grad_mask,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    cudaStream_t stream) {
  
  const int output_height = (input_height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_width = (input_width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_per_deformable_group = input_channels / deformable_groups;
  
  // Allocate workspace
  size_t col_buffer_size = input_channels * kernel_h * kernel_w * batch_size * output_height * output_width * sizeof(float);
  float* grad_col_buffer;
  cudaMalloc(&grad_col_buffer, col_buffer_size);
  
  // Compute gradient w.r.t. column buffer
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  
  const float alpha = 1.0f, beta = 0.0f;
  const int M = input_channels * kernel_h * kernel_w;
  const int N = batch_size * output_height * output_width;
  const int K = output_channels;
  
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
              N, M, K,
              &alpha,
              grad_output, N,
              weight, M,
              &beta,
              grad_col_buffer, N);
  
  // col2im for grad_input
  const int grad_input_kernels = input_channels * kernel_h * kernel_w * batch_size * output_height * output_width;
  dcn_v2_col2im_gpu_kernel<<<GET_BLOCKS(grad_input_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      grad_input_kernels, grad_col_buffer, offset, mask,
      input_channels, input_height, input_width, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, batch_size, deformable_groups,
      output_height, output_width, grad_input);
  
  // Compute gradients w.r.t. offset and mask
  const int offset_kernels = batch_size * deformable_groups * 2 * kernel_h * kernel_w * output_height * output_width;
  dcn_v2_col2im_coord_gpu_kernel<<<GET_BLOCKS(offset_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      offset_kernels, grad_col_buffer, grad_input, offset, mask,
      input_channels, input_height, input_width, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, batch_size, deformable_groups * 2 * kernel_h * kernel_w, deformable_groups,
      output_height, output_width, grad_offset, grad_mask);
  
  cublasDestroy(handle);
  cudaFree(grad_col_buffer);
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
  
  const int output_height = (input_height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_width = (input_width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_per_deformable_group = input_channels / deformable_groups;
  
  // Allocate workspace
  size_t col_buffer_size = input_channels * kernel_h * kernel_w * batch_size * output_height * output_width * sizeof(float);
  float* col_buffer;
  cudaMalloc(&col_buffer, col_buffer_size);
  
  // Forward im2col for weight gradient computation
  const int num_kernels = input_channels * batch_size * output_height * output_width;
  dcn_v2_im2col_gpu_kernel<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS, 0, stream>>>(
      num_kernels, input, offset, mask,
      input_height, input_width, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
      channel_per_deformable_group, batch_size, input_channels, deformable_groups,
      output_height, output_width, col_buffer);
  
  // Compute gradient w.r.t. weight
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  
  const float alpha = 1.0f, beta = 0.0f;
  const int M = output_channels;
  const int N = input_channels * kernel_h * kernel_w;
  const int K = batch_size * output_height * output_width;
  
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
              N, M, K,
              &alpha,
              col_buffer, K,
              grad_output, K,
              &beta,
              grad_weight, N);
  
  // Compute gradient w.r.t. bias
  if (grad_bias != nullptr) {
    const float alpha_bias = 1.0f, beta_bias = 0.0f;
    const int bias_M = output_channels;
    const int bias_N = 1;
    const int bias_K = batch_size * output_height * output_width;
    
    float* ones_vector;
    cudaMalloc(&ones_vector, bias_K * sizeof(float));
    
    // Fill ones_vector with 1.0f values
    const int ones_threads = 256;
    const int ones_blocks = (bias_K + ones_threads - 1) / ones_threads;
    fill_ones_kernel<<<ones_blocks, ones_threads, 0, stream>>>(ones_vector, bias_K);
    
    cublasSgemv(handle, CUBLAS_OP_N,
                bias_M, bias_K,
                &alpha_bias,
                grad_output, bias_M,
                ones_vector, 1,
                &beta_bias,
                grad_bias, 1);
    
    cudaFree(ones_vector);
  }
  
  cublasDestroy(handle);
  cudaFree(col_buffer);
}
