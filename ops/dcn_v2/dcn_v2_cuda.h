#ifndef DCN_V2_CUDA_H_
#define DCN_V2_CUDA_H_

#include <cuda_runtime.h>

// CUDA kernel declarations
void dcn_v2_forward_cuda(
    const float* input, const float* offset, const float* mask,
    const float* weight, const float* bias,
    float* output,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    cudaStream_t stream);

void dcn_v2_backward_input_cuda(
    const float* grad_output, const float* offset, const float* mask,
    const float* weight,
    float* grad_input, float* grad_offset, float* grad_mask,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    cudaStream_t stream);

void dcn_v2_backward_weight_cuda(
    const float* input, const float* offset, const float* mask,
    const float* grad_output,
    float* grad_weight, float* grad_bias,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_channels, int kernel_h, int kernel_w,
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int deformable_groups,
    cudaStream_t stream);

#endif  // DCN_V2_CUDA_H_
