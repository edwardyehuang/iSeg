#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "dcn_v2_cuda.h"

using namespace tensorflow;

REGISTER_OP("DCNv2Forward")
    .Input("input: float32")
    .Input("offset: float32")
    .Input("mask: float32")
    .Input("weight: float32")
    .Input("bias: float32")
    .Output("output: float32")
    .Attr("kernel_h: int")
    .Attr("kernel_w: int")
    .Attr("pad_h: int")
    .Attr("pad_w: int")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Attr("dilation_h: int")
    .Attr("dilation_w: int")
    .Attr("deformable_groups: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape, weight_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 4, &weight_shape));
      
      auto batch_size = c->Dim(input_shape, 0);
      auto input_height = c->Dim(input_shape, 1);
      auto input_width = c->Dim(input_shape, 2);
      auto output_channels = c->Dim(weight_shape, 3);
      
      int kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w;
      TF_RETURN_IF_ERROR(c->GetAttr("kernel_h", &kernel_h));
      TF_RETURN_IF_ERROR(c->GetAttr("kernel_w", &kernel_w));
      TF_RETURN_IF_ERROR(c->GetAttr("pad_h", &pad_h));
      TF_RETURN_IF_ERROR(c->GetAttr("pad_w", &pad_w));
      TF_RETURN_IF_ERROR(c->GetAttr("stride_h", &stride_h));
      TF_RETURN_IF_ERROR(c->GetAttr("stride_w", &stride_w));
      TF_RETURN_IF_ERROR(c->GetAttr("dilation_h", &dilation_h));
      TF_RETURN_IF_ERROR(c->GetAttr("dilation_w", &dilation_w));
      
      auto output_height = c->MakeDim((c->Value(input_height) + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1);
      auto output_width = c->MakeDim((c->Value(input_width) + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1);
      
      c->set_output(0, c->MakeShape({batch_size, output_height, output_width, output_channels}));
      return absl::OkStatus();
    });

REGISTER_OP("DCNv2BackwardInput")
    .Input("grad_output: float32")
    .Input("input: float32")
    .Input("offset: float32")
    .Input("mask: float32")
    .Input("weight: float32")
    .Output("grad_input: float32")
    .Output("grad_offset: float32")
    .Output("grad_mask: float32")
    .Attr("kernel_h: int")
    .Attr("kernel_w: int")
    .Attr("pad_h: int")
    .Attr("pad_w: int")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Attr("dilation_h: int")
    .Attr("dilation_w: int")
    .Attr("deformable_groups: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));  // grad_input has same shape as input
      c->set_output(1, c->input(2));  // grad_offset has same shape as offset
      c->set_output(2, c->input(3));  // grad_mask has same shape as mask
      return absl::OkStatus();
    });

REGISTER_OP("DCNv2BackwardWeight")
    .Input("grad_output: float32")
    .Input("input: float32")
    .Input("offset: float32")
    .Input("mask: float32")
    .Output("grad_weight: float32")
    .Output("grad_bias: float32")
    .Attr("kernel_h: int")
    .Attr("kernel_w: int")
    .Attr("pad_h: int")
    .Attr("pad_w: int")
    .Attr("stride_h: int")
    .Attr("stride_w: int")
    .Attr("dilation_h: int")
    .Attr("dilation_w: int")
    .Attr("deformable_groups: int")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input_shape, grad_output_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 4, &input_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &grad_output_shape));
      
      int kernel_h, kernel_w;
      TF_RETURN_IF_ERROR(c->GetAttr("kernel_h", &kernel_h));
      TF_RETURN_IF_ERROR(c->GetAttr("kernel_w", &kernel_w));
      
      auto input_channels = c->Dim(input_shape, 3);
      auto output_channels = c->Dim(grad_output_shape, 3);
      
      c->set_output(0, c->MakeShape({c->MakeDim(kernel_h), c->MakeDim(kernel_w), input_channels, output_channels}));
      c->set_output(1, c->MakeShape({output_channels}));
      return absl::OkStatus();
    });

class DCNv2ForwardOp : public OpKernel {
 public:
  explicit DCNv2ForwardOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("dilation_h", &dilation_h_));
    OP_REQUIRES_OK(context, context->GetAttr("dilation_w", &dilation_w_));
    OP_REQUIRES_OK(context, context->GetAttr("deformable_groups", &deformable_groups_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& offset = context->input(1);
    const Tensor& mask = context->input(2);
    const Tensor& weight = context->input(3);
    const Tensor& bias = context->input(4);

    const auto input_shape = input.shape();
    const auto weight_shape = weight.shape();
    
    const int batch_size = input_shape.dim_size(0);
    const int input_height = input_shape.dim_size(1);
    const int input_width = input_shape.dim_size(2);
    const int input_channels = input_shape.dim_size(3);
    const int output_channels = weight_shape.dim_size(3);
    
    const int output_height = (input_height + 2 * pad_h_ - (dilation_h_ * (kernel_h_ - 1) + 1)) / stride_h_ + 1;
    const int output_width = (input_width + 2 * pad_w_ - (dilation_w_ * (kernel_w_ - 1) + 1)) / stride_w_ + 1;

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, 
        TensorShape({batch_size, output_height, output_width, output_channels}), &output));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    const float* input_ptr = input.flat<float>().data();
    const float* offset_ptr = offset.flat<float>().data();
    const float* mask_ptr = mask.flat<float>().data();
    const float* weight_ptr = weight.flat<float>().data();
    const float* bias_ptr = bias.flat<float>().data();
    float* output_ptr = output->flat<float>().data();

    auto cuda_stream = stream_executor::gpu::AsGpuStreamValue(stream);
    dcn_v2_forward_cuda(
        input_ptr, offset_ptr, mask_ptr, weight_ptr, bias_ptr, output_ptr,
        batch_size, input_height, input_width, input_channels,
        output_channels, kernel_h_, kernel_w_,
        pad_h_, pad_w_, stride_h_, stride_w_,
        dilation_h_, dilation_w_, deformable_groups_,
        cuda_stream);
  }

 private:
  int kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  int stride_h_, stride_w_;
  int dilation_h_, dilation_w_;
  int deformable_groups_;
};

class DCNv2BackwardInputOp : public OpKernel {
 public:
  explicit DCNv2BackwardInputOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("dilation_h", &dilation_h_));
    OP_REQUIRES_OK(context, context->GetAttr("dilation_w", &dilation_w_));
    OP_REQUIRES_OK(context, context->GetAttr("deformable_groups", &deformable_groups_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_output = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& offset = context->input(2);
    const Tensor& mask = context->input(3);
    const Tensor& weight = context->input(4);

    const auto input_shape = input.shape();
    const int batch_size = input_shape.dim_size(0);
    const int input_height = input_shape.dim_size(1);
    const int input_width = input_shape.dim_size(2);
    const int input_channels = input_shape.dim_size(3);
    const int output_channels = grad_output.shape().dim_size(3);

    Tensor* grad_input = nullptr;
    Tensor* grad_offset = nullptr;
    Tensor* grad_mask = nullptr;
    
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, offset.shape(), &grad_offset));
    OP_REQUIRES_OK(context, context->allocate_output(2, mask.shape(), &grad_mask));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    const float* grad_output_ptr = grad_output.flat<float>().data();
    const float* offset_ptr = offset.flat<float>().data();
    const float* mask_ptr = mask.flat<float>().data();
    const float* weight_ptr = weight.flat<float>().data();
    float* grad_input_ptr = grad_input->flat<float>().data();
    float* grad_offset_ptr = grad_offset->flat<float>().data();
    float* grad_mask_ptr = grad_mask->flat<float>().data();

    auto cuda_stream = stream_executor::gpu::AsGpuStreamValue(stream);
    dcn_v2_backward_input_cuda(
        grad_output_ptr, offset_ptr, mask_ptr, weight_ptr,
        grad_input_ptr, grad_offset_ptr, grad_mask_ptr,
        batch_size, input_height, input_width, input_channels,
        output_channels, kernel_h_, kernel_w_,
        pad_h_, pad_w_, stride_h_, stride_w_,
        dilation_h_, dilation_w_, deformable_groups_,
        cuda_stream);
  }

 private:
  int kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  int stride_h_, stride_w_;
  int dilation_h_, dilation_w_;
  int deformable_groups_;
};

class DCNv2BackwardWeightOp : public OpKernel {
 public:
  explicit DCNv2BackwardWeightOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("kernel_h", &kernel_h_));
    OP_REQUIRES_OK(context, context->GetAttr("kernel_w", &kernel_w_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_h", &pad_h_));
    OP_REQUIRES_OK(context, context->GetAttr("pad_w", &pad_w_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_h", &stride_h_));
    OP_REQUIRES_OK(context, context->GetAttr("stride_w", &stride_w_));
    OP_REQUIRES_OK(context, context->GetAttr("dilation_h", &dilation_h_));
    OP_REQUIRES_OK(context, context->GetAttr("dilation_w", &dilation_w_));
    OP_REQUIRES_OK(context, context->GetAttr("deformable_groups", &deformable_groups_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_output = context->input(0);
    const Tensor& input = context->input(1);
    const Tensor& offset = context->input(2);
    const Tensor& mask = context->input(3);

    const auto input_shape = input.shape();
    const int batch_size = input_shape.dim_size(0);
    const int input_height = input_shape.dim_size(1);
    const int input_width = input_shape.dim_size(2);
    const int input_channels = input_shape.dim_size(3);
    const int output_channels = grad_output.shape().dim_size(3);

    Tensor* grad_weight = nullptr;
    Tensor* grad_bias = nullptr;
    
    OP_REQUIRES_OK(context, context->allocate_output(0, 
        TensorShape({kernel_h_, kernel_w_, input_channels, output_channels}), &grad_weight));
    OP_REQUIRES_OK(context, context->allocate_output(1, 
        TensorShape({output_channels}), &grad_bias));

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    const float* grad_output_ptr = grad_output.flat<float>().data();
    const float* input_ptr = input.flat<float>().data();
    const float* offset_ptr = offset.flat<float>().data();
    const float* mask_ptr = mask.flat<float>().data();
    float* grad_weight_ptr = grad_weight->flat<float>().data();
    float* grad_bias_ptr = grad_bias->flat<float>().data();

    auto cuda_stream = stream_executor::gpu::AsGpuStreamValue(stream);
    dcn_v2_backward_weight_cuda(
        input_ptr, offset_ptr, mask_ptr, grad_output_ptr,
        grad_weight_ptr, grad_bias_ptr,
        batch_size, input_height, input_width, input_channels,
        output_channels, kernel_h_, kernel_w_,
        pad_h_, pad_w_, stride_h_, stride_w_,
        dilation_h_, dilation_w_, deformable_groups_,
        cuda_stream);
  }

 private:
  int kernel_h_, kernel_w_;
  int pad_h_, pad_w_;
  int stride_h_, stride_w_;
  int dilation_h_, dilation_w_;
  int deformable_groups_;
};

REGISTER_KERNEL_BUILDER(Name("DCNv2Forward").Device(DEVICE_GPU), DCNv2ForwardOp);
REGISTER_KERNEL_BUILDER(Name("DCNv2BackwardInput").Device(DEVICE_GPU), DCNv2BackwardInputOp);
REGISTER_KERNEL_BUILDER(Name("DCNv2BackwardWeight").Device(DEVICE_GPU), DCNv2BackwardWeightOp);
