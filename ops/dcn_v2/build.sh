#!/bin/bash

# Build script for DCNv2 custom TensorFlow op
set -e

echo "Building DCNv2 custom TensorFlow op..."

# Get current directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Activate conda environment and get TensorFlow info
echo "Getting TensorFlow compilation flags..."
TF_CFLAGS=$(conda run --live-stream --name tf215 python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(conda run --live-stream --name tf215 python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

echo "TF_CFLAGS: $TF_CFLAGS"
echo "TF_LFLAGS: $TF_LFLAGS"

# Get CUDA paths from conda environment
CUDA_ROOT=$(conda run --live-stream --name tf215 python -c "
import os
import nvidia.cuda_nvcc
cuda_nvcc_path = nvidia.cuda_nvcc.__file__
cuda_root = os.path.dirname(os.path.dirname(cuda_nvcc_path))
print(cuda_root)
")

echo "CUDA_ROOT: $CUDA_ROOT"

# CUDA compiler flags
CUDA_CFLAGS="-I${CUDA_ROOT}/include"
CUDA_LFLAGS="-L${CUDA_ROOT}/lib -lcudart -lcublas"

# Compiler flags
CXXFLAGS="-std=c++17 -fPIC -O2 -DGOOGLE_CUDA=1"
NVCCFLAGS="-std=c++17 --expt-relaxed-constexpr -O2 -DGOOGLE_CUDA=1"

# GPU architectures (modify as needed for your GPU)
GPU_ARCHS="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89"

echo "Compiling CUDA kernel..."
conda run --live-stream --name tf215 nvcc $NVCCFLAGS $GPU_ARCHS $CUDA_CFLAGS $TF_CFLAGS \
    -c "${DIR}/dcn_v2_cuda.cu" -o "${DIR}/dcn_v2_cuda.cu.o"

echo "Compiling C++ op..."
conda run --live-stream --name tf215 g++ $CXXFLAGS $TF_CFLAGS $CUDA_CFLAGS \
    -c "${DIR}/dcn_v2_op.cc" -o "${DIR}/dcn_v2_op.cc.o"

echo "Linking shared library..."
conda run --live-stream --name tf215 g++ -shared -fPIC \
    "${DIR}/dcn_v2_cuda.cu.o" "${DIR}/dcn_v2_op.cc.o" \
    $TF_LFLAGS $CUDA_LFLAGS \
    -o "${DIR}/dcn_v2_op.so"

echo "Cleaning up object files..."
rm -f "${DIR}/dcn_v2_cuda.cu.o" "${DIR}/dcn_v2_op.cc.o"

echo "Build complete! Library saved as ${DIR}/dcn_v2_op.so"

# Test the library
echo "Testing the library..."
conda run --live-stream --name tf215 python -c "
import sys
sys.path.insert(0, '${DIR}')
from dcn_v2_op import DCNv2Optimized
print('Library loaded successfully!')
"

echo "Done!"
