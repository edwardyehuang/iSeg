## iSeg : A Keras 3 ~~TensorFlow 2~~ Library for Semantic Segmentation

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/tensorflow-gpu/badges/version.svg)](https://anaconda.org/conda-forge/tensorflow-gpu)

### News

March-14-2024: Add support for Gemma (Keras 2.15 compatible).

March-05-2024: Add support for keras 3. Note that I still retain support for keras 2 and currently only support the TensorFlow backend. Support for Pytorch and JAX backends will be released at a later date.

January-14-2024: Add support for TensorFlow 2.15

January-11-2024: Add [EVA02](https://github.com/baaivision/EVA/tree/master/EVA-02).

January-02-2024: Add DCNv3 and InternImage backbone.

April-01-2023: Add weights for ViT-SAM.

March-17-2023: Drop the support for old ResNet-50/101 h5 weights. Updated versions have been provided.

March-01-2023: Add support of TPU pod training, we will add an example project soon.

January-03-2023: Add support of [ConvNeXtV2](https://arxiv.org/abs/2301.00808).

November-12-2022: Add support of [MOAT](https://arxiv.org/abs/2210.01820).

September-7-2022: Drop support for TensorFlow < 2.10

March-16-2022: We add an example project in [here](https://github.com/edwardyehuang/CAR), which is the offical implementation of the paper [CAR: Class-aware Regularizations for Semantic Segmentation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136880514.pdf)

March-16-2022: The backbone weights are now available in [here](backbones/README.md), we will add more in future.


<img src="demo.png" width=400>

## Features
### Backbone support (Inputs size free)

- [x] Modern ResNet
- [x] Xception-65
- [x] MobileNetV2
- [x] EfficientNetV1
- [x] Feature Pyramid Network
- [x] HRNet
- [x] Vision Transformer
- [x] Swin Transformer 
- [x] MobileNetV2
- [x] ConvNeXt
- [x] MOAT
- [x] ConvNeXtV2
- [x] InternImage
- [x] EVA02

All backbones are independent of input size.

Weights can be downloaded in [here](backbones/README.md)

### Other features
- [x] Mixed precision training and inference
- [x] Fully deterministic result (100%, see https://github.com/NVIDIA/framework-determinism)
- [x] Training and inference on GPU <= 8
- [x] Training and inference on TPU/TPU Pod
- [x] Typical image augmentation
- [x] Support for Windows 10/11
- [x] Support for Windows WSL2
- [x] Support for Apple M1 chip macOS


## Requirements

* TensorFlow >= 2.10 (For iseg <= 0.04, we support TensorFlow >= 2.4)
* Mixed precision only supports GPU architectures after Volta (included).

## Installation (Conda)

The following order can avoid many bugs.
Make sure the NVIDIA and CUDA driver is the latest version.

```
conda create -n tf215 python=3.10 tqdm matplotlib gitpython -c conda-forge
pip install --upgrade pip setuptools
pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com tensorrt-libs==8.6.1
pip install tensorflow[and-cuda]==2.15 ml-dtypes
pip install tensorflow-text==2.15
pip install keras-nlp==0.8.2
```