# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import tensorflow as tf
import iseg.static_strings as ss

from iseg.backbones.resnet_common import *
from iseg.backbones.xception_common import Xception, xception65, build_atrous_xception
from iseg.backbones.mobilenetv2_common import MobileNetV2, build_atrous_mobilenetv2
from iseg.backbones.swin import swin_tiny_224, swin_base_384, swin_large_384

from iseg.backbones.backbone_registry import backbone_registry_dict

from iseg.backbones.efficientnet import *
from iseg.backbones.hrnet import HRNetW48, HRNetW32
from iseg.backbones.placeholder import PlaceHolder
from iseg.backbones.convnext import convnext_tiny, convnext_large, convnext_xlarge, build_dilated_convnext, convnext_xxlarge
from iseg.backbones.moat.moat import moat0, moat1, moat2, moat3, moat4
from iseg.backbones.convnext_v2 import convnext_v2_nano, convnext_v2_tiny, convnext_v2_large, convnext_v2_huge
from iseg.backbones.vit import ViT16B, ViT16L, ViT16B_SAM
from iseg.backbones.intern_image.intern_image import intern_image_tiny, intern_image_small, intern_image_huge


def get_backbone(
    name=ss.RESNET50,
    output_stride=32,
    resnet_multi_grids=[1, 2, 4],
    resnet_slim=True,
    custom_resblock=None,
    weights_path=None,
    return_endpoints=False,
    image_shape=(1, 513, 513, 3),
    label_shape=None,
    moat_use_pos_encoding=False,
):

    backbone = None

    name = name.lower()

    general_kwargs = {
        "return_endpoints": return_endpoints,
    }

    resnet_kwargs = {
        "use_bias": False,
        "replace_7x7_conv": True,
        "slim_behaviour": resnet_slim,
        "custom_block": custom_resblock,
    }

    if ss.RESNET in name:
        general_kwargs.update(resnet_kwargs)

    if ss.MOAT in name:
        general_kwargs.update({
            "use_pos_emb": moat_use_pos_encoding,
        })

    backbone_dicts = {
        ss.RESNET9: resnet9,
        ss.RESNET10: resnet10,
        ss.RESNET18: resnet18,
        ss.RESNET50: resnet50,
        ss.RESNET52: resnet50,
        ss.RESNET101: resnet101,
        ss.RESNET103: resnet101,
        ss.XCEPTION65: xception65,
        ss.EFFICIENTNETB0: EfficientNetB0,
        ss.EFFICIENTNETB1: EfficientNetB1,
        ss.EFFICIENTNETB2: EfficientNetB2,
        ss.EFFICIENTNETB3: EfficientNetB3,
        ss.EFFICIENTNETB4: EfficientNetB4,
        ss.EFFICIENTNETB5: EfficientNetB5,
        ss.EFFICIENTNETB6: EfficientNetB6,
        ss.EFFICIENTNETB7: EfficientNetB7,
        ss.EFFICIENTNETL2: EfficientNetL2,
        ss.SWIN_TINY_224: swin_tiny_224,
        ss.SWIN_BASE_384: swin_base_384,
        ss.SWIN_LARGE_384: swin_large_384,
        ss.MOBILENETV2: MobileNetV2,
        ss.HRNET_W48: HRNetW48,
        ss.HRNET_W32: HRNetW32,
        ss.CONVNEXT_TINY: convnext_tiny,
        ss.CONVNEXT_LARGE: convnext_large,
        ss.CONVNEXT_XLARGE: convnext_xlarge,
        ss.CONVNEXT_XXLARGE: convnext_xxlarge,
        ss.MOAT0: moat0,
        ss.MOAT1: moat1,
        ss.MOAT2: moat2,
        ss.MOAT3: moat3,
        ss.MOAT4: moat4,
        ss.CONVNEXT_V2_NANO: convnext_v2_nano,
        ss.CONVNEXT_V2_TINY: convnext_v2_tiny,
        ss.CONVNEXT_V2_LARGE: convnext_v2_large,
        ss.CONVNEXT_V2_HUGE: convnext_v2_huge,
        ss.VIT_B: ViT16B,
        ss.VIT_L: ViT16L,
        ss.VIT_B_SAM: ViT16B_SAM,
        ss.INTERN_IMAGE_TINY: intern_image_tiny,
        ss.INTERN_IMAGE_SMALL: intern_image_small,
        ss.INTERN_IMAGE_HUGE: intern_image_huge,
        ss.PLACEHOLDER: PlaceHolder,
    }

    backbone_dicts.update(backbone_registry_dict)

    if not name in backbone_dicts:
        raise ValueError(f"Backbone {name} currently not supported")

    backbone = backbone_dicts[name](**general_kwargs)

    if ss.RESNET in name:
        build_atrous_resnet(backbone, output_stride=output_stride)
        apply_multi_grid(backbone, block_index=-1, grids=resnet_multi_grids)
    elif ss.XCEPTION in name:
        build_atrous_xception(backbone, output_stride=output_stride)
    elif ss.EFFICIENTNET in name:
        build_dilated_efficientnet(backbone, output_stride=output_stride)
    elif name == ss.MOBILENETV2:
        build_atrous_mobilenetv2(backbone, output_stride=output_stride)
    elif ss.CONVNEXT in name:
        build_dilated_convnext(backbone, output_stride=output_stride)

    if weights_path is not None:
        if "swin" in name:
            if label_shape is None:
                backbone(tf.ones(image_shape))
            else:
                backbone((tf.ones(image_shape), tf.ones(label_shape)))
        else:
            if label_shape is None:
                backbone.build(input_shape=image_shape)  # backward compatibility
            else:
                backbone((tf.ones(image_shape), tf.ones(label_shape)))

        if ".h5" in weights_path[-3:]:
            print(f"Load backbone weights {weights_path} as H5 format")
            backbone.load_weights(weights_path, by_name=True) 
        elif ".ckpt" in weights_path[-5:]:
            print(f"Load backbone weights {weights_path} as ckpt format")
            backbone.load_weights(weights_path)
        else:
            raise ValueError(f"Weights {weights_path} not supported")

    return backbone


def is_slim_structure(resnet):

    stack_type = type(resnet.stacks[0])

    if stack_type is Stack:
        return False
    elif stack_type is Stack2:
        return True

    raise ValueError("Unknown stack")
