# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

from iseg.data_process.augments.resize_augment import ResizeAugment
from iseg.data_process.augments.random_scale_augment import RandomScaleAugment
from iseg.data_process.augments.pad_augment import PadAugment
from iseg.data_process.augments.random_crop_augment import RandomCropAugment
from iseg.data_process.augments.random_flip_augment import RandomFlipAugment
from iseg.data_process.augments.random_brightness_augment import RandomBrightnessAugment
from iseg.data_process.augments.random_contrast_augment import RandomContrastAugment
from iseg.data_process.augments.random_hue_augment import RandomHueAugment
from iseg.data_process.augments.random_saturation_augment import RandomSaturationAugment
from iseg.data_process.augments.random_photo_metric_distortions import RandomPhotoMetricDistortions
from iseg.data_process.augments.random_erasing_augment import RandomErasingAugment
from iseg.data_process.augments.random_jepg_quality_augment import RandomJEPGQualityAugment
from iseg.data_process.augments.random_noisy_eval_augment import RandomNoisyEvalAugment
from iseg.data_process.augments.random_rotate_augment import RandomRotateAugment