# ================================================================
# MIT License
# Copyright (c) 2021 edwardyehuang (https://github.com/edwardyehuang)
# ================================================================

import os

import tensorflow as tf
import numpy as np

from PIL import Image

from iseg.data_process.utils import pad_to_bounding_box, resize_to_range, normalize_value_range
from iseg.core_inference import *

from iseg.core_model import SegBase
from iseg.utils.data_loader import load_image_tensor_from_path

from tqdm import tqdm


def predict_with_dir(
    distribute_strategy,
    batch_size,
    model,
    num_class,
    input_dir,
    crop_height=513,
    crop_width=513,
    image_count=0,
    image_sets=None,
    scale_rates=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    flip=True,
    output_dir=None,
    image_predict_func=None
):
    
    if image_predict_func is None:
        image_predict_func = default_image_predict

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)

    print("Input dir : ", input_dir)
    print("Output dir : ", output_dir)

    with distribute_strategy.scope():

        paths = get_data_paths(input_dir, image_sets)
        ds = tf.data.Dataset.from_tensor_slices(paths)

        ds = ds.map(
            data_process(crop_height, crop_width), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        ds = ds.repeat()
        ds = ds.take(image_count + 1)
        ds = ds.batch(batch_size, drop_remainder=True)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = distribute_strategy.experimental_distribute_dataset(ds)
        # ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

        @tf.function
        def step_fn(image_tensor, output_size, output_name):
            image_tensor.set_shape([None, crop_height, crop_width, 3])
            image_predict_func(
                model,
                image_tensor=image_tensor,
                output_size=output_size,
                output_dir=output_dir,
                output_name=output_name,
                scale_rates=scale_rates,
                flip=flip,
            )

        counter = 0

        with tqdm(total=len(paths[0])) as pbar:

            for image_tensor, output_size, output_name in ds:
                
                distribute_strategy.run(step_fn, args=(image_tensor, output_size, output_name))

                counter += batch_size

                pbar.update(batch_size)


def get_data_paths(
    input_dir, 
    image_sets=None
):
    paths = []
    names = []

    for root, dirs, files in tf.io.gfile.walk(input_dir):
        for filename in files:
            file_path = root + os.sep + filename

            filename_wo_ext = os.path.splitext(filename)[0]

            if image_sets is not None and str.encode(filename_wo_ext) not in image_sets:
                continue

            paths.append(file_path)
            names.append(filename_wo_ext)

    return paths, names


def data_process (crop_height, crop_width):

    def inner_fn (file_path, filename_wo_ext):

        image_tensor, _ = load_image_tensor_from_path(file_path)
        image_size = tf.shape(image_tensor)[0:2]

        image_tensor = pad_to_bounding_box(
            image_tensor, 
            0, 
            0, 
            crop_height, 
            crop_width, 
            pad_value=[127.5, 127.5, 127.5]
        
        )

        image_tensor = normalize_value_range(image_tensor)

        return image_tensor, image_size, filename_wo_ext
    
    return inner_fn


def default_image_predict(
    model: SegBase,
    image_tensor,
    output_size,
    output_dir,
    output_name,
    output_ext=".png",
    scale_rates=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    flip=True,
):
    output_path = tf.strings.join([output_dir, output_name], separator=os.sep)

    image_tensor = tf.cast(image_tensor, tf.float32)

    logits = model.inference_with_multi_scales(image_tensor, training=False, scale_rates=scale_rates, flip=flip)

    logits = convert_to_list_if_single(logits)
    predicts = multi_results_handler(
        logits, lambda x: tf.argmax(x, axis=-1, output_type=tf.int32)
    )  # Now the size = [N, h, w]
    predicts = multi_results_handler(predicts, lambda x: tf.expand_dims(x, axis=-1))  # Now the size = [N, h, w, 1]

    for i in range(len(predicts)):
        batch_predict = predicts[i]
        batch_size = tf.shape(batch_predict)[0]

        for j in tf.range(batch_size):
            predict = batch_predict[j]
            orginal_size = output_size[j]

            predict = tf.image.crop_to_bounding_box(predict, 0, 0, orginal_size[0], orginal_size[1])
            predict = tf.cast(predict, tf.uint8)

            png = tf.io.encode_png(predict)
            path = output_path[j]

            if i > 0:
                path = path + "_{}".format(i)

            tf.io.write_file(path + output_ext, png)
