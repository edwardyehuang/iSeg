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


#@tf.function
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

        if image_sets is None:
            ds = tf.data.Dataset.from_generator(
                dir_data_generator, 
                args=(input_dir, crop_height, crop_width),
                output_types=(tf.float32, tf.int32, tf.string),
            )
        else:
            ds = tf.data.Dataset.from_generator(
                dir_data_generator_with_imagesets,
                args=(input_dir, crop_height, crop_width, image_sets),
                output_types=(tf.float32, tf.int32, tf.string),
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

        for image_tensor, output_size, output_name in ds:
            distribute_strategy.run(step_fn, args=(image_tensor, output_size, output_name))

            counter += batch_size

            tf.print("processed : ", counter)


def dir_data_generator(input_dir, crop_height=513, crop_width=513):

    return dir_data_generator_with_imagesets(input_dir, crop_height, crop_width)


def dir_data_generator_with_imagesets(
    input_dir, 
    crop_height=513, 
    crop_width=513, 
    image_sets=None
):

    for root, dirs, files in tf.io.gfile.walk(input_dir):
        for filename in files:
            file_path = root + os.sep + filename

            filename_wo_ext = os.path.splitext(filename)[0]

            if image_sets is not None and str.encode(filename_wo_ext) not in image_sets:
                continue
            
            image_tensor, _ = load_image_tensor_from_path(file_path)
            image_tensor = tf.expand_dims(image_tensor, axis=0)
            image_size = tf.shape(image_tensor)[1:3]

            image_tensor = pad_to_bounding_box(
                image_tensor, 
                0, 
                0, 
                crop_height, 
                crop_width, 
                pad_value=[127.5, 127.5, 127.5]
            
            )

            image_tensor = normalize_value_range(image_tensor)

            yield image_tensor, image_size, filename_wo_ext

            # output_path = tf.strings.join([output_dir, filename_wo_ext], separator=os.sep)
            # yield file_path, output_path


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
