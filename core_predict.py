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


def __load_batch_image_mapfn(input_path, output_path):

    batch_size = tf.shape(input_path)[0]
    max_height = tf.constant(1, tf.int32)
    max_width = tf.constant(1, tf.int32)

    batched_images = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, name="batched_images")

    orginal_sizes = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, name="orginal_sizes")

    for i in tf.range(batch_size):
        image_tensor = tf.image.decode_jpeg(tf.io.read_file(input_path[i]), channels=3)
        image_tensor = tf.expand_dims(image_tensor, axis=0)
        image_size = tf.shape(image_tensor)[1:3]

        image_tensor = tf.cast(image_tensor, tf.float32)

        batched_images = batched_images.write(i, image_tensor)
        orginal_sizes = orginal_sizes.write(i, image_size)

        image_height = image_size[0] + 1 if image_size[0] % 2 == 0 else image_size[0]
        image_width = image_size[1] + 1 if image_size[1] % 2 == 0 else image_size[1]

        max_height = max_height if max_height >= image_height else image_height
        max_width = max_width if max_width >= image_width else image_width

    for i in tf.range(batch_size):

        image_tensor = batched_images.read(i)
        image_tensor = pad_to_bounding_box(image_tensor, 0, 0, max_height, max_width, pad_value=[127.5, 127.5, 127.5])
        image_tensor = normalize_value_range(image_tensor)

        batched_images = batched_images.write(i, image_tensor)

    result_batched_images = batched_images.concat()
    result_orginal_sizes = orginal_sizes.stack()

    batched_images.close()
    orginal_sizes.close()

    tf.debugging.assert_equal(tf.shape(result_batched_images)[0], batch_size)
    tf.debugging.assert_equal(tf.shape(result_orginal_sizes)[0], batch_size)

    return result_batched_images, result_orginal_sizes, output_path


@tf.function
def predict_with_dir(
    distribute_strategy,
    batch_size,
    model,
    num_class,
    input_dir,
    image_count=0,
    image_sets=None,
    scale_rates=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    flip=True,
    output_dir=None,
):

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.mkdir(output_dir)

    with distribute_strategy.scope():

        if image_sets is None:
            ds = tf.data.Dataset.from_generator(
                dir_data_generator, output_types=(tf.string, tf.string), args=(input_dir, output_dir)
            )
        else:
            ds = tf.data.Dataset.from_generator(
                dir_data_generator_with_imagesets,
                output_types=(tf.string, tf.string),
                args=(input_dir, output_dir, image_sets),
            )

        ds = ds.repeat()
        ds = ds.take(image_count + 1)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.map(__load_batch_image_mapfn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = distribute_strategy.experimental_distribute_dataset(ds, tf.distribute.InputOptions())
        # ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

        @tf.function
        def step_fn(image_tensor, output_size, output_path):
            preidct_with_image(
                model,
                image_tensor=image_tensor,
                output_size=output_size,
                output_path=output_path,
                scale_rates=scale_rates,
                flip=flip,
            )

        counter = 0

        for image_tensor, output_size, output_path in ds:
            distribute_strategy.run(step_fn, args=(image_tensor, output_size, output_path))

            counter += batch_size

            tf.print(tf.strings.format("processed : {}", counter))


def dir_data_generator(input_dir, output_dir):

    return dir_data_generator_with_imagesets(input_dir, output_dir)


def dir_data_generator_with_imagesets(input_dir, output_dir, image_sets=None):

    for root, dirs, files in tf.io.gfile.walk(input_dir):
        for filename in files:
            file_path = root + os.sep + filename

            output_path = None
            filename_wo_ext = os.path.splitext(filename)[0]

            if image_sets is not None and str.encode(filename_wo_ext) not in image_sets:
                continue

            output_path = tf.strings.join([output_dir, filename_wo_ext], separator=os.sep)
            yield file_path, output_path


def preidct_with_image(
    model: SegBase,
    image_tensor,
    output_size,
    output_path,
    output_ext=".png",
    scale_rates=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    flip=True,
):

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
