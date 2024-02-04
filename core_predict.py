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
    distribute_strategy : tf.distribute.Strategy,
    batch_size : int,
    model,
    num_class : int,
    input_dir : str,
    crop_height=513,
    crop_width=513,
    image_count=0,
    image_sets=None,
    scale_rates=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    flip=True,
    output_dir=None,
    image_predict_func=None,
    output_ext=".png",
):
    
    if image_predict_func is None:
        image_predict_func = default_image_predict

    compiled_image_predict_func = None

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
        ds = ds.take(image_count + image_count % batch_size)
        ds = ds.batch(batch_size, drop_remainder=True)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        ds = ds.with_options(options)

        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        ds = distribute_strategy.experimental_distribute_dataset(ds)
        # ds = ds.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

        @tf.function(autograph=False)
        def step_fn(image_tensor):

            curent_dtype = tf.keras.backend.floatx()
            image_tensor = tf.cast(image_tensor, curent_dtype)

            nonlocal compiled_image_predict_func
            if compiled_image_predict_func is None:

                if hasattr(model, "backbone_use_xla"):
                    model.backbone_use_xla = False

                compiled_image_predict_func = tf.function(
                    image_predict_func, 
                    autograph=False,
                ).get_concrete_function(
                    model,
                    tf.TensorSpec([None, None, None, 3], curent_dtype),
                    scale_rates,
                    flip,
                )


            return compiled_image_predict_func(
                image_tensor,
            )
        
        @tf.function(autograph=False)
        def run_fn (image_tensor):
            return distribute_strategy.run(step_fn, args=(image_tensor,))
        

        with tqdm(total=len(paths[0])) as pbar:

            for image_tensor, output_size, output_name in ds:
                
                predicts = run_fn(image_tensor)

                output_name = distribute_strategy.experimental_local_results(
                    output_name
                )

                output_name = tf.concat(output_name, axis=0)
                output_path = tf.strings.join([output_dir, output_name], separator=os.sep)

                output_size = distribute_strategy.experimental_local_results(
                    output_size
                )
                output_size = tf.concat(output_size, axis=0)

                for i in range(len(predicts)):
                    batch_predict = predicts[i]
                    batch_predict = distribute_strategy.experimental_local_results(
                        batch_predict
                    )
                    batch_predict = tf.concat(batch_predict, axis=0)

                    for k in range(batch_size):
                        predict = batch_predict[k]
                        orginal_size = output_size[k]

                        predict = tf.image.crop_to_bounding_box(predict, 0, 0, orginal_size[0], orginal_size[1])
                        predict = tf.cast(predict, tf.uint8)

                        png = tf.io.encode_png(predict)
                        path = output_path[k]

                        if i > 0:
                            path = path + "_{}".format(i)

                        tf.io.write_file(path + output_ext, png)

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
    scale_rates=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    flip=True,
):
    
    # image_tensor = tf.cast(image_tensor, tf.int32)

    logits = model.inference_with_multi_scales(
        image_tensor, 
        training=False, 
        scale_rates=scale_rates, 
        flip=flip
    )

    logits = convert_to_list_if_single(logits)
    predicts = multi_results_handler(
        logits, lambda x: tf.argmax(x, axis=-1, output_type=tf.int32)
    )  # Now the size = [N, h, w]
    predicts = multi_results_handler(predicts, lambda x: tf.expand_dims(x, axis=-1))  # Now the size = [N, h, w, 1]

    return predicts


