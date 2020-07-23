import numpy as np
import tensorflow as tf


def parse_dataset(example_proto, img_size_h, img_size_w, img_size_d, dim_anno1, 
    dim_anno2, dim_anno3, dtype_img=tf.float64):
    features = {
        'inputs': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'annotations': tf.io.FixedLenFeature(shape=[dim_anno1], dtype=tf.float32),
        'psl_mean_ens': tf.io.FixedLenFeature(shape=[dim_anno2], dtype=tf.float32),
        'temp_mean_ens': tf.io.FixedLenFeature(shape=[dim_anno3], dtype=tf.float32),
        'year': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        'month': tf.io.FixedLenFeature(shape=[], dtype=tf.float32),
        'day': tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
    }

    parsed_features = tf.io.parse_single_example(example_proto, features=features)
    image = tf.io.decode_raw(parsed_features["inputs"], dtype_img)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [img_size_h, img_size_w, img_size_d])
    annotations = parsed_features["annotations"]
    psl = parsed_features["psl_mean_ens"]
    temp = parsed_features["temp_mean_ens"]
    year = parsed_features["year"]
    month = parsed_features["month"]
    day = parsed_features["day"]

    return image, annotations, psl, temp, year, month, day


def climate_dataset(directory, filenames, height, width, depth, dim_anno1, 
    dim_anno2, dim_anno3, dtype):

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: parse_dataset(x, height, width, depth, 
        dim_anno1, dim_anno2, dim_anno3, dtype_img=dtype))

    return dataset