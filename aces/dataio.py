# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import tensorflow as tf
from functools import partial


class DataIO:

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def create_tfrecord_from_file(filename):
        return tf.data.TFRecordDataset(filename, compression_type="GZIP")

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def get_single_sample(x):
        return tf.numpy_function(lambda _: 1, inp=[x], Tout=tf.int64)
    
    @staticmethod
    def filter_good_patches(patch):
        # the getdownload url has field names so we're using view here
        has_nan = np.isnan(np.sum(patch.view(np.float32)))
        has_inf = np.isinf(np.sum(patch.view(np.float32)))
        if has_nan or has_inf:
            return False
        return True

    @staticmethod
    def calculate_n_samples(**config):
        parser = partial(DataIO.parse_tfrecord_multi_label,
                         patch_size=config.get("PATCH_SHAPE_SINGLE"),
                         features=config.get("FEATURES"),
                         labels=config.get("LABELS"))
        tupler = partial(DataIO.to_tuple_multi_label, depth=config.get("OUT_CLASS_NUM"), x_only=True)

        tf_training_records = tf.data.Dataset.list_files(f"{str(config.get('TRAINING_DIR'))}/*")\
                                             .interleave(DataIO.create_tfrecord_from_file, num_parallel_calls=tf.data.AUTOTUNE)
        tf_training_records = tf_training_records.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        tf_training_records = tf_training_records.map(tupler, num_parallel_calls=tf.data.AUTOTUNE)
        d0_training_records = tf_training_records.map(DataIO.get_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        n_training_records = d0_training_records.reduce(np.int64(0), lambda x, y: x + y).numpy()

        tf_testing_records = tf.data.Dataset.list_files(f"{str(config.get('TESTING_DIR'))}/*")\
                                            .interleave(DataIO.create_tfrecord_from_file, num_parallel_calls=tf.data.AUTOTUNE)
        tf_testing_records = tf_testing_records.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        tf_testing_records = tf_testing_records.map(tupler, num_parallel_calls=tf.data.AUTOTUNE)
        d0_testing_records = tf_testing_records.map(DataIO.get_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        n_testing_records = d0_testing_records.reduce(np.int64(0), lambda x, y: x + y).numpy()

        tf_validation_records = tf.data.Dataset.list_files(f"{str(config.get('VALIDATION_DIR'))}/*")\
                                               .interleave(DataIO.create_tfrecord_from_file, num_parallel_calls=tf.data.AUTOTUNE)
        tf_validation_records = tf_validation_records.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tf_validation_records = tf_validation_records.map(tupler, num_parallel_calls=tf.data.AUTOTUNE)
        d0_validation_records = tf_validation_records.map(DataIO.get_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        n_validation_records = d0_validation_records.reduce(np.int64(0), lambda x, y: x + y).numpy()

        return n_training_records, n_testing_records, n_validation_records


    @staticmethod
    def print_dataset_info(dataset: tf.data.Dataset, dataset_name: str) -> None:
        logging.info(dataset_name)
        for inputs, outputs in dataset.take(1):
            try:
                logging.info(f"inputs: {inputs.dtype.name} {inputs.shape}")
                print(inputs)
                logging.info(f"outputs: {outputs.dtype.name} {outputs.shape}")
                print(outputs)
            except:
                logging.info(f" > inputs:")
                for name, values in inputs.items():
                    logging.info(f"    {name}: {values.dtype.name} {values.shape}")
                logging.info(f" > outputs: {outputs.dtype.name} {outputs.shape}")

    @staticmethod
    @tf.function
    def random_transform(dataset: tf.Tensor) -> tf.Tensor:
        x = tf.random.uniform(())

        if x < 0.14:
            return tf.concat([dataset, tf.image.flip_left_right(dataset)], 0)
        elif tf.math.logical_and(x >= 0.14, x < 0.28):
            return tf.concat([dataset, tf.image.flip_left_right(dataset)], 0)
        elif tf.math.logical_and(x >= 0.28, x < 0.42):
            return tf.concat([dataset, tf.image.flip_left_right(tf.image.flip_up_down(dataset))], 0)
        elif tf.math.logical_and(x >= 0.42, x < 0.56):
            return tf.concat([dataset, tf.image.rot90(dataset, k=1)], 0)
        elif tf.math.logical_and(x >= 0.56, x < 0.70):
            return tf.concat([dataset, tf.image.rot90(dataset, k=2)], 0)
        elif tf.math.logical_and(x >= 0.70, x < 0.84):
            return tf.concat([dataset, tf.image.rot90(dataset, k=3)], 0)
        else:
            return tf.concat([dataset, tf.image.flip_left_right(tf.image.rot90(dataset, k=2))], 0)

    @staticmethod
    @tf.function
    def flip_inputs_up_down(inputs: tf.Tensor) -> tf.Tensor:
        return tf.image.flip_up_down(inputs)

    @staticmethod
    @tf.function
    def flip_inputs_left_right(inputs: tf.Tensor) -> tf.Tensor:
        return tf.image.flip_left_right(inputs)

    @staticmethod
    @tf.function
    def transpose_inputs(inputs: tf.Tensor) -> tf.Tensor:
        flip_up_down = tf.image.flip_up_down(inputs)
        transpose = tf.image.flip_left_right(flip_up_down)
        return transpose

    @staticmethod
    @tf.function
    def rotate_inputs_90(inputs: tf.Tensor) -> tf.Tensor:
        return tf.image.rot90(inputs, k=1)

    @staticmethod
    @tf.function
    def rotate_inputs_180(inputs: tf.Tensor) -> tf.Tensor:
        return tf.image.rot90(inputs, k=2)

    @staticmethod
    @tf.function
    def rotate_inputs_270(inputs: tf.Tensor) -> tf.Tensor:
        return tf.image.rot90(inputs, k=3)

    @staticmethod
    @tf.function
    def parse_tfrecord(example_proto: tf.Tensor, patch_size: int, features: list = None, labels: list = None) -> tf.data.Dataset:
        keys = features + labels
        columns = [
            tf.io.FixedLenFeature(shape=[patch_size, patch_size], dtype=tf.float32) for _ in keys
        ]
        proto_struct = dict(zip(keys, columns))
        inputs = tf.io.parse_single_example(example_proto, proto_struct)
        inputs_list = [inputs.get(key) for key in keys]
        stacked = tf.stack(inputs_list, axis=0)
        stacked = tf.transpose(stacked, [1, 2, 0])
        return tf.data.Dataset.from_tensors(stacked)

    @staticmethod
    @tf.function
    def to_tuple(dataset: tf.Tensor, n_features: int = None, inverse_labels: bool = False) -> tuple:
        features = dataset[:, :, :, :n_features]
        labels = dataset[:, :, :, n_features:]
        if inverse_labels:
            labels_inverse = tf.math.abs(labels - 1)
            labels = tf.concat([labels_inverse, labels], axis=-1)
        return features, labels

    @staticmethod
    @tf.function
    def parse_tfrecord_with_name(example_proto: tf.Tensor, patch_size: int, features: list = None, labels: list = None) -> tf.data.Dataset:
        keys = features + labels
        columns = [
            tf.io.FixedLenFeature(shape=[patch_size, patch_size], dtype=tf.float32) for _ in keys
        ]
        proto_struct = dict(zip(keys, columns))
        return tf.io.parse_single_example(example_proto, proto_struct)

    @staticmethod
    @tf.function
    def to_tuple_with_name(inputs: tf.Tensor, features: list = None, labels: list = None, n_classes: int = 1) -> tuple:
        return (
            {name: inputs[name] for name in features},
            tf.one_hot(tf.cast(inputs[labels[0]], tf.uint8), n_classes)
        )

    @staticmethod
    @tf.function
    def parse_tfrecord_dnn(example_proto: tf.Tensor, features: list = None, labels: list = None) -> tuple:
        keys = features + labels
        columns = [
            tf.io.FixedLenFeature(shape=[1], dtype=tf.float32) for _ in keys
        ]
        proto_struct = dict(zip(keys, columns))
        parsed_features = tf.io.parse_single_example(example_proto, proto_struct)
        label = parsed_features.pop(labels[0])
        label = tf.cast(label, tf.int32)
        return parsed_features, label

    @staticmethod
    @tf.function
    def to_tuple_dnn(dataset: dict, label: tf.Tensor, depth: int = 1) -> tuple:
        return tf.transpose(list(dataset.values())), tf.one_hot(indices=label, depth=depth)

    @staticmethod
    @tf.function
    def parse_tfrecord_multi_label(example_proto: tf.data.Dataset, patch_size: int, features: list = None, labels: list = None) -> tuple:
        keys = features + labels
        columns = [
            tf.io.FixedLenFeature(shape=[patch_size, patch_size], dtype=tf.float32) for _ in keys
        ]
        proto_struct = dict(zip(keys, columns))
        parsed_features = tf.io.parse_single_example(example_proto, proto_struct)
        label = parsed_features.pop(labels[0])
        return parsed_features, label

    @staticmethod
    @tf.function
    def to_tuple_multi_label(dataset: dict, label: tf.Tensor, depth: int = 1, x_only: bool = False) -> tuple:
        label = tf.cast(label, tf.uint8)
        label = tf.one_hot(indices=label, depth=depth)
        parsed_dataset = tf.transpose(list(dataset.values()))
        if x_only:
            return parsed_dataset
        return parsed_dataset, label

    @staticmethod
    def _get_dataset(files: list, features: list, labels: list, patch_shape: list, batch_size: int, buffer_size: int = 1000, training: bool = False, **kwargs) -> tf.data.Dataset:
        dnn = kwargs.get('dnn', False)
        inverse_labels = kwargs.get('inverse_labels', False)
        depth = kwargs.get('depth', len(labels))
        multi_label_unet = kwargs.get('multi_label_unet', False)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')

        if dnn:
            parser = partial(DataIO.parse_tfrecord_dnn, features=features, labels=labels)
            split_data = partial(DataIO.to_tuple_dnn, depth=depth)
            dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            return dataset

        if multi_label_unet:
            parser = partial(DataIO.parse_tfrecord_multi_label, features=features, labels=labels, patch_shape=patch_shape)
            split_data = partial(DataIO.to_tuple_multi_label, n_features=len(features), depth=depth)
            dataset = dataset.interleave(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if training:
                dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size) \
                                 .map(DataIO.random_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                 .map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            else:
                dataset = dataset.batch(batch_size).map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            return dataset

        parser = partial(DataIO.parse_tfrecord, features=features, labels=labels)
        split_data = partial(DataIO.to_tuple, n_features=len(features), inverse_labels=inverse_labels)
        dataset = dataset.interleave(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if training:
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size) \
                             .map(DataIO.random_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                             .map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.batch(batch_size).map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    @staticmethod
    def get_dataset(pattern: str, features: list, labels: list, patch_size: int, batch_size: int, n_classes: int = 1) -> tf.data.Dataset:
        logging.info(f"Loading dataset from {pattern}")
        logging.info(f"list_files: {tf.data.Dataset.list_files(pattern)}")

        parser = partial(DataIO.parse_tfrecord_multi_label, patch_size=patch_size, features=features, labels=labels)
        tupler = partial(DataIO.to_tuple_multi_label, depth=n_classes)

        dataset = tf.data.Dataset.list_files(pattern).interleave(DataIO.create_tfrecord_from_file)
        dataset = dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(tupler, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.cache()
        dataset = dataset.shuffle(512)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
