# -*- coding: utf-8 -*-

"""
ACES Data Processor Module
This module provides functions for data input/output and preprocessing for the ACES project.
"""

import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
import tensorflow as tf
from functools import partial

__all__ = ["DataProcessor"]

class DataProcessor:
    """
    ACES Data processor Class

    This class provides functions for data input/output and preprocessing for the ACES project.
    """

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def create_tfrecord_from_file(filename: str) -> tf.data.TFRecordDataset:
        """
        Create a TensorFlow Dataset from a TFRecord file.

        Parameters:
        filename (str): The filename of the TFRecord file.

        Returns:
        tf.data.TFRecordDataset: The TensorFlow Dataset created from the TFRecord file.
        """
        return tf.data.TFRecordDataset(filename, compression_type="GZIP")

    @staticmethod
    @tf.autograph.experimental.do_not_convert
    def get_single_sample(x):
        """
        Get a single sample from a dataset.

        Parameters:
        x: The input tensor.

        Returns:
        tf.Tensor: The single sample.
        """
        return tf.numpy_function(lambda _: 1, inp=[x], Tout=tf.int64)

    @staticmethod
    def calculate_n_samples(**config):
        """
        Calculate the number of samples in the training, testing, and validation datasets.

        Parameters:
        **config: The configuration settings.

        Returns:
        int: The number of training samples.
        int: The number of testing samples.
        int: The number of validation samples.
        """
        parser = partial(DataProcessor.parse_tfrecord_multi_label,
                         patch_size=config.get("PATCH_SHAPE_SINGLE"),
                         features=config.get("FEATURES"),
                         labels=config.get("LABELS"))
        tupler = partial(DataProcessor.to_tuple_multi_label, depth=config.get("OUT_CLASS_NUM"), x_only=True)

        tf_training_records = tf.data.Dataset.list_files(f"{str(config.get('TRAINING_DIR'))}/*")\
                                             .interleave(DataProcessor.create_tfrecord_from_file, num_parallel_calls=tf.data.AUTOTUNE)
        tf_training_records = tf_training_records.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        tf_training_records = tf_training_records.map(tupler, num_parallel_calls=tf.data.AUTOTUNE)
        d0_training_records = tf_training_records.map(DataProcessor.get_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        n_training_records = d0_training_records.reduce(np.int64(0), lambda x, y: x + y).numpy()

        tf_testing_records = tf.data.Dataset.list_files(f"{str(config.get('TESTING_DIR'))}/*")\
                                            .interleave(DataProcessor.create_tfrecord_from_file, num_parallel_calls=tf.data.AUTOTUNE)
        tf_testing_records = tf_testing_records.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        tf_testing_records = tf_testing_records.map(tupler, num_parallel_calls=tf.data.AUTOTUNE)
        d0_testing_records = tf_testing_records.map(DataProcessor.get_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        n_testing_records = d0_testing_records.reduce(np.int64(0), lambda x, y: x + y).numpy()

        tf_validation_records = tf.data.Dataset.list_files(f"{str(config.get('VALIDATION_DIR'))}/*")\
                                               .interleave(DataProcessor.create_tfrecord_from_file, num_parallel_calls=tf.data.AUTOTUNE)
        tf_validation_records = tf_validation_records.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        tf_validation_records = tf_validation_records.map(tupler, num_parallel_calls=tf.data.AUTOTUNE)
        d0_validation_records = tf_validation_records.map(DataProcessor.get_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        n_validation_records = d0_validation_records.reduce(np.int64(0), lambda x, y: x + y).numpy()

        return n_training_records, n_testing_records, n_validation_records


    @staticmethod
    def print_dataset_info(dataset: tf.data.Dataset, dataset_name: str) -> None:
        """
        Print information about a dataset.

        Parameters:
        dataset (tf.data.Dataset): The dataset to print information about.
        dataset_name (str): The name of the dataset.
        """
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
        """
        Apply random transformations to a dataset.

        Parameters:
        dataset (tf.Tensor): The input dataset.

        Returns:
        tf.Tensor: The transformed dataset.
        """
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
        """
        Flip the inputs of a dataset vertically.

        Parameters:
        inputs (tf.Tensor): The input dataset.

        Returns:
        tf.Tensor: The dataset with flipped inputs.
        """
        return tf.image.flip_up_down(inputs)

    @staticmethod
    @tf.function
    def flip_inputs_left_right(inputs: tf.Tensor) -> tf.Tensor:
        """
        Flip the inputs of a dataset horizontally.

        Parameters:
        inputs (tf.Tensor): The input dataset.

        Returns:
        tf.Tensor: The dataset with flipped inputs.
        """
        return tf.image.flip_left_right(inputs)

    @staticmethod
    @tf.function
    def transpose_inputs(inputs: tf.Tensor) -> tf.Tensor:
        """
        Transpose the inputs of a dataset.

        Parameters:
        inputs (tf.Tensor): The input dataset.

        Returns:
        tf.Tensor: The transposed dataset.
        """
        flip_up_down = tf.image.flip_up_down(inputs)
        transpose = tf.image.flip_left_right(flip_up_down)
        return transpose

    @staticmethod
    @tf.function
    def rotate_inputs_90(inputs: tf.Tensor) -> tf.Tensor:
        """
        Rotate the inputs of a dataset by 90 degrees.

        Parameters:
        inputs (tf.Tensor): The input dataset.

        Returns:
        tf.Tensor: The dataset with rotated inputs.
        """
        return tf.image.rot90(inputs, k=1)

    @staticmethod
    @tf.function
    def rotate_inputs_180(inputs: tf.Tensor) -> tf.Tensor:
        """
        Rotate the inputs of a dataset by 180 degrees.

        Parameters:
        inputs (tf.Tensor): The input dataset.

        Returns:
        tf.Tensor: The dataset with rotated inputs.
        """
        return tf.image.rot90(inputs, k=2)

    @staticmethod
    @tf.function
    def rotate_inputs_270(inputs: tf.Tensor) -> tf.Tensor:
        """
        Rotate the inputs of a dataset by 270 degrees.

        Parameters:
        inputs (tf.Tensor): The input dataset.

        Returns:
        tf.Tensor: The dataset with rotated inputs.
        """
        return tf.image.rot90(inputs, k=3)

    @staticmethod
    @tf.function
    def parse_tfrecord(example_proto: tf.Tensor, patch_size: int, features: list = None, labels: list = None) -> tf.data.Dataset:
        """
        Parse a TFRecord example.

        Parameters:
        example_proto (tf.Tensor): The example to parse.
        patch_size (int): The size of the patch.
        features (list, optional): The list of feature names to include. Default is None.
        labels (list, optional): The list of label names to include. Default is None.

        Returns:
        tf.data.Dataset: The parsed dataset.
        """
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
        """
        Convert a dataset to a tuple of features and labels.

        Parameters:
        dataset (tf.Tensor): The input dataset.
        n_features (int, optional): The number of features. Default is None.
        inverse_labels (bool, optional): Whether to inverse the labels. Default is False.

        Returns:
        tuple: A tuple containing the features and labels.
        """
        features = dataset[:, :, :, :n_features]
        labels = dataset[:, :, :, n_features:]
        if inverse_labels:
            labels_inverse = tf.math.abs(labels - 1)
            labels = tf.concat([labels_inverse, labels], axis=-1)
        return features, labels

    @staticmethod
    @tf.function
    def parse_tfrecord_with_name(example_proto: tf.Tensor, patch_size: int, features: list = None, labels: list = None) -> tf.data.Dataset:
        """
        Parse a TFRecord example with named features.

        Parameters:
        example_proto (tf.Tensor): The example to parse.
        patch_size (int): The size of the patch.
        features (list, optional): The list of feature names to include. Default is None.
        labels (list, optional): The list of label names to include. Default is None.

        Returns:
        tf.data.Dataset: The parsed dataset.
        """
        keys = features + labels
        columns = [
            tf.io.FixedLenFeature(shape=[patch_size, patch_size], dtype=tf.float32) for _ in keys
        ]
        proto_struct = dict(zip(keys, columns))
        return tf.io.parse_single_example(example_proto, proto_struct)

    @staticmethod
    @tf.function
    def to_tuple_with_name(inputs: tf.Tensor, features: list = None, labels: list = None, n_classes: int = 1) -> tuple:
        """
        Convert inputs with named features to a tuple of features and one-hot encoded labels.

        Parameters:
        inputs (tf.Tensor): The input dataset.
        features (list, optional): The list of feature names. Default is None.
        labels (list, optional): The list of label names. Default is None.
        n_classes (int, optional): The number of classes for one-hot encoding. Default is 1.

        Returns:
        tuple: A tuple containing the features and one-hot encoded labels.
        """
        return (
            {name: inputs[name] for name in features},
            tf.one_hot(tf.cast(inputs[labels[0]], tf.uint8), n_classes)
        )

    @staticmethod
    @tf.function
    def parse_tfrecord_dnn(example_proto: tf.Tensor, features: list = None, labels: list = None) -> tuple:
        """
        Parse a TFRecord example for DNN models.

        Parameters:
        example_proto (tf.Tensor): The example to parse.
        features (list, optional): The list of feature names to include. Default is None.
        labels (list, optional): The list of label names to include. Default is None.

        Returns:
        tuple: A tuple containing the parsed features and labels.
        """
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
        """
        Convert a dataset for DNN models to a tuple of features and one-hot encoded labels.

        Parameters:
        dataset (dict): The input dataset.
        label (tf.Tensor): The label.
        depth (int, optional): The depth of one-hot encoding. Default is 1.

        Returns:
        tuple: A tuple containing the features and one-hot encoded labels.
        """
        return tf.transpose(list(dataset.values())), tf.one_hot(indices=label, depth=depth)

    @staticmethod
    @tf.function
    def parse_tfrecord_multi_label(example_proto: tf.data.Dataset, patch_size: int, features: list = None, labels: list = None) -> tuple:
        """
        Parse a TFRecord example with multiple labels.

        Parameters:
        example_proto (tf.data.Dataset): The example to parse.
        patch_size (int): The size of the patch.
        features (list, optional): The list of feature names to include. Default is None.
        labels (list, optional): The list of label names to include. Default is None.

        Returns:
        tuple: A tuple containing the parsed features and labels.
        """
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
        """
        Convert a dataset with multiple labels to a tuple of features and multi-hot encoded labels.

        Parameters:
        dataset (tuple): The input dataset.
        n_labels (int, optional): The number of labels. Default is 1.

        Returns:
        tuple: A tuple containing the features and multi-hot encoded labels.
        """
        label = tf.cast(label, tf.uint8)
        label = tf.one_hot(indices=label, depth=depth)
        parsed_dataset = tf.transpose(list(dataset.values()))
        if x_only:
            return parsed_dataset
        return parsed_dataset, label

    @staticmethod
    def _get_dataset(files: list, features: list, labels: list, patch_shape: list, batch_size: int, buffer_size: int = 1000, training: bool = False, **kwargs) -> tf.data.Dataset:
        """
        Get a TFRecord dataset.

        Parameters:
        filenames (list): The list of file names.
        patch_size (int): The size of the patch.
        features (list, optional): The list of feature names to include. Default is None.
        labels (list, optional): The list of label names to include. Default is None.
        batch_size (int, optional): The batch size. Default is 1.
        shuffle (bool, optional): Whether to shuffle the dataset. Default is False.
        n_labels (int, optional): The number of labels. Default is 1.
        num_parallel_calls (int, optional): The number of parallel calls. Default is tf.data.experimental.AUTOTUNE.
        drop_remainder (bool, optional): Whether to drop the remainder of batches. Default is False.
        cache (bool, optional): Whether to cache the dataset. Default is False.

        Returns:
        tf.data.Dataset: The TFRecord dataset.
        """
        dnn = kwargs.get('dnn', False)
        inverse_labels = kwargs.get('inverse_labels', False)
        depth = kwargs.get('depth', len(labels))
        multi_label_unet = kwargs.get('multi_label_unet', False)
        dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')

        if dnn:
            parser = partial(DataProcessor.parse_tfrecord_dnn, features=features, labels=labels)
            split_data = partial(DataProcessor.to_tuple_dnn, depth=depth)
            dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.batch(batch_size)
            return dataset

        if multi_label_unet:
            parser = partial(DataProcessor.parse_tfrecord_multi_label, features=features, labels=labels, patch_shape=patch_shape)
            split_data = partial(DataProcessor.to_tuple_multi_label, n_features=len(features), depth=depth)
            dataset = dataset.interleave(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if training:
                dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size) \
                                 .map(DataProcessor.random_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                                 .map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            else:
                dataset = dataset.batch(batch_size).map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

            return dataset

        parser = partial(DataProcessor.parse_tfrecord, features=features, labels=labels)
        split_data = partial(DataProcessor.to_tuple, n_features=len(features), inverse_labels=inverse_labels)
        dataset = dataset.interleave(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if training:
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size) \
                             .map(DataProcessor.random_transform, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
                             .map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        else:
            dataset = dataset.batch(batch_size).map(split_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset

    @staticmethod
    def get_dataset(pattern: str, features: list, labels: list, patch_size: int, batch_size: int, n_classes: int = 1) -> tf.data.Dataset:
        """
        Get a TFRecord dataset.

        Parameters:
        filenames (list): The list of file names.
        patch_size (int): The size of the patch.
        features (list, optional): The list of feature names to include. Default is None.
        labels (list, optional): The list of label names to include. Default is None.
        batch_size (int, optional): The batch size. Default is 1.
        shuffle(bool, optional): Whether to shuffle the dataset. Default is False.
        n_labels (int, optional): The number of labels. Default is 1.
        num_parallel_calls (int, optional): The number of parallel calls. Default is tf.data.experimental.AUTOTUNE.
        drop_remainder (bool, optional): Whether to drop the remainder of batches. Default is False.
        cache (bool, optional): Whether to cache the dataset. Default is False.

        Returns:
        tf.data.Dataset: The TFRecord dataset.
        """
        logging.info(f"Loading dataset from {pattern}")

        parser = partial(DataProcessor.parse_tfrecord_multi_label, patch_size=patch_size, features=features, labels=labels)
        tupler = partial(DataProcessor.to_tuple_multi_label, depth=n_classes)

        dataset = tf.data.Dataset.list_files(pattern).interleave(DataProcessor.create_tfrecord_from_file)
        dataset = dataset.map(parser, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(tupler, num_parallel_calls=tf.data.AUTOTUNE)
        # dataset = dataset.cache()
        dataset = dataset.shuffle(512)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
