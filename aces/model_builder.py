# -*- coding: utf-8 -*-

"""
model_builder.py: Model Builder Class for Creating and Compiling Neural Network Models

This module provides a `ModelBuilder` class that is responsible for creating and compiling neural network models.
It includes methods for building and compiling models of different types, such as DNN, CNN, and U-Net, based on the
provided specifications. The class also contains utility methods for constructing custom layers and defining metrics.
"""

import tensorflow as tf
from tensorflow import keras

from aces.metrics import Metrics
from aces.remote_sensing import RemoteSensingFeatures as rs


class ModelBuilder:
    """
    ModelBuilder Class for Creating and Compiling Neural Network Models

    This class provides methods for building and compiling neural network models of different types, such as DNN, CNN, and U-Net,
    based on the provided specifications. It includes utility methods for constructing custom layers and defining metrics.

    Attributes:
        in_size (int): The input size of the models.
        out_classes (int): The number of output classes for classification models.
        optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for model compilation.
        loss (tf.keras.losses.Loss): The loss function to use for model compilation.

    Methods:
        build_model(model_type, **kwargs):
            Builds and compiles a neural network model based on the provided model type.
        build_and_compile_dnn_model(**kwargs):
            Builds and compiles a Deep Neural Network (DNN) model.
        build_and_compile_cnn_model(**kwargs):
            Builds and compiles a Convolutional Neural Network (CNN) model.
        build_and_compile_unet_model(**kwargs):
            Builds and compiles a U-Net model.
        _build_and_compile_unet_model(**kwargs):
            Helper method for building and compiling a U-Net model.
    """

    def __init__(self, in_size, out_classes, optimizer, loss):
        """
        Initialize ModelBuilder with input size, output classes, optimizer, and loss.

        Args:
            in_size (int): The input size of the models.
            out_classes (int): The number of output classes for classification models.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for model compilation.
            loss (tf.keras.losses.Loss): The loss function to use for model compilation.
        """
        self.in_size = in_size
        self.out_classes = out_classes
        self.optimizer = optimizer
        self.loss = loss

    def build_model(self, model_type, **kwargs):
        """
        Builds and compiles a neural network model based on the provided model type.

        Args:
            model_type (str): The type of the model to build ("dnn", "cnn", "unet").
            **kwargs: Additional keyword arguments specific to the model type.

        Returns:
            keras.Model: The compiled neural network model.

        Raises:
            ValueError: If an invalid model type is provided.
        """
        if model_type == "dnn":
            return self.build_and_compile_dnn_model(**kwargs)
        elif model_type == "cnn":
            return self.build_and_compile_cnn_model(**kwargs)
        elif model_type == "unet":
            return self.build_and_compile_unet_model(**kwargs)
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    def build_and_compile_dnn_model(self, **kwargs):
        """
        Builds and compiles a Deep Neural Network (DNN) model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled DNN model.
        """
        FINAL_ACTIVATION = kwargs.get("FINAL_ACTIVATION", "sigmoid")
        INITIAL_BIAS = kwargs.get("INITIAL_BIAS", None)
        # DNN_DURING_ONLY = kwargs.get("DURING_ONLY", False)

        if INITIAL_BIAS is not None:
            INITIAL_BIAS = tf.keras.initializers.Constant(INITIAL_BIAS)

        inputs = keras.Input(shape=(None, self.in_size), name="input_layer")

        input_features = rs.concatenate_features_for_dnn(inputs)

        # Create a custom input layer that accepts 4 channels
        y = keras.layers.Conv1D(64, 3, activation="relu", padding="same", name="conv1")(input_features)
        y = keras.layers.MaxPooling1D(2, padding="same")(y)
        y = keras.layers.Conv1D(32, 3, activation="relu", padding="same", name="conv2")(y)
        y = keras.layers.MaxPooling1D(2, padding="same")(y)
        y = keras.layers.Conv1D(self.in_size, 2, activation="relu", padding="same", name="conv4")(y)

        all_inputs = keras.layers.concatenate([inputs, y])

        x = keras.layers.Dense(256, activation="relu")(all_inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dropout(0.2)(x)
        output = keras.layers.Dense(self.out_classes, activation=FINAL_ACTIVATION, bias_initializer=INITIAL_BIAS)(x)

        model = keras.models.Model(inputs=inputs, outputs=output)
        metrics_list = [
            Metrics.precision(),
            Metrics.recall(),
            keras.metrics.categorical_accuracy,
            Metrics.dice_coef,
            Metrics.f1_m,
            Metrics.one_hot_io_u(self.out_classes),
        ]

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
        return model

    def build_and_compile_cnn_model(self, **kwargs):
        """
        Builds and compiles a Convolutional Neural Network (CNN) model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled CNN model.
        """
        print(kwargs)
        inputs = keras.Input(shape=(kwargs.get("PATCH_SHAPE", 128)[0], kwargs.get("PATCH_SHAPE", 128)[0], self.in_size))
        x = keras.layers.Conv2D(32, 3, activation="relu", name="convd-1", padding="same")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.15)(x)
        x = keras.layers.Conv2D(64, 3, activation="relu", name="convd-2", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.15)(x)
        x = keras.layers.Conv2D(128, 3, activation="relu", name="convd-3", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.15)(x)
        x = keras.layers.Conv2D(128, 3, activation="relu", name="convd-4", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.15)(x)
        outputs = keras.layers.Conv2D(self.out_classes, (1, 1), activation="softmax", name="final_conv")(x)
        model = keras.Model(inputs, outputs, name="cnn_model")

        metrics_list = [
            Metrics.precision(),
            Metrics.recall(),
            keras.metrics.Accuracy(),
            Metrics.one_hot_io_u(self.out_classes),
        ]

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
        return model

    def build_and_compile_unet_model(self, **kwargs):
        """
        Builds and compiles a U-Net model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled U-Net model.
        """
        if len(kwargs.get("physical_devices")) > 0:
            with tf.distribute.MirroredStrategy().scope():
                return self._build_and_compile_unet_model(**kwargs)
        else:
            print("No distributed strategy found.")
            return self._build_and_compile_unet_model(**kwargs)

    def _build_and_compile_unet_model(self, **kwargs):
        """
        Helper method for building and compiling a U-Net model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled U-Net model.
        """
        inputs = keras.Input(shape=(None, None, self.in_size))

        input_features = rs.concatenate_features_for_cnn(inputs)

        y = keras.layers.Conv2D(64, 3, activation="relu", padding="same", name="conv1")(input_features)
        y = keras.layers.Conv2D(32, 3, activation="relu", padding="same", name="conv2")(y)
        y = keras.layers.Conv2D(self.in_size, 3, activation="relu", padding="same", name="conv4")(y)

        all_inputs = keras.layers.concatenate([inputs, y]) # inputs # 

        x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(all_inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        previous_block_activation = x

        l2_regularizer = keras.regularizers.l2(0.001)

        for filters in [64, 128]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv2D(filters, 3, padding="same", depthwise_initializer="he_normal",
                                              bias_initializer="he_normal", depthwise_regularizer=l2_regularizer)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv2D(filters, 3, padding="same", depthwise_initializer="he_normal",
                                              bias_initializer="he_normal", depthwise_regularizer=l2_regularizer)(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same", bias_initializer="he_normal",
                                           kernel_initializer="he_normal", bias_regularizer=l2_regularizer,
                                           kernel_regularizer=l2_regularizer)(
                previous_block_activation
            )
            x = keras.layers.add([x, residual])
            previous_block_activation = x

        for filters in [128, 64, 32]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.Conv2DTranspose(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.UpSampling2D(2)(x)

            residual = keras.layers.UpSampling2D(2)(previous_block_activation)
            residual = keras.layers.Conv2D(filters, 1, padding="same")(residual)
            x = keras.layers.add([x, residual])
            previous_block_activation = x

        outputs = keras.layers.Conv2D(self.out_classes, 3, activation=kwargs.get("ACTIVATION_FN"), padding="same", name="final_conv")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="unet")

        metrics_list = [
            Metrics.precision(),
            Metrics.recall(),
            keras.metrics.Accuracy(),
            Metrics.one_hot_io_u(self.out_classes),
        ]

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
        return model
