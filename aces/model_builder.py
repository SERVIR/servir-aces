# -*- coding: utf-8 -*-

"""
model_builder.py:

Model Builder Class for Creating and Compiling Neural Network Models

This module provides a `ModelBuilder` class that is responsible for creating and compiling neural network models.
It includes methods for building and compiling models of different types, such as Deep Neural Network (**DNN**),
Convolutional Neural Network (**CNN**), and **U-Net**, based on the
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

    def __init__(self, features, out_classes, optimizer, loss):
        """
        Initialize ModelBuilder with input size, output classes, optimizer, and loss.

        Args:
            in_size (int): The input size of the models.
            out_classes (int): The number of output classes for classification models.
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to use for model compilation.
            loss (tf.keras.losses.Loss): The loss function to use for model compilation.
        """
        self.features = features
        self.in_size = len(features)
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
        FOR_AI_PLATFORM = kwargs.get("FOR_AI_PLATFORM", False)
        if model_type == "dnn":
            if FOR_AI_PLATFORM:
                return self.build_and_compile_dnn_model_for_ai_platform(**kwargs)
            else:
                return self.build_and_compile_dnn_model(**kwargs)
        elif model_type == "cnn":
            return self.build_and_compile_cnn_model(**kwargs)
        elif model_type == "unet":
            if FOR_AI_PLATFORM:
                return self.build_and_compile_unet_model_for_ai_platform(**kwargs)
            else:
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
        INITIAL_BIAS = kwargs.get("INITIAL_BIAS", None)
        print(f"INITIAL_BIAS: {INITIAL_BIAS}")

        if INITIAL_BIAS is not None:
            INITIAL_BIAS = tf.keras.initializers.Constant(INITIAL_BIAS)
        else:
            INITIAL_BIAS = "zeros"

        inputs = keras.Input(shape=(None, self.in_size), name="input_layer")

        x = keras.layers.Dense(256, activation="relu")(inputs)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE", 0.))(x)
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE", 0.))(x)
        x = keras.layers.Dense(64, activation="relu")(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE", 0.))(x)
        x = keras.layers.Dense(32, activation="relu")(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE", 0.))(x)
        output = keras.layers.Dense(self.out_classes, activation=kwargs.get("ACTIVATION_FN"), bias_initializer=INITIAL_BIAS)(x)

        model = keras.models.Model(inputs=inputs, outputs=output)
        metrics_list = [
            Metrics.precision(),
            Metrics.recall(),
            keras.metrics.CategoricalAccuracy(),
            Metrics.one_hot_io_u(self.out_classes),
        ]

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
        return model


    def build_and_compile_dnn_model_for_ai_platform(self, **kwargs):
        """
        Builds and compiles a Deep Neural Network (DNN) model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled DNN model.
        """
        INITIAL_BIAS = kwargs.get("INITIAL_BIAS", None)
        print(f"INITIAL_BIAS: {INITIAL_BIAS}")
        # DNN_DURING_ONLY = kwargs.get("DURING_ONLY", False)

        DERIVE_FEATURES = kwargs.get("DERIVE_FEATURES", False)

        if INITIAL_BIAS is not None:
            INITIAL_BIAS = tf.keras.initializers.Constant(INITIAL_BIAS)
        else:
            INITIAL_BIAS = "zeros"

        inputs_main = keras.Input(shape=(None, None, self.in_size), name="input_layer")

        if DERIVE_FEATURES:
            inputs = rs.concatenate_features_for_dnn(inputs_main)
        else:
            inputs = inputs_main

        x = keras.layers.Conv2D(256, (1, 1), activation="relu")(inputs)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE"), 0.)(x)
        x = keras.layers.Conv2D(128, (1, 1), activation="relu")(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE"), 0.)(x)
        x = keras.layers.Conv2D(64, (1, 1), activation="relu")(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE"), 0.)(x)
        x = keras.layers.Conv2D(32, (1, 1), activation="relu")(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE"), 0.)(x)
        output = keras.layers.Conv2D(self.out_classes, (1, 1), activation=kwargs.get("ACTIVATION_FN"), bias_initializer=INITIAL_BIAS)(x)

        model = keras.models.Model(inputs=inputs_main, outputs=output)

        wrapped_model = ModelWrapper(ModelPreprocess(self.features), model)

        metrics_list = [
            Metrics.precision(),
            Metrics.recall(),
            keras.metrics.CategoricalAccuracy(),
            Metrics.one_hot_io_u(self.out_classes),
        ]
        wrapped_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
        return model, wrapped_model

    def build_and_compile_cnn_model(self, **kwargs):
        """
        Builds and compiles a Convolutional Neural Network (CNN) model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled CNN model.
        """
        inputs = keras.Input(shape=(kwargs.get("PATCH_SHAPE", 128)[0], kwargs.get("PATCH_SHAPE", 128)[0], self.in_size))
        x = keras.layers.Conv2D(32, 3, activation="relu", name="convd-1", padding="same")(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE"), 0.)(x)
        x = keras.layers.Conv2D(64, 3, activation="relu", name="convd-2", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE"), 0.)(x)
        x = keras.layers.Conv2D(128, 3, activation="relu", name="convd-3", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE"), 0.)(x)
        x = keras.layers.Conv2D(128, 3, activation="relu", name="convd-4", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(kwargs.get("DROPOUT_RATE"), 0.)(x)
        outputs = keras.layers.Conv2D(self.out_classes, (1, 1), activation="softmax", name="final_conv")(x)
        model = keras.Model(inputs, outputs, name="cnn_model")

        metrics_list = [
            Metrics.precision(),
            Metrics.recall(),
            keras.metrics.CategoricalAccuracy(),
            Metrics.one_hot_io_u(self.out_classes),
        ]

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
        return model

    def build_and_compile_unet_model_for_ai_platform(self, **kwargs):
        """
        Builds and compiles a U-Net model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled U-Net model.
        """
        if len(kwargs.get("physical_devices")) > 0:
            with tf.distribute.MirroredStrategy().scope():
                return self._build_and_compile_unet_model_for_ai_plaform(**kwargs)
        else:
            print("No distributed strategy found.")
            return self._build_and_compile_unet_model(**kwargs)

    def _build_and_compile_unet_model_for_ai_plaform(self, **kwargs):
        """
        Helper method for building and compiling a U-Net model.
        """
        model = self._build_and_compile_unet_model(**kwargs)
        wrapped_model = ModelWrapper(ModelPreprocess(self.features), model)
        metrics_list = [
            Metrics.precision(),
            Metrics.recall(),
            keras.metrics.CategoricalAccuracy(),
            Metrics.one_hot_io_u(self.out_classes),
        ]
        wrapped_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
        return model, wrapped_model

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
                model = self._build_and_compile_unet_model(**kwargs)
                metrics_list = [
                    Metrics.precision(),
                    Metrics.recall(),
                    keras.metrics.CategoricalAccuracy(),
                    Metrics.one_hot_io_u(self.out_classes),
                ]
                model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
                return model
        else:
            print("No distributed strategy found.")
            model = self._build_and_compile_unet_model(**kwargs)
            metrics_list = [
                Metrics.precision(),
                Metrics.recall(),
                keras.metrics.CategoricalAccuracy(),
                Metrics.one_hot_io_u(self.out_classes),
            ]
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=metrics_list)
            return model

    def _build_and_compile_unet_model(self, **kwargs):
        """
        Helper method for building and compiling a U-Net model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled U-Net model.
        """
        inputs = keras.Input(shape=(None, None, self.in_size))

        DERIVE_FEATURES = kwargs.get("DERIVE_FEATURES", False)

        if DERIVE_FEATURES:
            input_features = rs.concatenate_features_for_cnn(inputs)
        else:
            input_features = inputs


        x = keras.layers.Conv2D(32, 3, strides=2, padding="same")(input_features)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)

        previous_block_activation = x

        l2_regularizer = keras.regularizers.l2(0.001)

        for filters in [64, 128, 256]:
            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.Activation("relu")(x)
            x = keras.layers.SeparableConv2D(filters, 3, padding="same")(x)
            x = keras.layers.BatchNormalization()(x)

            x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

            residual = keras.layers.Conv2D(filters, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = keras.layers.add([x, residual])
            previous_block_activation = x

        for filters in [256, 128, 64, 32]:
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

        return model

    def _build_and_compile_vanilla_unet_model(self, **kwargs):
        """
        Helper method for building and compiling a U-Net model.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            keras.Model: The compiled U-Net model.
        """
        inputs = keras.Input(shape=(None, None, self.in_size))

        DERIVE_FEATURES = kwargs.get("DERIVE_FEATURES", False)

        print(f"DERIVE_FEATURES: {DERIVE_FEATURES}")

        if DERIVE_FEATURES:
            input_features = rs.concatenate_features_for_cnn(inputs)
        else:
            input_features = inputs

        c1 = keras.layers.Conv2D(16, (3, 3), padding="same")(input_features)
        c1 = keras.layers.BatchNormalization()(c1)
        c1 = keras.layers.Activation("relu")(c1)
        p1 = keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = keras.layers.Conv2D(32, (3, 3), padding="same")(p1)
        c2 = keras.layers.BatchNormalization()(c2)
        c2 = keras.layers.Activation("relu")(c2)
        p2 = keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = keras.layers.Conv2D(64, (3, 3), padding="same")(p2)
        c3 = keras.layers.BatchNormalization()(c3)
        c3 = keras.layers.Activation("relu")(c3)
        p3 = keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = keras.layers.Conv2D(128, (3, 3), padding="same")(p3)
        c4 = keras.layers.BatchNormalization()(c4)
        c4 = keras.layers.Activation("relu")(c4)
        p4 = keras.layers.MaxPooling2D((2, 2))(c4)

        c5 = keras.layers.Conv2D(256, (3, 3), padding="same")(p4)
        c5 = keras.layers.BatchNormalization()(c5)
        c5 = keras.layers.Activation("relu")(c5)

        # Decoder
        u6 = keras.layers.UpSampling2D((2, 2))(c5)
        u6 = keras.layers.Conv2D(128, (3, 3), padding="same")(u6)
        u6 = keras.layers.BatchNormalization()(u6)
        u6 = keras.layers.Activation("relu")(u6)
        u6 = keras.layers.Add()([u6, c4])

        u7 = keras.layers.UpSampling2D((2, 2))(u6)
        u7 = keras.layers.Conv2D(64, (3, 3), padding="same")(u7)
        u7 = keras.layers.BatchNormalization()(u7)
        u7 = keras.layers.Activation("relu")(u7)
        u7 = keras.layers.Add()([u7, c3])

        u8 = keras.layers.UpSampling2D((2, 2))(u7)
        u8 = keras.layers.Conv2D(32, (3, 3), padding="same")(u8)
        u8 = keras.layers.BatchNormalization()(u8)
        u8 = keras.layers.Activation("relu")(u8)
        u8 = keras.layers.Add()([u8, c2])

        u9 = keras.layers.UpSampling2D((2, 2))(u8)
        u9 = keras.layers.Conv2D(16, (3, 3), padding="same")(u9)
        u9 = keras.layers.BatchNormalization()(u9)
        u9 = keras.layers.Activation("relu")(u9)
        u9 = keras.layers.Add()([u9, c1])

        # Output Layer
        output_layer = keras.layers.Conv2D(self.out_classes, 3, activation=kwargs.get("ACTIVATION_FN"), padding="same", name="final_conv")(u9)

        # Build Model
        model = keras.Model(inputs=inputs, outputs=output_layer, name="vanilla_unet")

        return model



class AddExtraFeatures(tf.keras.layers.Layer):
    def __init__(self, added_features):
        super().__init__()
        self.added_features = added_features

    def call(self, features_dict, labels):
        features_dict = rs.derive_features_for_dnn(features_dict, self.added_features)
        return features_dict, labels



# A Layer to stack and reshape the input tensors.
class ModelPreprocess(keras.layers.Layer):
    def __init__(self, in_features, **kwargs):
        self.in_features = in_features
        super(ModelPreprocess, self).__init__(**kwargs)

    def call(self, features_dict):
        # (None, 1, 1, 1) -> (None, 1, 1, P) for dnn
        # (None, H, W, 1) -> (None, H, W, P) for cnn/unet
        return tf.concat([features_dict[b] for b in self.in_features], axis=3)

    def get_config(self):
        config = super().get_config()
        return config


# A Model that wraps the base model with the preprocessing layer.
class ModelWrapper(keras.Model):
    def __init__(self, preprocessing, backbone, **kwargs):
        super().__init__(**kwargs)
        self.preprocessing = preprocessing
        self.backbone = backbone

    def call(self, features_dict):
        x = self.preprocessing(features_dict)
        return self.backbone(x)

    def get_config(self):
        config = super().get_config()
        return config

@tf.autograph.experimental.do_not_convert
class DeSerializeInput(tf.keras.layers.Layer):
    def __init__(self, in_features, **kwargs):
        self.in_features = in_features
        super().__init__(**kwargs)

    def call(self, inputs_dict):
        deserialize = {
            k: tf.map_fn(lambda x: tf.io.parse_tensor(x, tf.float32),
                         tf.io.decode_base64(v),
                         fn_output_signature=tf.float32)
            for (k, v) in inputs_dict.items() if k in self.in_features
        }
        return deserialize

    def get_config(self):
        config = super().get_config()
        return config


@tf.autograph.experimental.do_not_convert
class ReSerializeOutput(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, output_tensor, name):
        if not name: name = "output"
        return {name: tf.identity(tf.map_fn(
            lambda x: tf.io.encode_base64(tf.io.serialize_tensor(x)),
            output_tensor, fn_output_signature=tf.string
            ),
            name=name
        )}

    def get_config(self):
        config = super().get_config()
        return config
