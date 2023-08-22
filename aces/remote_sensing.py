# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras


class RemoteSensingFeatures:
    """
    A class for generating remote sensing features using TensorFlow.

    This class provides static methods to compute various remote sensing indices and concatenate them into feature tensors.
    """
    @staticmethod
    def normalized_difference(c1: tf.Tensor, c2: tf.Tensor, name: str = "nd") -> tf.Tensor:
        """
        Compute the normalized difference index between two spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the normalized difference index.

        """
        nd_f = keras.layers.Lambda(lambda x: ((x[0] - x[1]) / (x[0] + x[1])), name=name)([c1, c2])
        nd_inf = keras.layers.Lambda(lambda x: (x[0] - x[1]), name=f"{name}_inf")([c1, c2])
        return tf.where(tf.math.is_finite(nd_f), nd_f, nd_inf, name=name)

    @staticmethod
    def evi(c1: tf.Tensor, c2: tf.Tensor, c3: tf.Tensor, name: str = "evi") -> tf.Tensor:
        """
        Compute the enhanced vegetation index (EVI) using three spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            c3: A TensorFlow tensor representing the third spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the EVI.

        """
        _evi = keras.layers.Lambda(lambda x: 2.5 * ((x[0] - x[1]) / (x[0] + 6 * x[1] - 7.5 * x[2] + 1)), name=name)([c1, c2, c3])
        return _evi

    @staticmethod
    def savi(c1: tf.Tensor, c2: tf.Tensor, name: str = "savi") -> tf.Tensor:
        """
        Compute the soil-adjusted vegetation index (SAVI) between two spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the SAVI.

        """
        savi_f = keras.layers.Lambda(lambda x: ((x[0] - x[1]) / (x[0] + x[1] + 0.5)) * 1.5, name=name)([c1, c2])
        return savi_f

    @staticmethod
    def msavi(c1: tf.Tensor, c2: tf.Tensor, name: str = "msavi") -> tf.Tensor:
        """
        Compute the modified soil-adjusted vegetation index (MSAVI) between two spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the MSAVI.

        """
        msavi_f = keras.layers.Lambda(lambda x: (((2 * x[0] + 1) - tf.sqrt(((2 * x[0] + 1) * (2 * x[0] + 1)) - 8 * (x[0] - x[1]))) / 2), name=name)([c1, c2])
        return msavi_f

    @staticmethod
    def mtvi2(c1: tf.Tensor, c2: tf.Tensor, c3: tf.Tensor, name: str = "mtvi2") -> tf.Tensor:
        """
        Compute the modified transformed vegetation index 2 (MTVI2) using three spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            c3: A TensorFlow tensor representing the third spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the MTVI2.

        """
        mtvi2_f = keras.layers.Lambda(lambda x: (1.5 * (1.2 * (x[0] - x[2]) - 2.5 * (x[1] - x[2]))) / (tf.sqrt(((2 * x[0] + 1) * (2 * x[0] + 1)) - (6 * x[0] - 5 * tf.sqrt(x[1])) - 0.5)), name=name)([c1, c2, c3])
        return mtvi2_f

    @staticmethod
    def vari(c1: tf.Tensor, c2: tf.Tensor, c3: tf.Tensor, name: str = "vari") -> tf.Tensor:
        """
        Compute the visible atmospheric resistant index (VARI) using three spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            c3: A TensorFlow tensor representing the third spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the VARI.

        """
        vari_f = keras.layers.Lambda(lambda x: ((x[0] - x[1]) / (x[0] + x[1] - x[2])), name=name)([c1, c2, c3])
        vari_inf = keras.layers.Lambda(lambda x: (x[0] - x[1]), name=f"{name}_inf")([c1, c2, c3])
        return tf.where(tf.math.is_finite(vari_f), vari_f, vari_inf, name=name)

    @staticmethod
    def tgi(c1: tf.Tensor, c2: tf.Tensor, c3: tf.Tensor, name: str = "tgi") -> tf.Tensor:
        """
        Compute the triangular greenness index (TGI) using three spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            c3: A TensorFlow tensor representing the third spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the TGI.

        """
        tgi_f = keras.layers.Lambda(lambda x: ((120 * (x[1] - x[2])) - (190 * (x[1] - x[0]))) / 2, name=name)([c1, c2, c3])
        return tgi_f

    @staticmethod
    def ratio(c1: tf.Tensor, c2: tf.Tensor, name: str = "ratio") -> tf.Tensor:
        """
        Compute the ratio between two spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the ratio between the spectral bands.

        """
        ratio_f = keras.layers.Lambda(lambda x: x[0] / x[1], name=name)([c1, c2])
        ratio_inf = keras.layers.Lambda(lambda x: x[0], name=f"{name}_inf")([c1, c2])
        return tf.where(tf.math.is_finite(ratio_f), ratio_f, ratio_inf, name=name)

    @staticmethod
    def nvi(c1: tf.Tensor, c2: tf.Tensor, name: str = "nvi") -> tf.Tensor:
        """
        Compute the normalized vegetation index (NVI) between two spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the NVI.

        """
        nvi_f = keras.layers.Lambda(lambda x: x[0] / (x[0] + x[1]), name=name)([c1, c2])
        nvi_inf = keras.layers.Lambda(lambda x: x[0], name=f"{name}_inf")([c1, c2])
        return tf.where(tf.math.is_finite(nvi_f), nvi_f, nvi_inf, name=name)

    @staticmethod
    def diff_band(c1: tf.Tensor, c2: tf.Tensor, name: str = "diff") -> tf.Tensor:
        """
        Compute the difference between two spectral bands.

        Args:
            c1: A TensorFlow tensor representing the first spectral band.
            c2: A TensorFlow tensor representing the second spectral band.
            name: A string specifying the name for the operation.

        Returns:
            A TensorFlow tensor representing the difference between the spectral bands.

        """
        diff = keras.layers.Lambda(lambda x: x[0] - x[1], name=name)([c1, c2])
        return diff

    @staticmethod
    def concatenate_features_for_cnn(input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Concatenate remote sensing features for Convolutional Neural Network (CNN) input.

        Args:
            input_tensor: A TensorFlow tensor representing the input remote sensing data.

        Returns:
            A TensorFlow tensor representing the concatenated features for CNN input.

        """
        red_before = input_tensor[:, :, :, 0:1]
        green_before = input_tensor[:, :, :, 1:2]
        blue_before = input_tensor[:, :, :, 2:3]
        nir_before = input_tensor[:, :, :, 3:4]
        red_during = input_tensor[:, :, :, 4:5]
        green_during = input_tensor[:, :, :, 5:6]
        blue_during = input_tensor[:, :, :, 6:7]
        nir_during = input_tensor[:, :, :, 7:8]

        ndvi_before = RemoteSensingFeatures.normalized_difference(nir_before, red_before, name="ndvi_before")
        ndvi_during = RemoteSensingFeatures.normalized_difference(nir_during, red_during, name="ndvi_during")
        evi_before = RemoteSensingFeatures.evi(nir_before, red_before, blue_before, name="evi_before")
        evi_during = RemoteSensingFeatures.evi(nir_during, red_during, blue_during, name="evi_during")
        ndwi_before = RemoteSensingFeatures.normalized_difference(green_before, nir_before, name="ndwi_before")
        ndwi_during = RemoteSensingFeatures.normalized_difference(green_during, nir_during, name="ndwi_during")
        savi_before = RemoteSensingFeatures.savi(nir_before, red_before, name="savi_before")
        savi_during = RemoteSensingFeatures.savi(nir_during, red_during, name="savi_during")
        # msavi_before = RemoteSensingFeatures.msavi(nir_before, red_before, name="msavi_before")
        # msavi_during = RemoteSensingFeatures.msavi(nir_during, red_during, name="msavi_during")
        mtvi2_before = RemoteSensingFeatures.mtvi2(nir_before, red_before, green_before, name="mtvi2_before")
        mtvi2_during = RemoteSensingFeatures.mtvi2(nir_during, red_during, green_during, name="mtvi2_during")
        # vari is not used because the computation gave some nan values
        # vari_before = RemoteSensingFeatures.vari(green_before, red_before, blue_before, name="vari_before")
        # vari_during = RemoteSensingFeatures.vari(green_during, red_during, blue_during, name="vari_during")
        # tgi_before = RemoteSensingFeatures.tgi(green_before, red_before, blue_before, name="tgi_before")
        # tgi_during = RemoteSensingFeatures.tgi(green_during, red_during, blue_during, name="tgi_during")

        red_diff1 = RemoteSensingFeatures.diff_band(red_before, red_during, name="diff1")
        green_diff1 = RemoteSensingFeatures.diff_band(green_before, green_during, name="diff2")
        blue_diff1 = RemoteSensingFeatures.diff_band(blue_before, blue_during, name="diff3")
        nir_diff1 = RemoteSensingFeatures.diff_band(nir_before, nir_during, name="diff4")

        return keras.layers.concatenate(
            [input_tensor, ndvi_before, ndvi_during, evi_before, evi_during, ndwi_before, ndwi_during, savi_before, savi_during,
             mtvi2_before, mtvi2_during,
             red_diff1, green_diff1, blue_diff1, nir_diff1],
            name="input_features"
        )

    @staticmethod
    def concatenate_features_for_dnn(input_tensor: tf.Tensor) -> tf.Tensor:
        """
        Concatenate remote sensing features for Deep Neural Network (DNN) input.

        Args:
            input_tensor: A TensorFlow tensor representing the input remote sensing data.

        Returns:
            A TensorFlow tensor representing the concatenated features for DNN input.

        """
        red_before = input_tensor[:, :, :, 0:1]
        green_before = input_tensor[:, :, :, 1:2]
        blue_before = input_tensor[:, :, :, 2:3]
        nir_before = input_tensor[:, :, :, 3:4]
        red_during = input_tensor[:, :, :, 4:5]
        green_during = input_tensor[:, :, :, 5:6]
        blue_during = input_tensor[:, :, :, 6:7]
        nir_during = input_tensor[:, :, :, 7:8]

        ndvi_before = RemoteSensingFeatures.normalized_difference(nir_before, red_before, name="ndvi_before")
        ndvi_during = RemoteSensingFeatures.normalized_difference(nir_during, red_during, name="ndvi_during")
        evi_before = RemoteSensingFeatures.evi(nir_before, red_before, blue_before, name="evi_before")
        evi_during = RemoteSensingFeatures.evi(nir_during, red_during, blue_during, name="evi_during")
        ndwi_before = RemoteSensingFeatures.normalized_difference(green_before, nir_before, name="ndwi_before")
        ndwi_during = RemoteSensingFeatures.normalized_difference(green_during, nir_during, name="ndwi_during")
        savi_before = RemoteSensingFeatures.savi(nir_before, red_before, name="savi_before")
        savi_during = RemoteSensingFeatures.savi(nir_during, red_during, name="savi_during")
        msavi_before = RemoteSensingFeatures.msavi(nir_before, red_before, name="msavi_before")
        msavi_during = RemoteSensingFeatures.msavi(nir_during, red_during, name="msavi_during")
        mtvi2_before = RemoteSensingFeatures.mtvi2(nir_before, red_before, green_before, name="mtvi2_before")
        mtvi2_during = RemoteSensingFeatures.mtvi2(nir_during, red_during, green_during, name="mtvi2_during")
        vari_before = RemoteSensingFeatures.vari(green_before, red_before, blue_before, name="vari_before")
        vari_during = RemoteSensingFeatures.vari(green_during, red_during, blue_during, name="vari_during")
        tgi_before = RemoteSensingFeatures.tgi(green_before, red_before, blue_before, name="tgi_before")
        tgi_during = RemoteSensingFeatures.tgi(green_during, red_during, blue_during, name="tgi_during")

        return keras.layers.concatenate(
            [input_tensor, ndvi_before, ndvi_during, evi_before, evi_during, ndwi_before, ndwi_during, savi_before, savi_during,
             msavi_before, msavi_during, mtvi2_before, mtvi2_during, vari_before, vari_during, tgi_before, tgi_during],
            name="input_features"
        )

    @staticmethod
    def derive_features_for_dnn(features_dict: dict, added_features: list = []) -> dict:
        """
        Concatenate remote sensing features for Deep Neural Network (DNN) input.

        Args:
            input_tensor: A TensorFlow tensor representing the input remote sensing data.

        Returns:
            A TensorFlow tensor representing the concatenated features for DNN input.

        """
        red_before = features_dict["red_before"]
        green_before = features_dict["green_before"]
        blue_before = features_dict["blue_before"]
        nir_before = features_dict["nir_before"]
        red_during = features_dict["red_during"]
        green_during = features_dict["green_during"]
        blue_during = features_dict["blue_during"]
        nir_during = features_dict["nir_during"]

        feature_col = {
            "ndvi_before": RemoteSensingFeatures.normalized_difference(nir_before, red_before, name="ndvi_before"),
            "ndvi_during": RemoteSensingFeatures.normalized_difference(nir_during, red_during, name="ndvi_during"),
            "evi_before": RemoteSensingFeatures.evi(nir_before, red_before, blue_before, name="evi_before"),
            "evi_during": RemoteSensingFeatures.evi(nir_during, red_during, blue_during, name="evi_during"),
            "ndwi_before": RemoteSensingFeatures.normalized_difference(green_before, nir_before, name="ndwi_before"),
            "ndwi_during": RemoteSensingFeatures.normalized_difference(green_during, nir_during, name="ndwi_during"),
            "savi_before": RemoteSensingFeatures.savi(nir_before, red_before, name="savi_before"),
            "savi_during": RemoteSensingFeatures.savi(nir_during, red_during, name="savi_during"),
            "msavi_before": RemoteSensingFeatures.msavi(nir_before, red_before, name="msavi_before"),
            "msavi_during": RemoteSensingFeatures.msavi(nir_during, red_during, name="msavi_during"),
            "mtvi2_before": RemoteSensingFeatures.mtvi2(nir_before, red_before, green_before, name="mtvi2_before"),
            "mtvi2_during": RemoteSensingFeatures.mtvi2(nir_during, red_during, green_during, name="mtvi2_during"),
            "vari_before": RemoteSensingFeatures.vari(green_before, red_before, blue_before, name="vari_before"),
            "vari_during": RemoteSensingFeatures.vari(green_during, red_during, blue_during, name="vari_during"),
            "tgi_before": RemoteSensingFeatures.tgi(green_before, red_before, blue_before, name="tgi_before"),
            "tgi_during": RemoteSensingFeatures.tgi(green_during, red_during, blue_during, name="tgi_during"),
        }

        for feature in added_features:
            features_dict[feature] = feature_col[feature]

        return features_dict
