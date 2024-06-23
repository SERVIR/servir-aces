# -*- coding: utf-8 -*-

import re, os
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import tensorflow as tf

__all__ = ["TFUtils", "Utils"]

class TFUtils:
    @staticmethod
    def beam_serialize(patch: np.ndarray) -> bytes:
        features = {
            name: tf.train.Feature(
                float_list=tf.train.FloatList(value=patch[name].flatten())
            )
            for name in patch.dtype.names
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()

    @staticmethod
    def configure_memory_growth() -> List:
        """
        Configure TensorFlow to allocate GPU memory dynamically.

        If GPUs are found, this method enables memory growth for each GPU.
        """
        physical_devices = tf.config.list_physical_devices("GPU")
        # self.config.physical_devices = physical_devices
        if len(physical_devices):
            print(f" > Found {len(physical_devices)} GPUs")
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                return physical_devices
            except Exception as err:
                print(err)
        else:
            print(" > No GPUs found")
            return []


class Utils:
    """
    Utils: Utility Functions for ACES

    This class provides utility functions for plotting, splitting data.
    """
    @staticmethod
    def split_dataset(element, num_partitions: int, validation_ratio: float = 0.2, test_ratio: float = 0.2) -> int:
        import random
        weights = [1 - validation_ratio - test_ratio, validation_ratio, test_ratio]
        return random.choices([0, 1, 2], weights)[0]

    @staticmethod
    def plot_metrics(metrics, history, epoch, model_save_dir):
        """
        Plot the training and validation metrics over epochs.

        Args:
            metrics: List of metrics to plot.
            history: Training history containing metric values.
            epoch: Number of epochs.
            model_save_dir: Directory to save the plot.

        Returns:
            None.
        """
        fig, ax = plt.subplots(nrows=len(metrics), sharex=True, figsize=(15, len(metrics) * 6))
        colors = ["#1f77b4", "#ff7f0e", "red", "green", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
        for i, metric in enumerate(metrics):
            try:
                ax[i].plot(history[metric], color=colors[i], label=f"Training {metric.upper()}")
                ax[i].plot(history[f"val_{metric}"], linestyle=":", marker="o", markersize=3, color=colors[i], label=f"Validation {metric.upper()}")
                ax[i].set_ylabel(metric.upper())
                ax[i].legend()
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Skipping {metric}.")
                continue

        ax[i].set_xticks(range(1, epoch + 1, 4))
        ax[i].set_xticklabels(range(1, epoch + 1, 4))
        ax[i].set_xlabel("Epoch")
        fig.savefig(f"{model_save_dir}/training.png", dpi=1000)

    @staticmethod
    def filter_good_patches(patch):
        """
        Filter patches to remove those with NaN or infinite values.

        Parameters:
        patch (np.ndarray): The patch to filter.

        Returns:
        bool: True if the patch has no NaN or infinite values, False otherwise.
        """
        # the getdownload url has field names so we"re using view here
        has_nan = np.isnan(np.sum(patch.view(np.float32)))
        has_inf = np.isinf(np.sum(patch.view(np.float32)))
        if has_nan or has_inf:
            return False
        return True

    @staticmethod
    def convert_camel_to_snake(strings):
        converted_strings = []
        for string in strings:
            converted_string = re.sub(r"(?<!^)(?=[A-Z])", "_", string).lower()
            converted_strings.append(converted_string)
        return converted_strings

    @staticmethod
    def parse_params(feature_name):
        # Fetch the feature_name string from the environment
        params = os.getenv(feature_name)

        # Normalize the string by replacing newlines with commas and stripping unwanted spaces
        params = params.replace("\n", ",").replace(" ", "")

        # Split the string into a list by commas
        params_list = params.split(",")

        # Optionally, you can remove any empty strings that may occur in the list
        params_list = [feature for feature in params_list if feature]

        return params_list
