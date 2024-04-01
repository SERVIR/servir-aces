# -*- coding: utf-8 -*-

"""
metrics.py: Custom Metrics for Model Evaluation and Utility Functions for Model Visualization

This module provides a collection of custom metrics that can be used for evaluating model performance in tasks such as
image segmentation. This module includes a wide variety of evaluation metrics listed below.
Additionally, it contains utility functions for plotting and visualizing model metrics during training.
"""

import matplotlib.pyplot as plt

from tensorflow import keras
from keras import backend as K

__all__ = ["Metrics"]

class Metrics:
    """
    A class containing various metrics functions for model evaluation.

    This class provides a collection of static methods, each representing a different evaluation metric. These metrics can be used to
    evaluate the performance of classification or segmentation models during training or testing.

    Methods:
        precision():
            Computes the precision metric.
        recall():
            Computes the recall metric.
        f1_m():
            Computes the F1 score metric. [van1979information, chicco2020advantages]
        one_hot_io_u(num_classes):
            Computes the Intersection over Union (IoU) metric for each class. [jaccard1901etude]
        true_positives():
            Computes the true positives metric also known as sensitivity.[yerushalmy1947statistical]
        false_positives():
            Computes the false positives metric. [cohen2013statistical]
        true_negatives():
            Computes the true negatives metric also known as specificity. [yerushalmy1947statistical]
        false_negatives():
            Computes the false negatives metric. [cohen2013statistical]
        binary_accuracy():
            Computes the binary accuracy metric.
        auc():
            Computes the Area Under the Curve (AUC) metric. [provost2001robust]
        prc():
            Computes the Precision-Recall Curve (PRC) metric.
        dice_coef():
            Computes the Dice coefficient metric. [milletari2016v]
        dice_loss():
            Computes the Dice loss metric. [sudre2017generalised]
        bce_loss():
            Computes the Binary Cross-Entropy (BCE) loss metric. [zhu2018negative; yi2004automated]
        bce_dice_loss():
            Computes the BCE and Dice loss metric. cite [taghanaki2019combo]
        tversky():
            Computes the Tversky index metric.[tversky1977features]
        tversky_loss():
            Computes the Tversky loss metric. [salehi2017tversky]
        focal_tversky_loss():
            Computes the focal Tversky loss metric [abraham2019novel]
    """

    @staticmethod
    def recall_m(y_true, y_pred):
        """
        Calculate the recall metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Recall metric value.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_pred):
        """
        Calculate the precision metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Precision metric value.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def f1_m(y_true, y_pred):
        """
        Calculate the F1-score metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            F1-score metric value.
        """
        precision = Metrics.precision_m(y_true, y_pred)
        recall = Metrics.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    @staticmethod
    def dice_coef(y_true, y_pred, smooth=1):
        """
        Calculate the Dice coefficient metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            smooth: Smoothing parameter.

        Returns:
            Dice coefficient metric value.
        """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def dice_loss(y_true, y_pred, smooth=1):
        """
        Calculate the Dice loss metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            smooth: Smoothing parameter.

        Returns:
            Dice loss metric value.
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        true_sum = K.sum(K.square(y_true), -1)
        pred_sum = K.sum(K.square(y_pred), -1)
        return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        """
        Calculate the BCE-Dice loss metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            BCE-Dice loss metric value.
        """
        return Metrics.bce_loss(y_true, y_pred) + Metrics.dice_loss(y_true, y_pred)

    @staticmethod
    def bce_loss(y_true, y_pred):
        """
        Calculate the Binary Cross-Entropy (BCE) loss metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            BCE loss metric value.
        """
        return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2)

    @staticmethod
    def tversky(y_true, y_pred, smooth=1, alpha=0.7):
        """
        Calculate the Tversky index metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            smooth: Smoothing parameter.
            alpha: Weighting factor.

        Returns:
            Tversky index metric value.
        """
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    @staticmethod
    def tversky_loss(y_true, y_pred):
        """
        Calculate the Tversky loss metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Tversky loss metric value.
        """
        return 1 - Metrics.tversky(y_true, y_pred)

    @staticmethod
    def focal_tversky_loss(y_true, y_pred, gamma=0.75):
        """
        Calculate the focal Tversky loss metric.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            gamma: Focusing parameter.

        Returns:
            Focal Tversky loss metric value.
        """
        return K.pow((1 - Metrics.tversky(y_true, y_pred)), gamma)

    @staticmethod
    def true_positives():
        """
        Create a metric for counting true positives.

        Returns:
            True positives metric.
        """
        return keras.metrics.TruePositives(name="tp")

    @staticmethod
    def false_positives():
        """
        Create a metric for counting false positives.

        Returns:
            False positives metric.
        """
        return keras.metrics.FalsePositives(name="fp")

    @staticmethod
    def true_negatives():
        """
        Create a metric for counting true negatives.

        Returns:
            True negatives metric.
        """
        return keras.metrics.TrueNegatives(name="tn")

    @staticmethod
    def false_negatives():
        """
        Create a metric for counting false negatives.

        Returns:
            False negatives metric.
        """
        return keras.metrics.FalseNegatives(name="fn")

    @staticmethod
    def binary_accuracy():
        """
        Create a metric for calculating binary accuracy.

        Returns:
            Binary accuracy metric.
        """
        return keras.metrics.BinaryAccuracy(name="accuracy")

    # check difference between this and precision_m output
    @staticmethod
    def precision():
        """
        Create a metric for calculating precision.

        Returns:
            Precision metric.
        """
        return keras.metrics.Precision(name="precision")

    @staticmethod
    def recall():
        """
        Create a metric for calculating recall.

        Returns:
            Recall metric.
        """
        return keras.metrics.Recall(name="recall")

    @staticmethod
    def auc():
        """
        Create a metric for calculating Area Under the Curve (AUC).

        Returns:
            AUC metric.
        """
        return keras.metrics.AUC(name="auc")

    @staticmethod
    def prc():
        """
        Create a metric for calculating Precision-Recall Curve (PRC).

        Returns:
            PRC metric.
        """
        return keras.metrics.AUC(name="prc", curve="PR")

    @staticmethod
    def one_hot_io_u(num_classes, name="one_hot_io_u"):
        """
        Create a metric for calculating Intersection over Union (IoU) using one-hot encoding.

        Args:
            num_classes: Number of classes.
            name: Name of the metric.

        Returns:
            One-hot IoU metric.
        """
        return keras.metrics.OneHotIoU(num_classes=num_classes, target_class_ids=list(range(num_classes)), name=name)
