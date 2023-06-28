# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import backend as K


class Metrics:
    @staticmethod
    def recall_m(y_true: keras.Tensor, y_pred: keras.Tensor) -> keras.Tensor:
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true: keras.Tensor, y_pred: keras.Tensor) -> keras.Tensor:
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def f1_m(y_true: keras.Tensor, y_pred: keras.Tensor) -> keras.Tensor:
        precision = Metrics.precision_m(y_true, y_pred)
        recall = Metrics.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    @staticmethod
    def dice_coef(y_true: keras.Tensor, y_pred: keras.Tensor, smooth: float = 1) -> keras.Tensor:
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    @staticmethod
    def dice_loss(y_true: keras.Tensor, y_pred: keras.Tensor, smooth: float = 1) -> keras.Tensor:
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        true_sum = K.sum(K.square(y_true), -1)
        pred_sum = K.sum(K.square(y_pred), -1)
        return 1 - ((2. * intersection + smooth) / (true_sum + pred_sum + smooth))

    @staticmethod
    def bce_dice_loss(y_true: keras.Tensor, y_pred: keras.Tensor) -> keras.Tensor:
        return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2) + Metrics.dice_loss(y_true, y_pred)

    @staticmethod
    def bce_loss(y_true: keras.Tensor, y_pred: keras.Tensor) -> keras.Tensor:
        return keras.losses.binary_crossentropy(y_true, y_pred, label_smoothing=0.2)

    @staticmethod
    def tversky(y_true: keras.Tensor, y_pred: keras.Tensor, smooth: float = 1, alpha: float = 0.7) -> keras.Tensor:
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos)
        false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)

    @staticmethod
    def tversky_loss(y_true: keras.Tensor, y_pred: keras.Tensor) -> keras.Tensor:
        return 1 - Metrics.tversky(y_true, y_pred)

    @staticmethod
    def focal_tversky_loss(y_true: keras.Tensor, y_pred: keras.Tensor, gamma: float = 0.75) -> keras.Tensor:
        tv = Metrics.tversky(y_true, y_pred)
        return K.pow((1 - tv), gamma)

    @staticmethod
    def true_positives() -> keras.Tensor: return keras.metrics.TruePositives(name='tp')
    
    @staticmethod
    def false_positives() -> keras.Tensor: return keras.metrics.FalsePositives(name='fp')
    
    @staticmethod
    def true_negatives() -> keras.Tensor: return keras.metrics.TrueNegatives(name='tn')
    
    @staticmethod
    def false_negatives() -> keras.Tensor: return keras.metrics.FalseNegatives(name='fn')
    
    @staticmethod
    def binary_accuracy() -> keras.Tensor: return keras.metrics.BinaryAccuracy(name='accuracy')
    
    # check difference between this and precision_m output
    @staticmethod
    def precision() -> keras.Tensor: return keras.metrics.Precision(name='precision')
    
    @staticmethod
    def recall() -> keras.Tensor: return keras.metrics.Recall(name='recall')
    
    @staticmethod
    def auc() -> keras.Tensor: return keras.metrics.AUC(name='auc')
    
    @staticmethod
    def prc() -> keras.Tensor: return keras.metrics.AUC(name='prc', curve='PR')
    
    @staticmethod
    def one_hot_io_u(num_classes: int, name: str = 'one_hot_io_u') -> keras.Tensor:
        return keras.metrics.OneHotIoU(num_classes=num_classes, target_class_ids=list(range(num_classes)), name=name)


class Utils:
    @staticmethod
    def plot_metrics(metrics, history, epoch, model_save_dir):
        fig, ax = plt.subplots(nrows=len(metrics), sharex=True, figsize=(15, len(metrics) * 6))
        colors = ['#1f77b4', '#ff7f0e', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']

        for i, metric in enumerate(metrics):
            ax[i].plot(history[metric], color=colors[i], label=f'Training {metric.upper()}')
            ax[i].plot(history[f'val_{metric}'], linestyle=':', marker='o', markersize=3, color=colors[i], label=f'Validation {metric.upper()}')
            ax[i].set_ylabel(metric.upper())
            ax[i].legend()

        ax[i].set_xticks(range(1, len(epoch) + 1, 4))
        ax[i].set_xticklabels(range(1, len(epoch) + 1, 4))
        ax[i].set_xlabel('Epoch')
        fig.savefig(f"{model_save_dir}/training.png", dpi=700)
