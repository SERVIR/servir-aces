# -*- coding: utf-8 -*-

"""
ACES Configuration Module
This module provides the configuration settings for the ACES project.
"""

from pathlib import Path
import os
import ast

from apache_beam.options.pipeline_options import PipelineOptions
import tensorflow as tf
from tensorflow import keras

from aces.metrics import Metrics

from dotenv import load_dotenv
load_dotenv('.env')


class Config:
    """
    ACES Configuration Class

    This class contains the configuration settings for the ACES project.

    Attributes:
        BASEDIR (str): The base directory for data I/O information.
        DATADIR (str): The directory for data collection experiments.
        MODEL_NAME (str): The name of the ACES model.
        MODEL_CHECKPOINT_NAME (str): The name for model checkpoints.
        LABELS (list): A list of labels used in the project.
        PATCH_SHAPE (tuple): The shape of training patches.
        TRAIN_SIZE (int): The size of the training dataset.
        TEST_SIZE (int): The size of the testing dataset.
        VAL_SIZE (int): The size of the validation dataset.
        N_POS (int): The number of positive samples.
        N_NEG (int): The number of negative samples.
        BATCH_SIZE (int): The batch size for model training.
        EPOCHS (int): The number of epochs for model training.
        BUFFER_SIZE (int): The buffer size for shuffling the dataset.
        RAMPUP_EPOCHS (int): The number of ramp-up epochs.
        SUSTAIN_EPOCHS (int): The number of sustain epochs.
        USE_ADJUSTED_LR (bool): Flag to use adjusted learning rate.
        MAX_LR (float): The maximum learning rate.
        MID_LR (float): The intermediate learning rate.
        MIN_LR (float): The minimum learning rate.
        LEARNING_RATE (float): The initial learning rate.
        DROPOUT_RATE (float): The dropout rate for the model.
        CALLBACK_PARAMETER (str): The parameter used for callbacks.
        ACTIVATION_FN (str): The activation function for the model.
        OPTIMIZER (str): The optimizer used for model training.
        LOSS (str): The loss function used for model training.
        MODEL_TYPE (str): The type of model (cnn, dnn, unet).
        METRICS (str): The metrics used for model evaluation.
        OUT_CLASS_NUM (int): The number of output classes.
        PCA_COMPONENTS (int): The number of PCA components.
        EE_USER (str): The Earth Engine username.
        EE_OUPUT_ASSET (str): The Earth Engine output asset path.
        EE_SERVICE_CREDENTIALS (str): The path to Earth Engine service account credentials.
        GCS_PROJECT (str): The Google Cloud Storage project name.
        GCS_BUCKET (str): The Google Cloud Storage bucket name.
        GCS_EEIFIED_DIR (str): The directory for Earth Engineified data in GCS.
        GCS_CHECKPOINT_DIR (str): The directory for model checkpoints in GCS.
        FEATURES (str): The list of features used in the model.
    """

    physical_devices = tf.config.list_physical_devices("GPU")
    DISTRIBUTED_STRATEGY = tf.distribute.MirroredStrategy() if len(physical_devices) > 1 else None

    BASEDIR = Path(os.getenv('BASEDIR'))
    DATADIR = BASEDIR / os.getenv('DATADIR')

    TRAINING_DIR = DATADIR / 'training'
    TESTING_DIR = DATADIR / 'testing'
    VALIDATION_DIR= DATADIR / 'validation'

    OUTPUT_DIR = BASEDIR / 'output'

    MODEL_NAME = os.getenv('MODEL_NAME')
    MODEL_CHECKPOINT_NAME: str = os.getenv('MODEL_CHECKPOINT_NAME')

    METRICS = os.getenv('METRICS').split('\n')

    FEATURES = os.getenv('FEATURES').split('\n')
    LABELS = ast.literal_eval(os.getenv('LABELS'))

    # patch size for training
    PATCH_SHAPE = ast.literal_eval(os.getenv('PATCH_SHAPE'))
    PATCH_SHAPE_SINGLE = PATCH_SHAPE[0]

    # Sizes of the testing, and evaluation datasets
    TRAIN_SIZE = int(os.getenv('TRAIN_SIZE'))
    TEST_SIZE = int(os.getenv('TEST_SIZE'))
    VAL_SIZE = int(os.getenv('VAL_SIZE'))

    # define number of positive and negative samples
    N_POS = int(os.getenv('N_POS'))
    N_NEG = int(os.getenv('N_NEG'))
    initial_bias = None # np.log([self.N_POS/self.N_NEG])
    
    MODEL_TYPE = os.getenv('MODEL_TYPE')

    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    EPOCHS = int(os.getenv('EPOCHS'))
    RAMPUP_EPOCHS = int(os.getenv('RAMPUP_EPOCHS'))
    SUSTAIN_EPOCHS = int(os.getenv('SUSTAIN_EPOCHS'))

    BUFFER_SIZE = int(os.getenv('BUFFER_SIZE'))

    USE_ADJUSTED_LR = os.getenv('USE_ADJUSTED_LR') == 'True'
    LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
    MAX_LR = float(os.getenv('MAX_LR'))
    MID_LR = float(os.getenv('MID_LR'))
    MIN_LR = float(os.getenv('MIN_LR'))
    DROPOUT_RATE = float(os.getenv('DROPOUT_RATE'))

    LOSS = os.getenv('LOSS')
    if LOSS == 'custom_focal_tversky_loss':
        LOSS = Metrics.focal_tversky_loss
        LOSS_TXT = 'focal_tversky_loss'
    else:
        LOSS = LOSS
        LOSS_TXT = LOSS

    OPTIMIZER = os.getenv('OPTIMIZER')
    if  OPTIMIZER == 'custom':
        OPTIMIZER = keras.optimizers.Adam(learning_rate=3E-4)
    else:
        OPTIMIZER = OPTIMIZER

    OUT_CLASS_NUM = int(os.getenv('OUT_CLASS_NUM'))

    ACTIVATION_FN = 'sigmoid' if OUT_CLASS_NUM == 1 else 'softmax'
    CALLBACK_PARAMETER = os.getenv('CALLBACK_PARAMETER')

    # cloud stuff
    GCS_PROJECT = os.getenv('GCS_PROJECT')
    GCS_BUCKET = os.getenv('GCS_BUCKET')
    EE_SERVICE_CREDENTIALS = os.getenv('EE_SERVICE_CREDENTIALS')
    
    beam_options = PipelineOptions([], direct_num_workers=0, direct_running_mode="multi_processing", runner="DirectRunner")

    def __init__(self) -> None:
        self.physical_devices = Config.physical_devices
        self.DISTRIBUTED_STRATEGY = Config.DISTRIBUTED_STRATEGY

        self.BASEDIR = Config.BASEDIR
        self.DATADIR = Config.DATADIR

        self.TRAINING_DIR = Config.TRAINING_DIR
        print(f'TRAINING_DIR: {self.TRAINING_DIR}')
        self.TESTING_DIR = Config.TESTING_DIR
        self.VALIDATION_DIR = Config.VALIDATION_DIR

        self.OUTPUT_DIR = Config.OUTPUT_DIR

        self.MODEL_NAME = Config.MODEL_NAME
        self.MODEL_CHECKPOINT_NAME = Config.MODEL_CHECKPOINT_NAME
        
        self.METRICS = Config.METRICS

        self.FEATURES = Config.FEATURES
        print(f'FEATURES: {self.FEATURES}')
        print(f'FEATURES: {len(self.FEATURES)}')
        self.LABELS = Config.LABELS

        # patch size for training
        self.PATCH_SHAPE = Config.PATCH_SHAPE
        self.PATCH_SHAPE_SINGLE = Config.PATCH_SHAPE_SINGLE

        # Sizes of the testing, and evaluation datasets
        self.TRAIN_SIZE = Config.TRAIN_SIZE
        self.TEST_SIZE = Config.TEST_SIZE
        self.VAL_SIZE = Config.VAL_SIZE

        # define number of positive and negative samples
        self.N_POS = Config.N_POS
        self.N_NEG = Config.N_NEG
        self.initial_bias = Config.initial_bias
        print(f'initial bias: {self.initial_bias}')
        
        self.MODEL_TYPE = Config.MODEL_TYPE

        self.BATCH_SIZE = Config.BATCH_SIZE
        print(f'BATCH_SIZE: {self.BATCH_SIZE}')
        self.EPOCHS = Config.EPOCHS
        self.RAMPUP_EPOCHS = Config.RAMPUP_EPOCHS
        self.SUSTAIN_EPOCHS = Config.SUSTAIN_EPOCHS

        self.BUFFER_SIZE = Config.BUFFER_SIZE

        self.USE_ADJUSTED_LR = Config.USE_ADJUSTED_LR
        self.LEARNING_RATE = Config.LEARNING_RATE
        self.MAX_LR = Config.MAX_LR
        self.MID_LR = Config.MID_LR
        self.MIN_LR = Config.MIN_LR
        self.DROPOUT_RATE = Config.DROPOUT_RATE

        self.LOSS = Config.LOSS
        self.LOSS_TXT = Config.LOSS_TXT
        self.OPTIMIZER = Config.OPTIMIZER

        self.OUT_CLASS_NUM = Config.OUT_CLASS_NUM

        self.ACTIVATION_FN = Config.ACTIVATION_FN
        self.CALLBACK_PARAMETER = Config.CALLBACK_PARAMETER

        # cloud stuff
        self.GCS_PROJECT = Config.GCS_PROJECT
        self.GCS_BUCKET = Config.GCS_BUCKET
        self.EE_SERVICE_CREDENTIALS = Config.EE_SERVICE_CREDENTIALS
        self.beam_options = Config.beam_options
