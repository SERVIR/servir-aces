# -*- coding: utf-8 -*-

from pathlib import Path
import os
# import datetime
import ast

import numpy as np
import tesnorflow as tf
from tensorflow import keras
from model import metrics

from dotenv import load_dotenv
load_dotenv('.env')


class Config:
    """Configuration class for the project."""

    def __init__(self) -> None:
        self.physical_devices = tf.config.list_physical_devices('GPU')
        self.strategy = tf.distribute.MirroredStrategy(['GPU:0', 'GPU:1', 'GPU:2', 'GPU:3'])

        self.BASEDIR: Path = Path(os.getenv('BASEDIR'))
        self.DATADIR: Path = self.BASEDIR / os.getenv('DATADIR')

        self.TRAINING_DIR: Path = self.DATADIR / 'training'
        print(f'TRAINING_DIR: {self.TRAINING_DIR}')
        self.TESTING_DIR: Path = self.DATADIR / 'testing'
        self.VALIDATION_DIR: Path = self.DATADIR / 'validation'

        self.OUTPUT_DIR: Path = self.BASEDIR / 'output'

        self.MODEL_NAME: str = os.getenv('MODEL_NAME')
        self.MODEL_CHECKPOINT_NAME: str = os.getenv('MODEL_CHECKPOINT_NAME')
        
        self.METRICS = os.getenv('METRICS').split('\n')

        # today: str = datetime.date.today().strftime('%Y_%m_%d')
        # iterator: int = 1
        # while True:
        #     model_dir_name: str = f'trial_{today}_V{iterator}'
        #     self.MODEL_SAVE_DIR: Path = self.OUTPUT_DIR / model_dir_name
        #     try:
        #         os.mkdir(self.MODEL_SAVE_DIR)
        #     except FileExistsError:
        #         print(f'> {self.MODEL_SAVE_DIR} exists, creating another version...')
        #         iterator += 1
        #         continue
        #     break
        # print('***************************************************************************')

        self.FEATURES = os.getenv('FEATURES').split('\n')
        print(f'FEATURES: {self.FEATURES}')
        print(f'FEATURES: {len(self.FEATURES)}')
        self.LABELS = ast.literal_eval(os.getenv('LABELS'))

        # patch size for training
        self.PATCH_SHAPE = ast.literal_eval(os.getenv('PATCH_SHAPE'))
        self.PATCH_SHAPE_SINGLE = self.PATCH_SHAPE[0]

        # Sizes of the testing, and evaluation datasets
        self.TRAIN_SIZE = int(os.getenv('TRAIN_SIZE'))
        self.TEST_SIZE = int(os.getenv('TEST_SIZE'))
        self.VAL_SIZE = int(os.getenv('VAL_SIZE'))

        # define number of positive and negative samples
        self.N_POS = int(os.getenv('N_POS'))
        self.N_NEG = int(os.getenv('N_NEG'))
        self.initial_bias = None # np.log([self.N_POS/self.N_NEG])
        print(f'initial bias: {self.initial_bias}')
        
        self.MODEL_TYPE = os.getenv('MODEL_TYPE')

        self.BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
        print(f'BATCH_SIZE: {self.BATCH_SIZE}')
        self.EPOCHS = int(os.getenv('EPOCHS'))
        self.RAMPUP_EPOCHS = int(os.getenv('RAMPUP_EPOCHS'))
        self.SUSTAIN_EPOCHS = int(os.getenv('SUSTAIN_EPOCHS'))

        self.BUFFER_SIZE = int(os.getenv('BUFFER_SIZE'))

        self.USE_ADJUSTED_LR = os.getenv('USE_ADJUSTED_LR') == 'True'
        self.LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
        self.MAX_LR = float(os.getenv('MAX_LR'))
        self.MID_LR = float(os.getenv('MID_LR'))
        self.MIN_LR = float(os.getenv('MIN_LR'))
        self.DROPOUT_RATE = float(os.getenv('DROPOUT_RATE'))

        LOSS = os.getenv('LOSS')
        if LOSS == 'custom':
            self.LOSS = metrics.focal_tversky_loss
            self.LOSS_TXT = 'focal_tversky_loss'
        else:
            self.LOSS = LOSS
            self.LOSS_TXT = LOSS

        OPTIMIZER = os.getenv('OPTIMIZER')
        if  OPTIMIZER == 'custom':
            self.OPTIMIZER = keras.optimizers.Adam(learning_rate=3E-4)
        else:
            self.OPTIMIZER = OPTIMIZER

        self.OUT_CLASS_NUM = int(os.getenv('OUT_CLASS_NUM'))

        self.ACTIVATION_FN = 'sigmoid' if self.out_classes == 1 else 'softmax'
        self.CALLBACK_PARAMETER = os.getenv('CALLBACK_PARAMETER')
