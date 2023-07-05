# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import os
import datetime
import glob
import pickle
from functools import partial

import tensorflow as tf
from tensorflow import keras
from keras import callbacks

from aces.model import ModelBuilder
from aces.dataio import DataIO
from aces.metrics import Utils


class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model_builder = ModelBuilder(
            in_size=len(self.config.FEATURES),
            out_classes=self.config.OUT_CLASS_NUM,
            optimizer=self.config.OPTIMIZER,
            loss=self.config.LOSS
        )
        self.build_model = partial(self.model_builder.build_model, model_type=self.config.MODEL_TYPE)

    def train_model(self) -> None:
        logging.info("****************************************************************************")
        print(f"****************************** Configure memory growth... ************************")
        self.configure_memory_growth()
        logging.info("****************************************************************************")
        logging.info("****************************** creating datasets... ************************")
        self.create_datasets(print_info=False)
        logging.info("****************************************************************************")
        logging.info("************************ building and compiling model... *******************")
        self.build_and_compile_model(print_model_summary=True)
        logging.info("****************************************************************************")
        logging.info("************************ preparing output directory... *********************")
        self.prepare_output_dir()
        # self.get_file_paths()
        logging.info("****************************************************************************")
        logging.info("****************************** training model... ***************************")
        self.start_training()
        self.evaluate_and_print_val()
        self.save_parameters()
        self.save_plots()
        self.save_models()

    def prepare_output_dir(self) -> None:
        """
        Prepare the output directory for saving models and results.
        Creates a directory with a timestamped name and increments the version number if necessary.
        """
        today = datetime.date.today().strftime("%Y_%m_%d")
        iterator = 1
        while True:
            self.model_dir_name = f"trial_{self.config.MODEL_TYPE}_{today}_V{iterator}"
            self.config.MODEL_SAVE_DIR = self.config.OUTPUT_DIR / self.model_dir_name
            try:
                os.mkdir(self.config.MODEL_SAVE_DIR)
            except FileExistsError:
                logging.info(f"> {self.config.MODEL_SAVE_DIR} exists, creating another version...")
                iterator += 1
                continue
            break

    def create_datasets(self, print_info: bool = False) -> None:
        """
        Create TensorFlow datasets for training, testing, and validation.
        """
        self.TRAINING_DATASET = DataIO.get_dataset(
            # self.config.TRAINING_FILES,
            f"{str(self.config.TRAINING_DIR)}/*",
            self.config.FEATURES,
            self.config.LABELS,
            self.config.PATCH_SHAPE[0],
            self.config.BATCH_SIZE,
            self.config.OUT_CLASS_NUM,
        ).repeat()
        self.TESTING_DATASET = DataIO.get_dataset(
            # self.config.TESTING_FILES,
            f"{str(self.config.TESTING_DIR)}/*",
            self.config.FEATURES,
            self.config.LABELS,
            self.config.PATCH_SHAPE[0],
            1,
            self.config.OUT_CLASS_NUM,
        ).repeat()
        self.VALIDATION_DATASET = DataIO.get_dataset(
            # self.config.VALIDATION_FILES,
            f"{str(self.config.VALIDATION_DIR)}/*",
            self.config.FEATURES,
            self.config.LABELS,
            self.config.PATCH_SHAPE[0],
            1,
            self.config.OUT_CLASS_NUM,
        )

        if print_info:
            logging.info("Printing dataset info:")
            DataIO.print_dataset_info(self.TRAINING_DATASET, "Training")
            DataIO.print_dataset_info(self.TESTING_DATASET, "Testing")
            DataIO.print_dataset_info(self.VALIDATION_DATASET, "Validation")

    def configure_memory_growth(self) -> None:
        """
        Configure TensorFlow to allocate GPU memory dynamically.
        """
        if self.config.physical_devices:
            logging.info(f" > Found {len(self.config.physical_devices)} GPUs")
            try:
                for device in self.config.physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
            except Exception as err:
                logging.error(err)
        else:
            logging.info(" > No GPUs found")

    def build_and_compile_model(self, print_model_summary: bool = True) -> None:
        self.model = self.build_model(**self.config.__dict__)
        if print_model_summary:  logging.info(self.model.summary())

    def start_training(self) -> None:
        model_checkpoint = callbacks.ModelCheckpoint(
            f"{str(self.config.MODEL_SAVE_DIR)}/{self.config.MODEL_CHECKPOINT_NAME}.h5",
            monitor=self.config.CALLBACK_PARAMETER,
            save_best_only=True,
            mode="max",
            verbose=1,
            save_weights_only=True,
        )  # save best model

        early_stopping = callbacks.EarlyStopping(
            monitor=self.config.CALLBACK_PARAMETER,
            patience=int(0.4 * self.config.EPOCHS),
            verbose=1,
            mode="max",
            restore_best_weights=True,
        )
        tensorboard = callbacks.TensorBoard(log_dir=str(self.config.MODEL_SAVE_DIR / "logs"), write_images=True)

        def lr_scheduler(epoch):
            if epoch < self.config.RAMPUP_EPOCHS:
                return self.config.MAX_LR
            elif epoch < self.config.RAMPUP_EPOCHS + self.config.SUSTAIN_EPOCHS:
                return self.config.MID_LR
            else:
                return self.config.MIN_LR

        lr_callback = callbacks.LearningRateScheduler(lambda epoch: lr_scheduler(epoch), verbose=True)

        model_callbacks = [model_checkpoint, tensorboard]
        if self.config.USE_ADJUSTED_LR:
            model_callbacks.append(lr_callback)

        self.model_callbacks = model_callbacks

        self.history = self.model.fit(
            x=self.TRAINING_DATASET,
            epochs=self.config.EPOCHS,
            steps_per_epoch=(self.config.TRAIN_SIZE // self.config.BATCH_SIZE),
            validation_data=self.TESTING_DATASET,
            validation_steps=(self.config.TEST_SIZE // self.config.BATCH_SIZE),
            callbacks=model_callbacks,
        )

        # logging.info(self.model.summary())

    def evaluate_and_print_val(self) -> None:
        logging.info("************************************************")
        logging.info("************************************************")
        logging.info("Validation")
        evaluate_results = self.model.evaluate(self.config.VALIDATION_DATASET)
        for name, value in zip(self.model.metrics_names, evaluate_results):
            logging.info(f"{name}: {value}")
        logging.info()

    def save_parameters(self) -> None:
        with open(f"{str(self.config.MODEL_SAVE_DIR)}/parameters.txt", "w") as f:
            f.write(f"MODEL_FUNCTION: {self.config.MODEL_FUNCTION}\n")
            f.write(f"TRAIN_SIZE: {self.config.TRAIN_SIZE}\n")
            f.write(f"TEST_SIZE: {self.config.TEST_SIZE}\n")
            f.write(f"VAL_SIZE: {self.config.VAL_SIZE}\n")
            f.write(f"BATCH_SIZE: {self.config.BATCH_SIZE}\n")
            f.write(f"EPOCHS: {self.config.EPOCHS}\n")
            f.write(f"LOSS: {self.config.LOSS_TXT}\n")
            f.write(f"BUFFER_SIZE: {self.config.BUFFER_SIZE}\n")
            f.write(f"LEARNING_RATE: {self.config.LEARNING_RATE}\n")
            if self.config.USE_ADJUSTED_LR:
                f.write(f"USE_ADJUSTED_LR: {self.config.USE_ADJUSTED_LR}\n")
                f.write(f"MAX_LR: {self.config.MAX_LR}\n")
                f.write(f"MID_LR: {self.config.MID_LR}\n")
                f.write(f"MIN_LR: {self.config.MIN_LR}\n")
                f.write(f"RAMPUP_EPOCHS: {self.config.RAMPUP_EPOCHS}\n")
                f.write(f"SUSTAIN_EPOCHS: {self.config.SUSTAIN_EPOCHS}\n")
            f.write(f"DROPOUT_RATE: {self.config.DROPOUT_RATE}\n")
            f.write(f"ACTIVATION_FN: {self.config.ACTIVATION_FN}\n")
            f.write(f"FEATURES: {self.config.FEATURES}\n")
            f.write(f"LABELS: {self.config.LABELS}\n")
            f.write(f"PATCH_SHAPE: {self.config.PATCH_SHAPE}\n")
            f.write(f"CALLBACK_PARAMETER: {self.config.CALLBACK_PARAMETER}\n")
            f.write(f"MODEL_NAME: {self.config.MODEL_NAME}.h5\n")
            f.write(f"MODEL_CHECKPOINT_NAME: {self.config.MODEL_CHECKPOINT_NAME}.h5\n")
        
        f.close()

    def save_plots(self) -> None:
        keras.utils.plot_model(self.model, f"{self.config.MODEL_SAVE_DIR}/model.png", show_shapes=True)
        with open(f"{self.config.MODEL_SAVE_DIR}/model.pkl", "wb") as f:
            pickle.dump(self.history.history, f)
        Utils.plot_metrics(self.config.METRICS, self.history.history, self.history.epoch, self.config.MODEL_SAVE_DIR)

    def save_models(self) -> None:
        self.model.save(f"{str(self.config.MODEL_SAVE_DIR)}/{self.config.MODEL_NAME}.h5", save_format="h5")
        self.model.save(f"{str(self.config.MODEL_SAVE_DIR)}/{self.config.MODEL_NAME}.tf", save_format="tf")
        self.model.save_weights(f"{str(self.config.MODEL_SAVE_DIR)}/modelWeights.h5", save_format="h5")
        self.model.save_weights(f"{str(self.config.MODEL_SAVE_DIR)}/modelWeights.tf", save_format="tf")

