# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

import os
import datetime
import json
import pickle
from functools import partial

import tensorflow as tf
from tensorflow import keras
from keras import callbacks

from aces.model_builder import ModelBuilder
from aces.data_processor import DataProcessor
from aces.metrics import Utils


class ModelTrainer:
    """
    A class for training deep learning models.

    Attributes:
        config: An object containing the configuration settings for model training.
        model_builder: An instance of ModelBuilder for building the model.
        build_model: A partial function for building the model with the specified model type.
    """
    def __init__(self, config):
        """
        Initialize the ModelTrainer object.

        Args:
            config: An object containing the configuration settings for model training.

        Attributes:
            config: The configuration settings for model training.
            model_builder: An instance of ModelBuilder for building the model.
            build_model: A partial function for building the model with the specified model type.
        """
        self.config = config
        self.model_builder = ModelBuilder(
            in_size=len(self.config.FEATURES),
            out_classes=self.config.OUT_CLASS_NUM,
            optimizer=self.config.OPTIMIZER,
            loss=self.config.LOSS
        )
        self.build_model = partial(self.model_builder.build_model, model_type=self.config.MODEL_TYPE)

    def train_model(self) -> None:
        """
        Train the model using the provided configuration settings.

        This method performs the following steps:
        1. Configures memory growth for TensorFlow.
        2. Creates TensorFlow datasets for training, testing, and validation.
        3. Builds and compiles the model.
        4. Prepares the output directory for saving models and results.
        5. Starts the training process.
        6. Evaluates and prints validation metrics.
        7. Saves training parameters, plots, and models.
        """
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
        logging.info("****************************************************************************")
        logging.info("****************************** training model... ***************************")
        self.start_training()
        logging.info("****************************************************************************")
        logging.info("****************************** evaluating model... *************************")
        self.evaluate_and_print_val()
        logging.info("****************************************************************************")
        logging.info("****************************** saving parameters... ************************")
        ModelTrainer.save_parameters(**self.config.__dict__)
        logging.info("****************************************************************************")
        logging.info("*************** saving model config and history object... ******************")
        self.save_history_object()
        ModelTrainer.save_model_config(self.config.MODEL_SAVE_DIR, **self.model.get_config())
        logging.info("****************************************************************************")
        logging.info("****************************** saving plots... *****************************")
        self.save_plots()
        logging.info("****************************************************************************")
        logging.info("****************************** saving models... ****************************")
        self.save_models()
        logging.info("****************************************************************************")

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

        Args:
            print_info: Flag indicating whether to print dataset information.

        Prints information about the created datasets if print_info is set to True.
        """
        self.TRAINING_DATASET = DataProcessor.get_dataset(
            # self.config.TRAINING_FILES,
            f"{str(self.config.TRAINING_DIR)}/*",
            self.config.FEATURES,
            self.config.LABELS,
            self.config.PATCH_SHAPE[0],
            self.config.BATCH_SIZE,
            self.config.OUT_CLASS_NUM,
        ).repeat()
        self.TESTING_DATASET = DataProcessor.get_dataset(
            # self.config.TESTING_FILES,
            f"{str(self.config.TESTING_DIR)}/*",
            self.config.FEATURES,
            self.config.LABELS,
            self.config.PATCH_SHAPE[0],
            1,
            self.config.OUT_CLASS_NUM,
        ).repeat()
        self.VALIDATION_DATASET = DataProcessor.get_dataset(
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
            DataProcessor.print_dataset_info(self.TRAINING_DATASET, "Training")
            DataProcessor.print_dataset_info(self.TESTING_DATASET, "Testing")
            DataProcessor.print_dataset_info(self.VALIDATION_DATASET, "Validation")

    def configure_memory_growth(self) -> None:
        """
        Configure TensorFlow to allocate GPU memory dynamically.

        If GPUs are found, this method enables memory growth for each GPU.
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
        """
        Build and compile the model.

        Args:
            print_model_summary: Flag indicating whether to print the model summary.

        Builds and compiles the model using the provided configuration settings.
        Prints the model summary if print_model_summary is set to True.
        """
        self.model = self.build_model(**self.config.__dict__)
        if print_model_summary:  logging.info(self.model.summary())

    def start_training(self) -> None:
        """
        Start the training process.

        Trains the model using the provided configuration settings and callbacks.
        """
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
        """
        Evaluate and print validation metrics.

        Evaluates the model on the validation dataset and prints the metrics.
        """
        logging.info("************************************************")
        logging.info("************************************************")
        logging.info("Validation")
        evaluate_results = self.model.evaluate(self.VALIDATION_DATASET)
        for name, value in zip(self.model.metrics_names, evaluate_results):
            logging.info(f"{name}: {value}")
        logging.info("\n")

    @staticmethod
    def save_parameters(**config) -> None:
        """
        Save the training parameters to a text file.

        Saves the training parameters used in the configuration settings to a text file.
        """
        with open(f"{str(config.get('MODEL_SAVE_DIR'))}/parameters.txt", "w") as f:
            f.write(f"TRAIN_SIZE: {config.get('TRAIN_SIZE')}\n")
            f.write(f"TEST_SIZE: {config.get('TEST_SIZE')}\n")
            f.write(f"VAL_SIZE: {config.get('VAL_SIZE')}\n")
            f.write(f"BATCH_SIZE: {config.get('BATCH_SIZE')}\n")
            f.write(f"EPOCHS: {config.get('EPOCHS')}\n")
            f.write(f"LOSS: {config.get('LOSS_TXT')}\n")
            f.write(f"BUFFER_SIZE: {config.get('BUFFER_SIZE')}\n")
            f.write(f"LEARNING_RATE: {config.get('LEARNING_RATE')}\n")
            if config.get('USE_ADJUSTED_LR'):
                f.write(f"USE_ADJUSTED_LR: {config.get('USE_ADJUSTED_LR')}\n")
                f.write(f"MAX_LR: {config.get('MAX_LR')}\n")
                f.write(f"MID_LR: {config.get('MID_LR')}\n")
                f.write(f"MIN_LR: {config.get('MIN_LR')}\n")
                f.write(f"RAMPUP_EPOCHS: {config.get('RAMPUP_EPOCHS')}\n")
                f.write(f"SUSTAIN_EPOCHS: {config.get('SUSTAIN_EPOCHS')}\n")
            f.write(f"DROPOUT_RATE: {config.get('DROPOUT_RATE')}\n")
            f.write(f"ACTIVATION_FN: {config.get('ACTIVATION_FN')}\n")
            f.write(f"FEATURES: {config.get('FEATURES')}\n")
            f.write(f"LABELS: {config.get('LABELS')}\n")
            f.write(f"PATCH_SHAPE: {config.get('PATCH_SHAPE')}\n")
            f.write(f"CALLBACK_PARAMETER: {config.get('CALLBACK_PARAMETER')}\n")
            f.write(f"MODEL_NAME: {config.get('MODEL_NAME')}.h5\n")
            f.write(f"MODEL_CHECKPOINT_NAME: {config.get('MODEL_CHECKPOINT_NAME')}.h5\n")        
        f.close()

    @staticmethod
    def save_model_config(save_dir, **model_config) -> None:
        with open(f"{save_dir}/config.json", "w") as f:
            json.dump(model_config, f, indent=4)
        f.close()

    def save_plots(self) -> None:
        """
        Save plots and model visualization.

        Saves the model architecture plot, training history plot, and model object.
        """
        logging.info(f"Saving plots and model visualization at {self.config.MODEL_SAVE_DIR}...")
        keras.utils.plot_model(self.model, f"{self.config.MODEL_SAVE_DIR}/model.png", show_shapes=True)
        Utils.plot_metrics([key.replace("val_", "") for key in self.history.history.keys() if key.startswith("val_")],
                           self.history.history, len(self.history.epoch), self.config.MODEL_SAVE_DIR)

    def save_history_object(self) -> None:
        """
        Save the history object.
        """
        with open(f"{self.config.MODEL_SAVE_DIR}/model.pkl", "wb") as f:
            pickle.dump(self.history.history, f)

    def save_models(self) -> None:
        """
        Save the trained models.

        Saves the trained models in different formats: h5 and tf formats.
        """
        self.model.save(f"{str(self.config.MODEL_SAVE_DIR)}/{self.config.MODEL_NAME}.h5", save_format="h5")
        self.model.save(f"{str(self.config.MODEL_SAVE_DIR)}/{self.config.MODEL_NAME}.tf", save_format="tf")
        self.model.save_weights(f"{str(self.config.MODEL_SAVE_DIR)}/modelWeights.h5", save_format="h5")
        self.model.save_weights(f"{str(self.config.MODEL_SAVE_DIR)}/modelWeights.tf", save_format="tf")

