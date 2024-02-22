# -*- coding: utf-8 -*-

"""
ACES Configuration Module
This module provides the configuration settings for the ACES project.
"""

from pathlib import Path
import os
import ast

from dotenv import load_dotenv
load_dotenv(".env")


class Config:
    """
    ACES Configuration Class

    This class contains the configuration settings for the ACES project. These are generated from the .env file.

    Attributes:
        BASEDIR (str): The base directory for data I/O information.
        DATADIR (str): The directory for data collection experiments. DATADIR = BASEDIR / DATADIR
        MODEL_NAME (str): The name of the model.
        MODEL_CHECKPOINT_NAME (str): The name for model checkpoints.
        MODEL_DIR_NAME (str): The name of the model directory. MODEL_DIR = OUTPUT_DIR / MODEL_DIR_NAME
        AUTO_MODEL_DIR_NAME (bool): Flag to use automatic model directory naming.
        # True generates as trial_MODELTYPE + datetime.now() + _v + version
        # False uses the MODEL_DIR_NAME
        FEATURES (str): The list of features used in the model.
        USE_ELEVATION (bool): Flag to use elevation data.
        USE_S1 (bool): Flag to use Sentinel-1 data.
        DERIVE_FEATURES (bool): Flag to derive features.
        ADDED_FEATURES (list): A list of additional features used in the model (used when DERIVE_FEATURES is true).
        LABELS (list): A list of labels used in the model.
        SCALE (int): The scale of the data.
        # Note: The seeding componenet needs fixing. It is not working as expected.
        USE_SEED (bool): Flag to use a seed for reproducbility.
        SEED (int): The seed used for reproducibility.
        PATCH_SHAPE (tuple): The shape of training patches.
        # buffer for prediction purpose
        # Half this will extend on the sides of each patch.
        # if zero; does not do buffer
        # else specify the size as tuple (e.g. 128, 128)
        KERNEL_BUFFER (tuple): The buffer for prediction purpose.
        TRAIN_SIZE (int): The size of the training dataset.
        TEST_SIZE (int): The size of the testing dataset.
        VAL_SIZE (int): The size of the validation dataset.
        BATCH_SIZE (int): The batch size for model training.
        EPOCHS (int): The number of epochs for model training.
        RAMPUP_EPOCHS (int): The number of ramp-up epochs.
        SUSTAIN_EPOCHS (int): The number of sustain epochs.
        USE_ADJUSTED_LR (bool): Flag to use adjusted learning rate.
        MAX_LR (float): The maximum learning rate.
        MID_LR (float): The intermediate learning rate.
        MIN_LR (float): The minimum learning rate.
        DROPOUT_RATE (float): The dropout rate for the model.
        CALLBACK_PARAMETER (str): The parameter used for callbacks.
        # patience for EARLY_STOPPING = int(0.3 * EPOCHS)
        # monitors CALLBACK_PARAMETER
        EARLY_STOPPING (bool): Flag to use early stopping.
        MODEL_TYPE (str): The type of model (current choices: cnn, dnn, unet).
        TRANSFORM_DATA (bool): Flag to transform data (rotation, flip) DNN
        does not do augmentation.
        ACTIVATION_FN (str): The activation function for the model.
        OPTIMIZER (str): The optimizer used for model training.
        # You can use any loss function available [here](https://www.tensorflow.org/api_docs/python/tf/keras/losses):
        # other available are: custom_focal_tversky_loss
        LOSS (str): The loss function used for model training.,
        OUT_CLASS_NUM (int): The number of output classes.
        # This parameter is used in script [5. prediction_unet.py](https://github.com/SERVIR/servir-aces/blob/main/workflow/v2/5.prediction_unet.py)
        # and [host_vertex_ai.py](https://github.com/SERVIR/servir-aces/blob/main/workflow/v2/host_vertex_ai.py) files
        USE_BEST_MODEL_FOR_INFERENCE (bool): Flag to use the best model for inference.
        USE_SERVICE_ACCOUNT (bool): Flag to use a service account for Earth Engine.
        # This is used when USE_SERVICE_ACCOUNT is True
        # This is used to
        EE_SERVICE_CREDENTIALS (str): The path to the Earth Engine service account credentials.
        # Used in the script [5.prediction_unet.py](https://github.com/SERVIR/servir-aces/blob/main/workflow/v2/5.prediction_unet.py) and
        # [5.prediction_dnn](https://github.com/SERVIR/servir-aces/blob/main/workflow/v2/5.prediction_dnn.py)
        EE_OUPUT_ASSET (str): The Earth Engine output (prediction) asset path.
        OUTPUT_NAME (str): The name of the output prediction for GEE asset, locally (in TF Format) and gcs output (in TFRecord format).
        GCS_BUCKET (str): The Google Cloud Storage bucket name.
        # Exported to this dir from [4.export_image_for_prediction.py](https://github.com/SERVIR/servir-aces/blob/main/workflow/v2/4.export_image_for_prediction.py)
        GCS_IMAGE_DIR (str): The Google Cloud Storage image directory to store the image for prediction.
        # used as file_name_prefix = {GCS_IMAGE_DIR}/{GCS_IMAGE_PREFIX} for exporting image
        GCS_IMAGE_PREFIX (str): The Google Cloud Storage image prefix.
        # Used in the script [host_vertex_ai.py](https://github.com/SERVIR/servir-aces/blob/main/workflow/v2/host_vertex_ai.py)
        GCS_PROJECT (str): The Google Cloud Storage project name.
        GCS_VERTEX_MODEL_SAVE_DIR (str): The Google Cloud Storage Vertex AI model save directory.
        GCS_REGION (str): The Google Cloud Storage region.
        GCS_VERTEX_CONTAINER_IMAGE (str): The Google Cloud Storage Vertex AI container image.
        # AI platform expects a need a serialized model, this parameter does this.
        # See its uses in `data_processor.py` and `model_training.py`
        USE_AI_PLATFORM (bool): Flag to use Google Cloud AI Platform.
        # Get machine type here: https://cloud.google.com/vertex-ai/docs/predictions/configure-compute
        GCP_MACHINE_TYPE (str): The Google Cloud Platform machine type.
    """
    BASEDIR = Path(os.getenv("BASEDIR"))
    DATADIR = BASEDIR / os.getenv("DATADIR")

    TRAINING_DIR = DATADIR / "training"
    TESTING_DIR = DATADIR / "testing"
    VALIDATION_DIR= DATADIR / "validation"

    OUTPUT_DIR = BASEDIR / os.getenv("OUTPUT_DIR")

    MODEL_NAME = os.getenv("MODEL_NAME")
    MODEL_CHECKPOINT_NAME = os.getenv("MODEL_CHECKPOINT_NAME")

    MODEL_DIR_NAME = os.getenv("MODEL_DIR_NAME")

    MODEL_DIR = OUTPUT_DIR / MODEL_DIR_NAME

    AUTO_MODEL_DIR_NAME = os.getenv("AUTO_MODEL_DIR_NAME") == "True"
    SCALE = int(os.getenv("SCALE"))

    FEATURES = os.getenv("FEATURES").split("\n")
    ADDED_FEATURES = os.getenv("ADDED_FEATURES").split("\n")
    USE_ELEVATION = os.getenv("USE_ELEVATION") == "True"
    USE_S1 = os.getenv("USE_S1") == "True"
    LABELS = ast.literal_eval(os.getenv("LABELS"))

    USE_SEED = os.getenv("USE_SEED") == "True"
    SEED = int(os.getenv("SEED"))

    # patch size for training
    PATCH_SHAPE = ast.literal_eval(os.getenv("PATCH_SHAPE"))
    PATCH_SHAPE_SINGLE = PATCH_SHAPE[0]
    KERNEL_BUFFER = os.getenv("KERNEL_BUFFER")
    if KERNEL_BUFFER == "0":
        KERNEL_BUFFER = None
    else:
        KERNEL_BUFFER = ast.literal_eval(KERNEL_BUFFER)

    # Sizes of the testing, and evaluation datasets
    TRAIN_SIZE = int(os.getenv("TRAIN_SIZE"))
    TEST_SIZE = int(os.getenv("TEST_SIZE"))
    VAL_SIZE = int(os.getenv("VAL_SIZE"))

    MODEL_TYPE = os.getenv("MODEL_TYPE")
    IS_DNN = True if MODEL_TYPE == "dnn" else False

    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    EPOCHS = int(os.getenv("EPOCHS"))
    RAMPUP_EPOCHS = int(os.getenv("RAMPUP_EPOCHS"))
    SUSTAIN_EPOCHS = int(os.getenv("SUSTAIN_EPOCHS"))

    USE_ADJUSTED_LR = os.getenv("USE_ADJUSTED_LR") == "True"
    MAX_LR = float(os.getenv("MAX_LR"))
    MID_LR = float(os.getenv("MID_LR"))
    MIN_LR = float(os.getenv("MIN_LR"))
    DROPOUT_RATE = float(os.getenv("DROPOUT_RATE"))

    LOSS = os.getenv("LOSS")

    OPTIMIZER = os.getenv("OPTIMIZER")

    OUT_CLASS_NUM = int(os.getenv("OUT_CLASS_NUM"))

    USE_BEST_MODEL_FOR_INFERENCE = os.getenv("USE_BEST_MODEL_FOR_INFERENCE") == "True"

    ACTIVATION_FN = "sigmoid" if OUT_CLASS_NUM == 1 else "softmax"
    CALLBACK_PARAMETER = os.getenv("CALLBACK_PARAMETER")

    EARLY_STOPPING = os.getenv("EARLY_STOPPING") == "True"
    TRANSFORM_DATA = os.getenv("TRANSFORM_DATA") == "True"
    DERIVE_FEATURES = os.getenv("DERIVE_FEATURES") == "True"

    # EE settings
    USE_SERVICE_ACCOUNT = os.getenv("USE_SERVICE_ACCOUNT") == "True"
    EE_SERVICE_CREDENTIALS = os.getenv("EE_SERVICE_CREDENTIALS")
    EE_USER = os.getenv("EE_USER")
    EE_OUTPUT_ASSET = os.getenv("EE_OUTPUT_ASSET")
    OUTPUT_NAME = os.getenv("OUTPUT_NAME")

    # cloud stuff
    GCS_PROJECT = os.getenv("GCS_PROJECT")
    GCS_BUCKET = os.getenv("GCS_BUCKET")
    GCS_IMAGE_DIR = os.getenv("GCS_IMAGE_DIR")
    GCS_IMAGE_PREFIX = os.getenv("GCS_IMAGE_PREFIX")
    GCS_VERTEX_MODEL_SAVE_DIR = os.getenv("GCS_VERTEX_MODEL_SAVE_DIR")
    GCS_REGION = os.getenv("GCS_REGION")
    GCS_VERTEX_CONTAINER_IMAGE = os.getenv("GCS_VERTEX_CONTAINER_IMAGE")

    USE_AI_PLATFORM = os.getenv("USE_AI_PLATFORM") == "True"

    GCP_MACHINE_TYPE = os.getenv("GCP_MACHINE_TYPE")

    def __init__(self) -> None:
        self.BASEDIR = Config.BASEDIR
        self.DATADIR = Config.DATADIR

        self.TRAINING_DIR = Config.TRAINING_DIR
        print(f"TRAINING_DIR: {self.TRAINING_DIR}")
        self.TESTING_DIR = Config.TESTING_DIR
        self.VALIDATION_DIR = Config.VALIDATION_DIR

        self.OUTPUT_DIR = Config.OUTPUT_DIR

        self.MODEL_NAME = Config.MODEL_NAME
        self.MODEL_CHECKPOINT_NAME = Config.MODEL_CHECKPOINT_NAME
        self.MODEL_DIR_NAME = Config.MODEL_DIR_NAME
        self.MODEL_DIR = Config.MODEL_DIR
        self.AUTO_MODEL_DIR_NAME = Config.AUTO_MODEL_DIR_NAME

        self.SCALE = Config.SCALE

        self.FEATURES = Config.FEATURES
        self.ADDED_FEATURES = Config.ADDED_FEATURES
        self.USE_ELEVATION = Config.USE_ELEVATION
        self.USE_S1 = Config.USE_S1
        self.LABELS = Config.LABELS

        self.USE_SEED = Config.USE_SEED
        self.SEED = Config.SEED

        # patch size for training
        self.PATCH_SHAPE = Config.PATCH_SHAPE
        self.PATCH_SHAPE_SINGLE = Config.PATCH_SHAPE_SINGLE
        self.KERNEL_BUFFER = Config.KERNEL_BUFFER

        # Sizes of the testing, and evaluation datasets
        self.TRAIN_SIZE = Config.TRAIN_SIZE
        self.TEST_SIZE = Config.TEST_SIZE
        self.VAL_SIZE = Config.VAL_SIZE

        self.MODEL_TYPE = Config.MODEL_TYPE
        self.IS_DNN = Config.IS_DNN

        self.BATCH_SIZE = Config.BATCH_SIZE
        print(f"BATCH_SIZE: {self.BATCH_SIZE}")
        self.EPOCHS = Config.EPOCHS
        self.RAMPUP_EPOCHS = Config.RAMPUP_EPOCHS
        self.SUSTAIN_EPOCHS = Config.SUSTAIN_EPOCHS

        self.USE_ADJUSTED_LR = Config.USE_ADJUSTED_LR
        self.MAX_LR = Config.MAX_LR
        self.MID_LR = Config.MID_LR
        self.MIN_LR = Config.MIN_LR
        self.DROPOUT_RATE = Config.DROPOUT_RATE

        self.LOSS = Config.LOSS
        self.OPTIMIZER = Config.OPTIMIZER

        self.OUT_CLASS_NUM = Config.OUT_CLASS_NUM

        self.USE_BEST_MODEL_FOR_INFERENCE = Config.USE_BEST_MODEL_FOR_INFERENCE

        self.ACTIVATION_FN = Config.ACTIVATION_FN
        self.CALLBACK_PARAMETER = Config.CALLBACK_PARAMETER

        self.EARLY_STOPPING = Config.EARLY_STOPPING
        self.TRANSFORM_DATA = Config.TRANSFORM_DATA
        self.DERIVE_FEATURES = Config.DERIVE_FEATURES

        # EE settings
        self.USE_SERVICE_ACCOUNT = Config.USE_SERVICE_ACCOUNT
        self.EE_SERVICE_CREDENTIALS = Config.EE_SERVICE_CREDENTIALS
        self.EE_USER = Config.EE_USER
        self.EE_OUTPUT_ASSET = Config.EE_OUTPUT_ASSET
        self.OUTPUT_NAME = Config.OUTPUT_NAME

        # cloud stuff
        self.GCS_PROJECT = Config.GCS_PROJECT
        self.GCS_BUCKET = Config.GCS_BUCKET
        self.GCS_IMAGE_DIR = Config.GCS_IMAGE_DIR
        self.GCS_IMAGE_PREFIX = Config.GCS_IMAGE_PREFIX

        self.GCS_VERTEX_MODEL_SAVE_DIR = Config.GCS_VERTEX_MODEL_SAVE_DIR
        self.GCS_REGION = Config.GCS_REGION
        self.GCS_VERTEX_CONTAINER_IMAGE = Config.GCS_VERTEX_CONTAINER_IMAGE

        self.USE_AI_PLATFORM = Config.USE_AI_PLATFORM

        self.GCP_MACHINE_TYPE = Config.GCP_MACHINE_TYPE
