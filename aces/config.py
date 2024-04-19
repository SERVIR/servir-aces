# -*- coding: utf-8 -*-

"""
ACES Configuration Module
This module provides the configuration settings for the ACES project.
"""

from pathlib import Path
import os
import ast
from dotenv import load_dotenv


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
        DATA_OUTPUT_DIR (str): The output directory for data export. Used in the workflow/v2/generate_training_patches script.
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

    def __init__(self, config_file, override=False) -> None:
        """
        ACES Configuration Class Constructor

        Args:
            config_file (str): The path to the configuration file.
            override (bool): Flag to override the configuration settings.
        """
        load_dotenv(config_file, override=override)

        self.BASEDIR = Path(os.getenv("BASEDIR"))
        _DATADIR = os.getenv("DATADIR")
        if _DATADIR.startswith("gs://"):
            self.DATADIR = _DATADIR
            self.TRAINING_DIR = f"{self.DATADIR}/training"
            self.TESTING_DIR = f"{self.DATADIR}/testing"
            self.VALIDATION_DIR = f"{self.DATADIR}/validation"
        else:
            self.DATADIR = self.BASEDIR / os.getenv("DATADIR")
            self.TRAINING_DIR = self.DATADIR / "training"
            self.TESTING_DIR = self.DATADIR / "testing"
            self.VALIDATION_DIR= self.DATADIR / "validation"

        print(f"BASEDIR: {self.BASEDIR}")
        print(f"DATADIR: {self.DATADIR}")

        self.OUTPUT_DIR = self.BASEDIR / os.getenv("OUTPUT_DIR")

        self.MODEL_NAME = os.getenv("MODEL_NAME")
        self.MODEL_CHECKPOINT_NAME = os.getenv("MODEL_CHECKPOINT_NAME")

        self.MODEL_DIR_NAME = os.getenv("MODEL_DIR_NAME")

        self.MODEL_DIR = self.OUTPUT_DIR / self.MODEL_DIR_NAME

        self.AUTO_MODEL_DIR_NAME = os.getenv("AUTO_MODEL_DIR_NAME") == "True"

        self.DATA_OUTPUT_DIR = os.getenv("DATA_OUTPUT_DIR")

        self.SCALE = int(os.getenv("SCALE"))

        self.FEATURES = os.getenv("FEATURES").split("\n")
        self.ADDED_FEATURES = os.getenv("ADDED_FEATURES").split("\n")
        self.USE_ELEVATION = os.getenv("USE_ELEVATION") == "True"
        self.USE_S1 = os.getenv("USE_S1") == "True"
        self.LABELS = ast.literal_eval(os.getenv("LABELS"))

        if self.USE_ELEVATION:
                self.FEATURES.extend(["elevation", "slope"])

        if self.USE_S1:
            self.FEATURES.extend(["vv_asc_before", "vh_asc_before", "vv_asc_during", "vh_asc_during",
                                  "vv_desc_before", "vh_desc_before", "vv_desc_during", "vh_desc_during"])

        print(f"using features: {self.FEATURES}")
        print(f"using labels: {self.LABELS}")

        self.USE_SEED = os.getenv("USE_SEED") == "True"
        self.SEED = int(os.getenv("SEED"))

        self.PRINT_INFO = os.getenv("USE_SEED") == "True"

        # patch size for training
        self.PATCH_SHAPE = ast.literal_eval(os.getenv("PATCH_SHAPE"))
        self.PATCH_SHAPE_SINGLE = self.PATCH_SHAPE[0]
        KERNEL_BUFFER = os.getenv("KERNEL_BUFFER")
        if KERNEL_BUFFER == "0":
            self.KERNEL_BUFFER = None
        else:
            self.KERNEL_BUFFER = ast.literal_eval(KERNEL_BUFFER)

        # Sizes of the testing, and evaluation datasets
        self.TRAIN_SIZE = int(os.getenv("TRAIN_SIZE"))
        self.TEST_SIZE = int(os.getenv("TEST_SIZE"))
        self.VAL_SIZE = int(os.getenv("VAL_SIZE"))

        self.MODEL_TYPE = os.getenv("MODEL_TYPE")
        self.IS_DNN = True if self.MODEL_TYPE == "dnn" else False

        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
        self.EPOCHS = int(os.getenv("EPOCHS"))
        self.RAMPUP_EPOCHS = int(os.getenv("RAMPUP_EPOCHS"))
        self.SUSTAIN_EPOCHS = int(os.getenv("SUSTAIN_EPOCHS"))

        self.USE_ADJUSTED_LR = os.getenv("USE_ADJUSTED_LR") == "True"
        self.MAX_LR = float(os.getenv("MAX_LR"))
        self.MID_LR = float(os.getenv("MID_LR"))
        self.MIN_LR = float(os.getenv("MIN_LR"))
        self.DROPOUT_RATE = float(os.getenv("DROPOUT_RATE"))

        self.LOSS = os.getenv("LOSS")

        self.OPTIMIZER = os.getenv("OPTIMIZER")

        self.OUT_CLASS_NUM = int(os.getenv("OUT_CLASS_NUM"))

        self.USE_BEST_MODEL_FOR_INFERENCE = os.getenv("USE_BEST_MODEL_FOR_INFERENCE") == "True"

        self.ACTIVATION_FN = "sigmoid" if self.OUT_CLASS_NUM == 1 else "softmax"
        self.CALLBACK_PARAMETER = os.getenv("CALLBACK_PARAMETER")

        self.EARLY_STOPPING = os.getenv("EARLY_STOPPING") == "True"
        self.TRANSFORM_DATA = os.getenv("TRANSFORM_DATA") == "True"
        self.DERIVE_FEATURES = os.getenv("DERIVE_FEATURES") == "True"

        # EE settings
        self.USE_SERVICE_ACCOUNT = os.getenv("USE_SERVICE_ACCOUNT") == "True"
        self.EE_SERVICE_CREDENTIALS = os.getenv("EE_SERVICE_CREDENTIALS")
        self.EE_USER = os.getenv("EE_USER")
        self.EE_OUTPUT_ASSET = os.getenv("EE_OUTPUT_ASSET")
        self.OUTPUT_NAME = os.getenv("OUTPUT_NAME")

        # cloud stuff
        self.GCS_PROJECT = os.getenv("GCS_PROJECT")
        self.GCS_BUCKET = os.getenv("GCS_BUCKET")
        self.GCS_IMAGE_DIR = os.getenv("GCS_IMAGE_DIR")
        self.GCS_IMAGE_PREFIX = os.getenv("GCS_IMAGE_PREFIX")
        self.GCS_VERTEX_MODEL_SAVE_DIR = os.getenv("GCS_VERTEX_MODEL_SAVE_DIR")
        self.GCS_REGION = os.getenv("GCS_REGION")
        self.GCS_VERTEX_CONTAINER_IMAGE = os.getenv("GCS_VERTEX_CONTAINER_IMAGE")

        self.USE_AI_PLATFORM = os.getenv("USE_AI_PLATFORM") == "True"

        self.GCP_MACHINE_TYPE = os.getenv("GCP_MACHINE_TYPE")
