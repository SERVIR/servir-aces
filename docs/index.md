# ACES (Agricultural Classification and Estimation Service)

[![image](https://img.shields.io/pypi/v/servir-aces.svg)](https://pypi.python.org/pypi/servir-aces)
[![Conda Recipe](https://img.shields.io/badge/recipe-servir--aces-green.svg)](https://github.com/conda-forge/servir-aces-feedstock)
[![image](https://img.shields.io/conda/vn/conda-forge/servir-aces.svg)](https://anaconda.org/conda-forge/servir-aces)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/servir-aces.svg)](https://anaconda.org/conda-forge/servir-aces)

ACES (Agricultural Classification and Estimation Service) is a Python module for generating training data and training machine learning models for remote sensing applications. It provides functionalities for data processing, data loading from Earth Engine, feature extraction, and model training.

## Features

- Data loading and processing from Earth Engine.
- Generation of training data for machine learning models.
- Training and evaluation of machine learning models (DNN, CNN, UNET).
- Inferences of the trained ML/DL models.
- Support for remote sensing feature extraction.
- Integration with Apache Beam for data processing.


## Usage

### Setup
ACES relies on environment variables defined in a `.env` file to configure various aspects of the training process. You can find an example `.env` file named [`.env.example`](https://github.com/SERVIR/servir-aces/blob/main/.env.example). Copy this file and rename it to `config.env` in your project directory. Edit the variables within `config.env` to suit your specific needs. You will need to change several environment settings before running. Below are explained some of the environment configuration.

### Environment Configuration
- **BASEDIR (str)**: The base directory for the project. This is the root directory where all project-related files and subdirectories reside. It serves as the root directory for other folders containing data, outputs, etc.

- **DATADIR (str)**: The directory for your data, and will be used to retrieve data necessary for training experiments. For this you can either specify the Google Cloud Storage (GCS) path starting with `gs://` or the path which can ben either an full absolute path or relative path to `BASEDIR`. This means, for example, the path to your data directory can be either `gs://aces-project/data` or the full absolute path as `/home/aces-project/data` or can be relative path to `BASEDIR`; so if `BASEDIR` is `/home/aces-project` and `DATADIR` is `data`, then the `DATADIR` is constructed as `/home/aces-project/data`. \
Then your training, testing, and validation dataset should be placed as sub-directory inside the `DATADIR` with sub-directory named as "training", "testing", and "validation" respectively. This means, from above example, if `DATADIR` is `gs://aces-project/data` or `/home/aces-project/data`, the training, testing, and validation dataset should be made available at `gs://aces-project/data/training` or `/home/aces-project/data/training`, `gs://aces-project/data/testing` or `/home/aces-project/data/testing`, and `gs://aces-project/data/validation` or `/home/aces-project/data/validation` respectively.

- **OUTPUT_DIR (str)**: This is the directory where all output files, such as trained models, evaluation results, model outpus, and other generated files during training, will be saved. Similar to `DATADIR` above, this can be either an absolute full path or a relative path to the `BASEDIR`. This means, for example, the path to your output directory can be either a full absolute path as `/home/aces-project/output` or can be relative path to `BASEDIR`; so if `BASEDIR` is `/home/aces-project` and `OUTPUT_DIR` is `data`, then the `OUTPUT_DIR` is constructed as `/home/aces-project/output`.

- **MODEL_DIR_NAME (str)**: This is the sub-directory inside the `OUTPUT_DIR` where the output of the trained model and other relevant files. The rationale behind the `MODEL_DIR_NAME` is to provide a versioning mechanism. So, for example, if you're training a DNN model with different `BATCH_SIZE` say 32 and 64, you could run those two experiments with a `MODEL_DIR_NAME` as `dnn_batch_32` and `dnn_batch_64`, which saves those two experiments under that sub-directory name inside the `OUTPUT_DIR`. \
***Note: MODEL_DIR_NAME is used to construct the MODEL_DIR config parameter which can be accessed as config.MODEL_DIR.***

- **MODEL_TYPE (str)**: This variable defines the type of deep learning model you want to train The default is "unet". The available choices are "dnn", "unet", and "cnn".
***Note current version does not expose all the model intracacies through the environment file but future version may include those depending on the need.***

- **FEATURES (str)**: This variable specifies the feature names used in your training data. Each feature should correspond to a band or channel in your data. The features can be either specified as a comma-separated string, newline-separated string, or newline-with-comma separated string.\
For example, `FEATURES` can be specified as comma-separated string as `FEATURES = "red_before, green_before, blue_before, nir_before, red_during, green_during, blue_during, nir_during"` or newline separated string as
```ini
  FEATURES = "red_before
  green_before
  blue_before
  nir_before
  red_during
  green_during
  blue_during
  nir_during"
```
```
  or newline-with-comma separated string as
```
```ini
  FEATURES = "red_before,
  green_before,
  blue_before,
  nir_before,
  red_during,
  green_during,
  blue_during,
  nir_during"
```

- **LABELS (str)**: This variable specifies the target labels used in the training process. Similar to `FEATURES` above, you can specify this as either a comma-separated string, newline-separated string, or newline-with-comma separated string.

- **PATCH_SHAPE (tuple)**: This variable specifies the shape of the single patch (WxH) used for training datasets. The number of channels are taken from `FEATURES`. This is useful for patch based algorithms like U-Net and CNN. This can be specified as a tuple of patch dimension. eg. `PATCH_SHAPE = (256, 256)`

- **TRAIN_SIZE, TEST_SIZE, VAL_SIZE (int)**: These variables define the number of samples used for training, testing, and validation, respectively. This can be speficied as the integer number. For example, `TRAIN_SIZE = 8531`, `TEST_SIZE = 1222`, `VAL_SIZE = 2404`.

- **OUT_CLASS_NUM (int)**: This variable specifies the number of output classes in the classification model. This can be speficied as the integer number. For example, `OUT_CLASS_NUM = 5`.

- **SCALE (int)**: This variable specifies the scale in which the analysis take place. This can be specified as the integer number. For example, `SCALE = 10`.

- **DROPOUT_RATE (float)**: This variable specifies the dropout rate for the model. This should be specified as the float number between 0 and 1. For example, `DROPOUT_RATE = 0.2` Anything below zero would be set to 0 and anything above 1 would be set to 1. If not specified, the `DROPOUT_RATE` would be set to 0.

- **LOSS (str)**: This variable specifies the loss function used for model training. Learn more about the loss function [here](https://keras.io/api/losses/). Use the named representation of the loss function. For example, in order to use `keras.losses.CategoricalCrossentropy`, you should specify `"categorical_crossentropy"`.

- **ACTIVATION_FN (str)**: This variable specifies the activation function to be used for model training. Learn more about the activation function available [here](https://keras.io/api/layers/activations/). Use the named representation of the activation function. For example, in order to use `keras.activations.softmax`, you should specify `"softmax"`.

- **OPTIMIZER (str)**: This variable specifies the optimizer function to be used for model training. Learn more about the optimizer function available [here](https://keras.io/api/optimizers/). Use the named representation of the activation function. For example, in order to use `keras.optimizers.Adam`, you should specify `"adam"`. \
***Note current implementation does not expose all the parameters of the LOSS, ACTIVATION, OPTIMIZER but future version may include those depending on the need.***

- **MODEL_CHECKPOINT_NAME (str)**: This variable specifies the name to be used for the model checkpoints. Learn more about the model checkpoints [here](https://keras.io/api/callbacks/model_checkpoint/). This helps save the Keras model at some frequency.

- **CALLBACK_PARAMETER (str)**: This variable specifies the call metric parameter that needs to be monitored. This is used as the monitor value in the ModelCheckPoint and EarlyStopping Callbacks. Learn more about callbacks [here](https://keras.io/api/callbacks/). If not specified, default value of `"val_loss"` is set.

- **EARLY_STOPPING (bool)**: This variable specifies whether to use EarlyStopping callback or not. Learn more about callbacks [here](https://keras.io/api/callbacks/). The default value is False. If `EARLY_STOPPING` is set to True, the parameter to monitor is the `CALLBACK_PARAMETER` set above, and the `patience` argument is set to 30% of the total epochs defined by `EPOCHS` configuration.

## Tutorial

***Note: We have a Jupyter Notebook Colab example that you can use to run without having to worry too much about setting up locally. You can find the relevant notebook [here](https://github.com/SERVIR/servir-aces/blob/main/notebook/Rice_Mapping_Bhutan_2021.ipynb). Note, however, the resources especially GPU may not be fully available via Colab for unpaid version***

ACES relies on environment variables defined in a `.env` file to configure various aspects of the training process. You can find an example `.env` file named [`.env.example`](https://github.com/SERVIR/servir-aces/blob/main/.env.example). Copy this file and rename it to `config.env` in your project directory. Edit the variables within `config.env` to suit your specific needs.

***Learn how to setup your `BASEDIR`, `DATADIR`, `OUTPUT_DIR` and `MODEL_DIR_NAME` from [Usage](#usage) above.***

For quickly running this, we have already prepared and exported the training datasets. They can be found at the Google Cloud Storage and we will use gsutil to get the dataset in our workspace. The dataset has training, testing, and validation subdirectory. Let's start by downloading these datasets in our workspace. Follow this [link](https://cloud.google.com/storage/docs/gsutil_install) to install gsutil.

*Note: If you're looking to produce your own datasets, you can follow this [notebook](https://colab.research.google.com/drive/1LCFLeSCu969wIW8TD68-j4hfIu7WiRIK?usp=sharing) which was used to produce these training, testing, and validation datasets provided here.*


```sh
mkdir DATADIR
gsutil -m cp -r gs://dl-book/chapter-1/dnn_planet_wo_indices/* DATADIR
```

The parent folder (if you run `gsutil -m cp -r gs://dl-book/chapter-1/* DATADIR`) has several dataset inside it. We use `dnn_planet_wo_indices` here because it has lightweight data and would be much faster to run. If you want to test U-Net model, you can use `unet_256x256_planet_wo_indices` folder instead. Each of these have training, testing, and validation sub-folder inside them.

We need to change some more parameters for this tutorial. We need to change the `MODEL_TYPE`. The default is "unet", let's change it to "dnn" since we will be training using that model.

```ini
MODEL_TYPE = "dnn"
```

Next define the `FEATURES` and the `LABELS` variables. You can refer to the [Usage](#usage) section above to learn more. But this dataset was prepared for the rice mapping application that uses before and during growing season information. So for this dataset here are `FEATURES`.

```ini
FEATURES = "red_before
green_before
blue_before
nir_before
red_during
green_during
blue_during
nir_during"
```

Similarly, since the dataset has a single label, it is going to be as below.

```ini
LABELS = "class"
```

In addition, the training sizes (`TRAIN_SIZE`, `TEST_SIZE`, and `VAL_SIZE`) needs to be changed. For this dataset, we know our size before hand. `aces` also provides handful of functions that we can use to calculate this. See this [notebook](https://colab.research.google.com/drive/12WgDI3ptFZHmcfOw89fmPSsDwO-sXIUH?usp=sharing) to learn more about how to do it. You should also change the `SCALE`, `DROPOUT_RATE`, `BATCH_SIZE`, `EPOCHS`, `LOSS`, `ACTIVATION`, `OUT_CLASS_NUM`, `OPTIMIZER`, `MODEL_CHECKPOINT_NAME` information as below.

```ini
TRAIN_SIZE = 8531
TEST_SIZE = 1222
VAL_SIZE = 2404
SCALE = 10
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
EPOCHS = 30
LOSS = "categorical_crossentropy"
ACTIVATION_FN = "softmax"
OPTIMIZER = "adam"
OUT_CLASS_NUM = 5
MODEL_CHECKPOINT_NAME = "modelCheckpoint"
```

These settings should be good enough to get started. To view the complete settings and what they meant, you can view them [here](https://github.com/SERVIR/servir-aces/blob/main/.env.example).

### Tutorial Configuration

Here's the example configuration file for running this tutorial (`config.env`)

```ini
BASEDIR="/path/to/your/project"
DATADIR="data"
OUTPUT_DIR="output"
MODEL_DIR_NAME="dnn_v1"
MODEL_TYPE="dnn"
FEATURES="red_before
green_before
blue_before
nir_before
red_during
green_during
blue_during
nir_during"
LABELS="class"
PATCH_SHAPE=(256, 256)
TRAIN_SIZE=8531
TEST_SIZE=1222
VAL_SIZE=2404
SCALE=10
DROPOUT_RATE=0.2
BATCH_SIZE=32
EPOCHS=30
LOSS="categorical_crossentropy"
ACTIVATION_FN="softmax"
OPTIMIZER="adam"
# "cropland_etc", "rice", "forest", "urban", "others_water_etc"
OUT_CLASS_NUM=5
MODEL_CHECKPOINT_NAME="modelCheckpoint"
```

### Running the ACES Module
After setting up your config.env file, you can use the ACES module as follows:

```python
from aces.config import Config
from aces.model_trainer import ModelTrainer

config_file = "config.env"
config = Config(config_file, override=True)
trainer = ModelTrainer(config)
trainer.train_model()
```

Once the model training is complete, it will save the trained model, evaluation results, plots, and other relevant files on the `MODEL_DIR` (sub-folder named `MODEL_DIR_NAME` inside the `OUTPUT_DIR`).

### Inference
For inference, you would need to get the images in a right format. To export images in the right way, refer to this [notebook](https://github.com/SERVIR/servir-aces/blob/main/notebook/export_image_for_prediction.ipynb), but for this, we already have the images prepared in the right way. You can download them in a similar manner you downloaded the training datasets. Change the IMAGEDIR to your appropriate directory.

```sh
mkdir IMAGEDIR
gsutil -m cp -r gs://dl-book/chapter-1/images/* IMAGEDIR
```

There are few more settings that needs to be changed before we run the predictions. They are:
- **OUTPUT_NAME (str)**: This is the name of the output prediction for GEE asset, locally (in TF Format) and gcs output (in TFRecord format).

- **GCS_PROJECT (str)**: This is the name of the Google Cloud Project that will be used to push the output from GCS to GEE for the prediction.

- **GCS_BUCKET (str)**: This is the name of the Google Cloud Bucket that will be used to store your prediction and should be inside the `GCS_PROJECT`.

- **EE_OUTPUT_ASSET (str)**: This is the name of the variable that will be used as the output path to the asset in Google Earth Engine (GEE) that is used to push the predictions to.

So in addition to the already defined variables above, these are the additional example configuration for the prediction in the configuration file (`config.env`).

```ini
OUTPUT_NAME = "prediction_dnn_v1"
GCS_PROJECT = "your-gcs-project"
GCS_BUCKET = "your-bucket"
EE_OUTPUT_ASSET = "your-gee-output-asset-path"
```

You will have to run the configuration settings again for the new configuration to take place.

```python
from aces.config import Config

config_file = "config.env"
config = Config(config_file, override=True)
```

We can then start constructing the actual path for the output file using the `OUTPUT_NAME` as, and print it to view.

```python
OUTPUT_IMAGE_FILE = str(config.MODEL_DIR / "prediction" / f"{config.OUTPUT_NAME}.TFRecord")
print(f"OUTPUT_IMAGE_FILE: {OUTPUT_IMAGE_FILE}")
```

Now let's get all the files inside the `IMAGEDIR`, and then separate out our actual image files and the JSON mixer file. The JSON mixer file inside the `IMAGEDIR` is generated when exporting images from GEE as TFRecords. This is a simple JSON file for defining the georeferencing of the patches.

```python
import glob

image_files_list = []
json_file = None

for f in glob.glob(f"{IMGDIR}/*"):
    if f.endswith(".tfrecord.gz"):
        image_files_list.append(f)
    elif f.endswith(".json"):
        json_file = f

# Make sure the files are in the right order.
image_files_list.sort()
```

Next, we will load the trained model and look at the model summary. The trained model is stored within the trained-model subdirectory in the MODEL_DIR.

```python
import tensorflow as tf

print(f"Loading model from {str(config.MODEL_DIR)}/trained-model")
this_model = tf.keras.models.load_model(f"{str(config.MODEL_DIR)}/trained-model")

print(this_model.summary())
```

Now let's get the relevant info from the JSON mixer file.

```python
import json

with open(json_file, encoding='utf-8') as jm: mixer = json.load(jm)

# Get relevant info from the JSON mixer file.
patch_width = mixer["patchDimensions"][0]
patch_height = mixer["patchDimensions"][1]
patches = mixer["totalPatches"]
patch_dimensions_flat = [patch_width * patch_height, 1]
```

Next let's create a TFDataset from our images as:

```python

def parse_image(example_proto):
    columns = [
        tf.io.FixedLenFeature(shape=patch_dimensions_flat, dtype=tf.float32) for k in config.FEATURES
    ]
    image_features_dict = dict(zip(config.FEATURES, columns))
    return tf.io.parse_single_example(example_proto, image_features_dict)


# Create a dataset from the TFRecord file(s).
image_dataset = tf.data.TFRecordDataset(image_files_list, compression_type="GZIP")
image_dataset = image_dataset.map(parse_image, num_parallel_calls=5)

# Break our long tensors into many little ones.
image_dataset = image_dataset.flat_map(
  lambda features: tf.data.Dataset.from_tensor_slices(features)
)

# Turn the dictionary in each record into a tuple without a label.
image_dataset = image_dataset.map(
  lambda data_dict: (tf.transpose(list(data_dict.values())), )
)

image_dataset = image_dataset.batch(patch_width * patch_height)
```

Finally, let's perform the prediction.

```python
predictions = this_model.predict(image_dataset, steps=patches, verbose=1)
print(f"predictions shape: {predictions.shape}")
```

Now let's write this predictions on the file.

```python

from pathlib import Path
import numpy as np

# Create the target directory if it doesn't exist
Path(OUTPUT_IMAGE_FILE).parent.mkdir(parents=True, exist_ok=True)

print(f"Writing predictions to {OUTPUT_IMAGE_FILE} ...")
writer = tf.io.TFRecordWriter(OUTPUT_IMAGE_FILE)

# Every patch-worth of predictions we"ll dump an example into the output
# file with a single feature that holds our predictions. Since our predictions
# are already in the order of the exported data, the patches we create here
# will also be in the right order.
patch = [[], [], [], [], [], []]

cur_patch = 1

for i, prediction in enumerate(predictions):
    patch[0].append(int(np.argmax(prediction)))
    patch[1].append(prediction[0][0])
    patch[2].append(prediction[0][1])
    patch[3].append(prediction[0][2])
    patch[4].append(prediction[0][3])
    patch[5].append(prediction[0][4])


    if i == 0:
        print(f"prediction.shape: {prediction.shape}")

    if (len(patch[0]) == patch_width * patch_height):
        if cur_patch % 100 == 0:
            print("Done with patch " + str(cur_patch) + " of " + str(patches) + "...")

        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                "prediction": tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=patch[0])),
                "cropland_etc": tf.train.Feature(
                    float_list=tf.train.FloatList(
                        value=patch[1])),
                "rice": tf.train.Feature(
                    float_list=tf.train.FloatList(
                        value=patch[2])),
                "forest": tf.train.Feature(
                    float_list=tf.train.FloatList(
                        value=patch[3])),
                "urban": tf.train.Feature(
                    float_list=tf.train.FloatList(
                        value=patch[4])),
                "others_etc": tf.train.Feature(
                    float_list=tf.train.FloatList(
                        value=patch[5])),
                }
            )
        )

        # Write the example to the file and clear our patch array so it"s ready for
        # another batch of class ids
        writer.write(example.SerializeToString())
        patch = [[], [], [], [], [], []]
        cur_patch += 1

writer.close()
```
### Uploading Predictions to GCP and GEE
Now we have write the prediction to the `OUTPUT_IMAGE_FILE`. You can upload this to GEE for visualization. To do this, you will need to upload to GCP and then to GEE.

You can upload to GCP using gsutil. The `OUTPUT_GCS_PATH` can be any path inside the `GCS_BUCKET` (e.g. `OUTPUT_GCS_PATH = f"gs://{config.GCS_BUCKET}/{config.OUTPUT_NAME}.TFRecord"`)

```sh
gsutil cp "{OUTPUT_IMAGE_FILE}" "{OUTPUT_GCS_PATH}"
```

Once the file is available on GCP, you can then upload to earthengine using `earthengine` commnad line.

```sh
earthengine upload image --asset_id={config.EE_OUTPUT_ASSET}/{config.OUTPUT_NAME} --pyramiding_policy=mode {OUTPUT_GCS_PATH} {json_file}
```

***Note: The inferencing is also available on this [notebook](https://github.com/SERVIR/servir-aces/blob/main/notebook/Rice_Mapping_Bhutan_2021.ipynb), scroll to `Inference using Saved U-Net Model` or `Inference using Saved DNN Model` depending upon which model you're using.***

## Contributing
Contributions to ACES are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

For Developers, to bump the version, refer [here](https://github.com/opengeos/cookiecutter-pypackage).

## License
This project is licensed under the [GNU General Public License v3.0](https://github.com/SERVIR/servir-aces/blob/main/LICENSE).
