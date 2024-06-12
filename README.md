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

***Note: We have a Jupyter Notebook Colab example that you can use to run without having to worry too much about setting up locally. You can find the relevant notebook [here](https://github.com/SERVIR/servir-aces/blob/main/notebook/Rice_Mapping_Bhutan_2021.ipynb). Note, however, the resources especially GPU may not be fully available via Colab***

***For detailed usage instructions, please refer to the [Usage Document](docs/usage.md)***

Define all your configuration in `.env` file. An example of the file is provided as [`.env.example`](https://github.com/SERVIR/servir-aces/blob/main/.env.example) file, which you can copy and rename as `config.env`.

You will need to change several environment settings before running. Some of them are:

BASEDIR (str): The base directory. \
DATADIR (str): The directory for your data which will be used in training experiments.

For quickly running this, we have already prepared and exported the training datasets. They can be found at the Google Cloud Storage and we will use gsutil to get the dataset in our workspace. The dataset has training, testing, and validation subdirectory. Let's start by downloading these datasets in our workspace. Follow this [link](https://cloud.google.com/storage/docs/gsutil_install) to install gsutil.

*Note: If you're looking to produce your own datasets, you can follow this [notebook](https://colab.research.google.com/drive/1LCFLeSCu969wIW8TD68-j4hfIu7WiRIK?usp=sharing) which was used to produce these training, testing, and validation datasets provided here.*

```sh
mkdir DATADIR
gsutil -m cp -r gs://dl-book/chapter-1/dnn_planet_wo_indices/* DATADIR
```

The parent folder (if you run `gsutil -m cp -r gs://dl-book/chapter-1/* DATADIR`) has several dataset inside it. We use `dnn_planet_wo_indices` here because it is lightweight data and would be much faster to run. If you want to test U-Net model, you can use `unet_256x256_planet_wo_indices` folder instead. Each of these have training, testing, and validation sub-folder inside them.

The datadir is constructed as `DATADIR = BASEDIR + DATADIR` \
OUTPUT_DIR (str): The base directory where the output would be saved.

You can then change the `MODEL_TYPE` which you are running. The default is "unet", so we need to change it to "dnn" since we will be training using that model.

*Note current version does not expose all the model intracacies through the environment file but future version may include those depending on the need.*

```
MODEL_TYPE = "dnn"
```

Next define the `FEATURES` and the `LABELS` variables. This dataset was prepared for the rice mapping application that uses before and during growing season information. So for this dataset, here are the example `FEATURES`. Note the `FEATURES` should be in the same format as shown below

```
FEATURES = "red_before
green_before
blue_before
nir_before
red_during
green_during
blue_during
nir_during"
```

Similarly, since the dataset has a single label, it is going to be as below. If you have mutiple labels, be sure to format them as the `FEATURES` above.

```
LABELS = "class"
```

While the DNN would not need this setting, if you want to run U-Net model, the `PATCH_SHAPE` for the training datasets needs to be changed as well:

```
PATCH_SHAPE = (256, 256)
```

In addition, the training sizes (`TRAIN_SIZE`, `TEST_SIZE`, and `VAL_SIZE`) needs to be changed. For this dataset, we know our size before hand. But `aces` also provides handful of functions that we can use to calculate this. See this [notebook](https://colab.research.google.com/drive/12WgDI3ptFZHmcfOw89fmPSsDwO-sXIUH?usp=sharing) to learn more about how to do it. You can also change the `BATCH_SIZE` and `EPOCHS` information.

```
TRAIN_SIZE = 8531
TEST_SIZE = 1222
VAL_SIZE = 2404
BATCH_SIZE = 32
EPOCHS = 30
```

Finally, you can change the `MODEL_DIR_NAME`. This is the directory name where you want to save the output of the training and other relevant files. This is used to construct the `MODEL_DIR`, which is constructed as `OUTPUT_DIR + MODEL_DIR_NAME`.

These settings should be good enough to get started. To view the complete settings and what they meant, you can view them [here](https://github.com/SERVIR/servir-aces/blob/main/.env.example).

After all the settings been correctly setup, here's an example of how to use the ACES module:

```python
from aces.config import Config
from aces.model_trainer import ModelTrainer

config_file = "config.env"
config = Config(config_file)
trainer = ModelTrainer(config)
trainer.train_model()
```

Once the model is finished running, it saves the trained model, evaluate the results on the testing dataset and saves it, produces the plots and saves it, and saves any other needed item on the `MODEL_DIR`, which is constructed as `OUTPUT_DIR + MODEL_DIR_NAME`.

For inference, you would need to get the images in a right format. To export images in the right way, refer to this [notebook](https://github.com/SERVIR/servir-aces/blob/main/notebook/export_image_for_prediction.ipynb), but for this, we already have the images prepared in the right way. You can download them in a similar manner you downloaded the training datasets.

```sh
mkdir IMAGEDIR
gsutil -m cp -r gs://dl-book/chapter-1/images/* IMAGEDIR
```

There are few more settings that needs to be changed before we run the predictions. One of that would be changing the `OUTPUT_NAME`. This is the name of the output prediction for GEE asset, locally (in TF Format) and gcs output (in TFRecord format). You would also need a GCP Project to push the output from GCS to GEE for the prediction. Similarly, you need GCP Bucket to store your prediction, and finally the EE_OUTPUT_ASSET to the as an output path for the prediction asset.

```
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
