# Usage

***Note:*** To make running things easier, we also have prepared `notebook`. You can find the detailed example on running things using this [main](https://github.com/SERVIR/servir-aces/blob/main/notebook/Rice_Mapping_Bhutan_2021.ipynb) notebook. This and several other notebooks are available on the `notebook` folder implementing DL methods. This notebook runs on colab, and most of the installation etc are taken care of.

To use servir-aces, you need training datasets. If you want to use/produce produce your own datasets, you can follow this [notebook](https://github.com/SERVIR/servir-aces/blob/main/notebook/generate_training_patches.ipynb) which goes in detail on how to produce the datasets for training, testing, and validation.

For quickly getting started to test the library, we have already prepared and exported the training datasets. They can be found at the google cloud storage and we will use `gsutil` to get the dataset in our workspace. Learn how you can install `gsutil` [here](https://cloud.google.com/storage/docs/gsutil_install). Let's start by downloading these datasets in our workspace. Navigate to your work folder and run.

```
gsutil -m cp -r gs://dl-book/chapter-1 .
```

Once you run this, you can find `unet_256x256_planet_wo_indices` and `dnn_planet_wo_indices` which are the prepared training data for running the U-Net and DNN model respectively. Each of these folder have sub-folder called `training`, `testing`, and `validation` which are the training, testing and validation dataset.


All the settings are provided through the `.env` file. A sample example of the `.env` file can be found [here](https://github.com/SERVIR/servir-aces/blob/main/.env.example). You can copy this and rename to `config.env` at your directory. There are several parameters that needs to be changed. Let's look at some of them here. Note these directory should exists before you can run them.


```
BASEDIR = "/content/"
OUTPUT_DIR = "/content/output"
```

We will start by training a U-Net model using the `unet_256x256_planet_wo_indices` dataset that we just downloaded. Let's go ahead and change our DATADIR in the `config.env` file as below.

```
DATADIR = "/content/datasets/unet_256x256_planet_wo_indices"
```

These datasets have RGBN from Planetscope mosiac. For this example, we will use the library to map the rice fields, and we use growing season and pre-growing season information. Thus, we have 8 optical bands, namely `red_before`, `green_before`, `blue_before`, `nir_before`, `red_during`, `green_during`, `blue_during`, and  `nir_during`. This information is provided via the `FEATURES` variable which already defaults to what we want for this example; see [here](https://github.com/SERVIR/servir-aces/blob/main/.env.example#L26).

In adidition, you can use `USE_ELEVATION` and `USE_S1` config to include the topographic and radar information if your dataset has any (or just put them in the `FEATURES` above). Since this datasets don't have topographic and radar features, so we won't be settting these config values. Similarly, these datasets are tiled to 256x256 pixels, so let's also change that.

```
USE_ELEVATION = False
USE_S1 = False

PATCH_SHAPE = (256, 256)
```

You can then change the `MODEL_TYPE` which you are running. The default is "unet". You can change to "dnn" if you are using that model.

*Note current version does not expose all the model intracacies through the environment file but future version may include those depending on the need.*

```
MODEL_TYPE = "unet"
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

Similarly, since the dataset has a single label, it is going to be

```
LABELS = ["class"]
```

Next, we need to calculate the size of the training, testing and validation dataset. For this, we know our size before hand. But `aces` also provides handful of functions that we can use to calculate this. See this [notebook](https://github.com/SERVIR/servir-aces/blob/main/notebook/count_sample_size.ipynb) to learn more about how to do it. We will also change the `BATCH_SIZE` to 32; if you have larger memory available, you can increase the `BATCH_SIZE`. You can run for longer `EPOCHS` by changing the `EPOCHS` paramter; we will keep it to 30 for now.

```
# Sizes of the training and evaluation datasets.
TRAIN_SIZE = 8531
TEST_SIZE = 1222
VAL_SIZE = 2404
BATCH_SIZE = 32
EPOCHS = 30
```

Finally, you can change the `MODEL_DIR_NAME`. This is the directory name where you want to save the output of the training and other relevant files. This is used to construct the `MODEL_DIR`, which is constructed as `OUTPUT_DIR + MODEL_DIR_NAME`.

```
MODEL_DIR_NAME = "unet_v1"
```

These settings should be good enough to get started. To view the complete settings and what they meant, you can view them [here](https://github.com/SERVIR/servir-aces/blob/main/.env.example).

To use servir-aces in a project.

```python

from aces import Config, ModelTrainer

if __name__ == "__main__":
    config_file = "config.env"
    config = Config(config_file)
```

Most of the config in the `config.env` is now available via the config instance. Let's check few of them here.

```python
print(config.TRAINING_DIR, config.OUTPUT_DIR, config.BATCH_SIZE, config.TRAIN_SIZE)
```

Next, let's make an instance of the `ModelTrainer` object. The `ModelTrainer` class provides various tools for training, buidling, compiling, and running specified deep learning models. (continue from above)

```python
    trainer = ModelTrainer(config, seed=42)
```

[`ModelTrainer`](https://servir.github.io/servir-aces/model_trainer/) class provides various functionality. We will use `train_model` function that helps to train the model using the provided configuration settings.

This method performs the following steps:
- Configures memory growth for TensorFlow.
- Creates TensorFlow datasets for training, testing, and validation.
- Builds and compiles the model.
- Prepares the output directory for saving models and results.
- Starts the training process.
- Evaluates and prints validation metrics.
- Saves training parameters, plots, and models.

continue from above

```python
    trainer.train_model()
```

***Note:*** this takes a while to run and needs access to the GPU to run it faster. If you don't have access to GPU or don't want to wait while its running, we provide you with the trained model via the Google Cloud Storage. After you get the data via `gsutil` above, you will see a folder called `models` inside it. For the U-Net, you can download the `unet_v1` folder and put it inside the `MODEL_DIR`, and for DNN, you can download the `dnn_v1` folder and place it inside the `MODEL_DIR`.
