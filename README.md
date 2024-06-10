# ACES (Agricultural Classification and Estimation Service)

[![image](https://img.shields.io/pypi/v/servir-aces.svg)](https://pypi.python.org/pypi/servir-aces)

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
cd DATADIR
!gsutil -m cp -r gs://dl-book/chapter-1/* DATADIR
```

The folder has several dataset inside it. If you're running a U-Net model, you can use the sub-folder `unet_256x256_planet_wo_indices`; for DNN model, you can use the sub-folder `dnn_planet_wo_indices`.

The datadir is constructed as `DATADIR = BASEDIR + DATADIR` \
OUTPUT_DIR (str): The base directory where the output would be saved.

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

The `PATCH_SHAPE` for the training datasets needs to be changed as well:

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

if __name__ == "__main__":
    config_file = "config.env"
    config = Config(config_file)
    trainer = ModelTrainer(config)
    trainer.train_model()
```

Once the model is finished running, it saves the trained model, evaluate the results on the testing dataset and saves it, produces the plots and saves it, and saves any other needed item on the `MODEL_DIR`, which is constructed as `OUTPUT_DIR + MODEL_DIR_NAME`.

For inference, refer to this [notebook](https://colab.research.google.com/drive/1wfzvIcpkjI4lT1oEADehD_Kp6iJD6hWr?authuser=2#scrollTo=hdFXAWSM7vKQ), scroll to `Inference using Saved U-Net Model` on how to do it.

## Contributing
Contributions to ACES are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License
This project is licensed under the [GNU General Public License v3.0](https://github.com/SERVIR/servir-aces/blob/main/LICENSE).
