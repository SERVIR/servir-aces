# Installation

## Earth Engine Account

Google Earth Engine (GEE) is used for data processing to get it ready in a usable TF Format. You can [sign up](https://earthengine.google.com/signup/) for a [Google Earth Engine](https://earthengine.google.com/) account.

![signup](https://i.imgur.com/ng0FzUT.png)

## Install from PyPI

**servir-aces** is available on [PyPI](https://pypi.org/project/servir-aces/). To install **servir-aces**, run this command in your terminal:

```bash
pip install servir-aces
```
<!--
## Install from conda-forge

**servir-aces** is also available on [conda-forge](https://anaconda.org/conda-forge/servir-aces). If you have
[Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/free/miniconda) installed on your computer, you can install servir-aces using the following command:

```bash
conda install servir-aces -c conda-forge
```

The geemap package has some optional dependencies, such as [apache-beam](https://beam.apache.org/). It is highly recommended that you create a fresh conda environment to install servir-aces. Follow the commands below to set up a conda env and install geemap:

```bash
conda create -n servir-aces python=3.11
conda activate servir-aces
conda install -n base mamba -c conda-forge
mamba install servir-aces -c conda-forge
```

-->

The optional dependencies can be installed using one of the following:

-   `pip install "servir-aces[extra]"`: installing all optional dependencies.

## Install from GitHub

To install the development version from GitHub using [Git](https://git-scm.com/), run the following command in your terminal:

```bash
pip install git+https://github.com/SERVIR/servir-aces
```
