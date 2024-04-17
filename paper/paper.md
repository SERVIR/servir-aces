---
title: "servir-aces: A Python Package for Training Machine Learning Models for Remote Sensing Applications"
tags:
    - Python
    - Remote Sensing
    - Crop Mapping
    - Machine Learning
    - Tensorflow
    - Google Earth Engine
    - mapping
    - Jupyter notebook
authors:
    - name: Biplov Bhandari
      orcid: 0000-0001-6169-8236
      affiliation: "1, 2"
    - name: Timothy Mayer
      orcid: 0000-0001-9489-9392
      affiliation: "1, 2"
affiliations:
    - name: Earth System Science Center, The University of Alabama in Huntsville, 320 Sparkman Drive, Huntsville, AL 35805, USA
      index: 1
    - name: SERVIR Science Coordination Office, NASA Marshall Space Flight Center, 320 Sparkman Drive, Huntsville, AL 35805, USA
      index: 2
date: 16 April 2024
bibliography: paper.bib
---

# Summary

**`servir-aces`** (ACES for Agricultural Classification and Estimation Service) is a Python package for generating training data using highly parallelized [apache-beam](https://beam.apache.org/) and [Google Earth Engine](https://earthengine.google.com/) [@Gorelick2017] workflows as well as for training various Machine Learning (ML) and Deep Learning (DL) Models for Remote Sensing Applications [@Mayer2023], [@Bhandari2024]. With various type of data available via GEE, and the smooth integration between GEE and TensorFlow (TF); models trained in TF can be easily loaded into GEE. This package provides functionalities for data processing, data loading from Earth Engine, feature extraction, and model training. The use of combining TF and GEE for has enabled several large scale ML and DL Remote Sensing Applications. Some of them include Wetland Areas Mapping [@Bakkestuen2023], Crop Type Mapping [@Poortinga2021], Surface Water Mapping [@Mayer2021], and Urban Mapping [@Parekh2021].

Despite prevalent research, specialized technical knowledge is required to setup and run various ML/DL models. Many of the practitioner/ scientist find it difficult to use DL framework. The **`servir-aces`** Python package is created to fill this grap. It makes it easier to export training data and train a DL models using cloud-based technology using GEE. Several examples are provided to make it easier for scientists to use it.

# **`servir-aces`** Audience

**`servir-aces`** is intended for development practitioner, researchers, and student who would like to utilize various freely available Earth Observation (EO) data using cloud-based GEE and TensorFlow ecosystem to perform large scale ML/DL related Remote Sensing Applications.

We also provide several notebook examples to showcase the usage of the **`servir-aces`**. Here we show how **`servir-aces`** can be used for crop-mapping related application. Ideally, the same process can be repeated for any kind of the image segmentation task.

# **`servir-aces`** Functionality

The major high-level functionality of the **servir-aces** packages are:
- Data loading and processing from GEE.
- Generation of training data for various Machine Learning (ML) and Deep Learning (DL) Models.
- Training and evaluation of ML/DL Models.
- Support for remote sensing feature extraction.
- Integration with Apache Beam for data processing.

The key functionality of **`servir-aces`** is organized into several modules:

-   [data_processor](https://servir.github.io/servir-aces/data_processor/): this module provides functions for data input/output and preprocessing for the image segmentation project.

-   [model_builder](https://servir.github.io/servir-aces/model_builder/): this module provides functionality for creating and compiling various Neural Network Models, including DNN, CNN, U-Net.

-   [model_trainer](https://servir.github.io/servir-aces/model_trainer/): this module provides functionality for training, buidling, compiling, and running specified deep learning models.

-   [metrics](https://servir.github.io/servir-aces/metrics/): this module provides custom metrics for evaluating model performance and provide utility functions for plotting and visualizing model metrics during training.

-   [ee_utils](https://servir.github.io/servir-aces/ee_utils/): a module for providing utility functions to handle Earth Engine API information and make authenticated requests.

-   [remote_sensing](https://servir.github.io/servir-aces/remote_sensing/): a module that provides various static methods to compute various remote sensing indices and analyse them.

# References