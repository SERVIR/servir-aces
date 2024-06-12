---
title: "servir-aces: A Python Package for Training Machine Learning Models for Remote Sensing Applications"
tags:
    - Python
    - Remote Sensing
    - Crop Mapping
    - Machine Learning
    - Tensorflow
    - Google Earth Engine
    - Mapping
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
date: 30 April 2024
bibliography: paper.bib
---

# Summary

**`servir-aces`** Agricultural Classification and Estimation Service (ACES) is a Python package for generating training data using highly parallelized [apache-beam](https://beam.apache.org/) and [Google Earth Engine (GEE)](https://earthengine.google.com/) [@Gorelick2017] workflows as well as for training various Machine Learning (ML) and Deep Learning (DL) models for Remote Sensing Applications [@Mayer2023], [@bhandari2024comparing].

# Statement of Need

Despite robust platforms, specialized technical knowledge is required to set up and run various Machine Learning (ML) and Deep Learning (DL) models, making it difficult for many development practitioners, scientists, and domain experts to implement them. The **`servir-aces`** Python package is designed to address this challenge by significantly lowering the barrier for users to export training data and both train and run DL models using cloud-based technology with their GEE workflows. Several examples are provided via runnable notebook to make it easier for scientists to utilize this emerging field of DL.

With petabytes of data available via GEE, and integration of the TensorFlow (TF) platfrom, models trained in TF can be easily loaded into GEE. This package provides functionalities for 1) data processing; 2) data loading from GEE; 3) feature extraction, 4) model training, and 5) model inference. The combination of TF and GEE has enabled several large scale ML and DL Remote Sensing applications, including Wetland Area Mapping [@Bakkestuen2023], Crop Type Mapping [@Bakkestuen2023], Surface Water Mapping [@Mayer2021], and Urban Mapping [@Parekh2021]. However, these applications tend to be developed ad-hoc without using a common package.

Several unified libraries like torchgeo [@Stewart_TorchGeo_Deep_Learning_2022] and [rastervision](https://github.com/azavea/raster-vision) exists, but they are primarily targeted for PyTorch user community. Some efforts for GEE & TensorFlow users, such as geemap [@Wu2020], are mostly for classical ML approaches like Random Forest, while [geospatial-ml](https://github.com/opengeos/geospatial-ml) has not seen much development since its inception. Thus, there is a need for unified libraries to train DL models integrating the GEE & TensorFlow user community.

**`servir-aces`** addresses this need by 1) Offering a streamlined application of commonly employed architectures (CNN, DNN, and U-NET); 2) Allowing end-users to rapidly adjust a wide range of model parameters for these common architectures, including activation functions, optimizers, loss functions, early stopping, dropout rate, batch size etc.; 3) More efficiently and effectively connecting across the Google Cloud ecosystem, linking Google Cloud, improved methods of parallelization via Apache beam, Vertex AI, TensorFlow, and Google Earth Engine; and 4) Enabling broader development and incorporation of several methods through the package's utility functions, such as providing a collated set of evaluation metrics for easier model performance comparisons, a class for generating Remote Sensing features essential for scientific community, and utility functionality for Apache Beam and Earth Engine. Although **`servir-aces`** was originally developed for agricultural-related applications, the library has been further developed to work for any kind of DL image segmentation tasks.

# **`servir-aces`** Audience

**`servir-aces`** is intended for development practitioner, researchers, scientists, software developers, and students who would like to utilize various freely available Earth Observation (EO) data using cloud-based GEE and TF ecosystem to perform large scale ML/DL related Remote Sensing applications.

We also provide several notebook examples to showcase the usage of the **`servir-aces`**. Here we show how **`servir-aces`** can be used for crop-mapping related application. Ideally, the same process can be repeated for any kind of the image segmentation task.

# **`servir-aces`** Functionality

The major high-level functionality of the **servir-aces** packages are:
- Data loading and processing from GEE.
- Generation of training data for various ML and DL models.
- Training and evaluation of ML/DL Models.
- Inferences of the trained ML/DL models.
- Support for remote sensing feature extraction.
- Integration with Apache Beam for data processing and parallelization.

The key functionality of **`servir-aces`** is organized into several modules:

-   [data_processor](https://servir.github.io/servir-aces/data_processor/): this module provides functionality for data input/output and preprocessing for the image segmentation project.

-   [model_builder](https://servir.github.io/servir-aces/model_builder/): this module provides functionality for creating and compiling various Neural Network Models, including DNN, CNN, U-Net.

-   [model_trainer](https://servir.github.io/servir-aces/model_trainer/): this module provides functionality for training, buidling, compiling, and running specified deep learning models.

-   [metrics](https://servir.github.io/servir-aces/metrics/): this module provides a host of statstical metrics, standard within the field, for evaluating model performance and provide utility functions for plotting and visualizing model metrics during training.

-   [ee_utils](https://servir.github.io/servir-aces/ee_utils/): this module for providing utility functions to handle GEE API information and authentication requests.

-   [remote_sensing](https://servir.github.io/servir-aces/remote_sensing/): this module provides various static methods to compute Remote Sensing indices for analysis.

# **`servir-aces`** Funding
This research was funded through the US Agency for International Development (USAID) and NASA initiative Cooperative Agreement Number: AID486-A-14-00002. Individuals affiliated with the University of Alabama in Huntsville (UAH) are funded through the NASA Applied Sciences Capacity Building Program, NASA Cooperative Agreement: 80MSFC22M004.

# **`servir-aces`** Acknowledgement
The authors would like to thank NASA’s Applied Sciences Program and Capacity Building Program, specifically Dr. Nancy Searby. We also want to thank the [SERVIR](https://servirglobal.net/) program especially Dan Irwin, Dr. Ashutosh Limaye, and Eric Anderson. Additionally, we would like to thank the USAID especially Dr. Pete Epanchin. We would also like to thank UAH specifically Dr. Rob Griffin and the [Lab for Applied Science (LAS)](https://www.uah.edu/essc/laboratory-for-applied-science) as well as [SERVIR's Geospatial Artificial Intelligence Working Group (Geo-AI WG)](https://tinyurl.com/servir-geo-ai-wg) for their support and collaboration over the years. Finally, wse are indebted to Dr. Nick Clinton from the Google Earth Outreach Team for the support.

# References
