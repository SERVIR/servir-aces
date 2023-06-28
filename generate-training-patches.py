# -*- coding: utf-8 -*-

"""
This script generates training data for a machine learning model.
The data is split into training, validation, and test sets using the Apache Beam library.
Earth Engine Python API is used to load and manipulate geospatial data.
The script uses TensorFlow to serialize the data into TFRecord format.
"""

# Necessary imports
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import ee
import numpy as np
from typing import List

# Before running this script, you need to authenticate to Google Cloud:
# https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python#before-you-begin

# ToDo: Is there a way to use seed value in the split?

__all__ = ['TrainingDataGenerator']

class TrainingDataGenerator:
    """
    A class to generate training data for machine learning models.
    This class utilizes Apache Beam for data processing and Earth Engine for handling geospatial data.
    """
    def __init__(self, include_after: bool = False):
        """
        Constructor for the TrainingDataGenerator class.
        
        Parameters:
        include_after (bool): If True, includes 'after' images in the generated data. Default is False.
        """
        self.output_bucket = "bhutan-aces"
        self.kernel_size = 128
        self.step = 5
        self.grace = 10
        self.scale = 90
        self.include_after = include_after
        self.test_ratio = 0.2
        self.validation_ratio = 0.2

    def load_data(self) -> None:
        """
        Load the necessary data from Earth Engine and prepare it for use.
        """
        ee.Initialize()

        self.l1 = ee.FeatureCollection("projects/servir-sco-assets/assets/Bhutan/BT_Admin_1")
        self.paro = self.l1.filter(ee.Filter.eq("ADM1_EN", "Paro"))

        self.sample_locations = ee.FeatureCollection("projects/servir-sco-assets/assets/Bhutan/ACES_2/paro_2021_all_class_samples")
        self.sample_size = self.sample_locations.size().getInfo()
        print("Sample size:", self.sample_size)
        self.sample_locations_list = self.sample_locations.toList(self.sample_size + self.grace)

        self.label = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/paro_2021_all_class_label").rename("class").unmask(0, False)
        # self.other = self.label.remap([0, 1], [1, 0]).rename(["other"])

        self.composite_after = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Paro_Rice_Composite_2021/composite_after")
        self.composite_before = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Paro_Rice_Composite_2021/composite_before")
        self.composite_during = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Paro_Rice_Composite_2021/composite_during")

        self.composite_before = self.composite_before.regexpRename("$(.*)", "_before")
        self.composite_after = self.composite_after.regexpRename("$(.*)", "_after")
        self.composite_during = self.composite_during.regexpRename("$(.*)", "_during")

        self.srtm = ee.Image("USGS/SRTMGL1_003")
        self.slope = ee.Algorithms.Terrain(self.srtm).select("slope")
        
        stats = TrainingDataGenerator.calculate_min_max_statistics(self.srtm, self.paro, self.scale)
        self.srtm = self.srtm.unitScale(stats.get("elevation_min"), stats.get("elevation_max"))
        self.slope = self.slope.unitScale(0, 90)

        if self.include_after:
            self.image = self.composite_before.addBands(self.composite_during) \
                             .addBands(self.composite_after).addBands(self.label).toFloat()
        else:
            self.image = self.composite_before.addBands(self.composite_during).addBands(self.label).toFloat()
                         
        self.selectors = self.image.bandNames().getInfo()

    @staticmethod
    def calculate_min_max_statistics(image: ee.Image, geometry: ee.FeatureCollection, scale: int = 30) -> ee.Dictionary:
        """
        Calculate min and max of an image over a specific region.

        Parameters:
        image (ee.Image): The image to calculate statistics on.
        geometry (ee.FeatureCollection): The region to calculate statistics over.
        scale (int, optional): The scale, in meters, of the projection to compute statistics in. Default is 30.

        Returns:
        ee.Dictionary: A dictionary containing the min and max of the image.
        """
        reducers = ee.Reducer.mean() \
            .combine(reducer2=ee.Reducer.min(), sharedInputs=True) \
            .combine(reducer2=ee.Reducer.max(), sharedInputs=True)

        stats = image.reduceRegion(
            reducer=reducers,
            geometry=geometry,
            scale=scale,
            maxPixels=1E13
        )

        return stats

    @staticmethod
    def yield_sample_points(index, sample_locations: ee.List) -> List:
        import ee
        ee.Initialize()
        point = ee.Feature(sample_locations.get(index)).geometry().getInfo()
        return point["coordinates"]

    @staticmethod
    def get_training_patch(coords: List[float], image: ee.Image, bands: List[str] = [],
                           scale: int = 5, patch_size: int = 128) -> np.ndarray:
        """Get a training patch centered on the coordinates."""
        import ee
        ee.Initialize()
        from google.api_core import exceptions, retry
        import requests
        import numpy as np
        from typing import List
        import io

        @retry.Retry()
        def get_patch(image: ee.Image, region: ee.Geometry, bands: List[str], patch_size: int) -> np.ndarray:
            """Get the patch of pixels in the geometry as a Numpy array."""
            # Create the URL to download the band values of the patch of pixels.
            url = image.getDownloadURL({
                'region': region,
                'dimensions': [patch_size, patch_size],
                'format': "NPY",
                'bands': bands,
            })
            # Download the pixel data. If we get "429: Too Many Requests" errors,
            # it's safe to retry the request.
            response = requests.get(url)
            if response.status_code == 429:
                # The retry.Retry library only works with `google.api_core` exceptions.
                raise exceptions.TooManyRequests(response.text)
                # Still raise any other exceptions to make sure we got valid data.
            response.raise_for_status()

            # Load the NumPy file data and return it as a NumPy array.
            return np.load(io.BytesIO(response.content), allow_pickle=True)

        point = ee.Geometry.Point(coords)
        region = point.buffer(scale * patch_size / 2, 1).bounds(1)
        return get_patch(image, region, bands, patch_size)

    @staticmethod
    def serialize(patch: np.ndarray) -> bytes:
        import tensorflow as tf

        features = {
            name: tf.train.Feature(
                float_list=tf.train.FloatList(value=patch[name].flatten())
            )
            for name in patch.dtype.names
        }
        example = tf.train.Example(features=tf.train.Features(feature=features))
        return example.SerializeToString()

    @staticmethod
    def split_dataset(element, num_partitions: int, validation_ratio: float = 0.2, test_ratio: float = 0.2) -> int:
        import random
        
        weights = [1 - validation_ratio - test_ratio, validation_ratio, test_ratio]
        return random.choices([0, 1, 2], weights)[0]

    def generate_training_data(self) -> None:
        """
        Use Apache Beam to generate training, validation, and test data from the loaded data.
        """
        beam_options = PipelineOptions([], direct_num_workers=0, direct_running_mode="multi_processing", runner="DirectRunner")
        with beam.Pipeline(options=beam_options) as pipeline:
            training_data, validation_data, test_data = (
                pipeline
                | "Create range" >> beam.Create(range(0, self.sample_size, 1))
                | "Yield sample points" >> beam.Map(TrainingDataGenerator.yield_sample_points, self.sample_locations_list)
                | "Get patch" >> beam.Map(TrainingDataGenerator.get_training_patch, self.image, self.selectors, self.scale, self.kernel_size)
                | "Serialize" >> beam.Map(TrainingDataGenerator.serialize)
                | "Split dataset" >> beam.Partition(TrainingDataGenerator.split_dataset, 3, validation_ratio=self.validation_ratio, test_ratio=self.test_ratio)
            )

            # Write the datasets to TFRecord files in the output bucket
            training_data | "Write training data" >> beam.io.WriteToTFRecord(
                f"gs://{self.output_bucket}/experiments_paro_{self.kernel_size}x{self.kernel_size}_before_during{'_after' if self.include_after else ''}_training/training", file_name_suffix=".tfrecord.gz"
            )
            validation_data | "Write validation data" >> beam.io.WriteToTFRecord(
                f"gs://{self.output_bucket}/experiments_paro_{self.kernel_size}x{self.kernel_size}_before_during{'_after' if self.include_after else ''}_validation/validation", file_name_suffix=".tfrecord.gz"
            )
            test_data | "Write test data" >> beam.io.WriteToTFRecord(
                f"gs://{self.output_bucket}/experiments_paro_{self.kernel_size}x{self.kernel_size}_before_during{'_after' if self.include_after else ''}_testing/testing", file_name_suffix=".tfrecord.gz"
            )

    def run(self) -> None:
        """
        Run the training data generation process.
        """
        self.load_data()
        self.generate_training_data()


if __name__ == "__main__":
    print("Program started..")
    generator = TrainingDataGenerator()
    generator.run()
    print("\nProgram completed.")
