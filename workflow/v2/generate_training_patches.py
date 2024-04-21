# -*- coding: utf-8 -*-

"""
This script generates training data for a machine learning model.
The data is split into training, validation, and test sets using the Apache Beam library.
Earth Engine Python API is used to load and manipulate geospatial data.
The script uses TensorFlow to serialize the data into TFRecord format.
"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

import ee
import argparse

try:
    from aces.ee_utils import EEUtils
    from aces.utils import TFUtils, Utils
    from aces.config import Config
except ModuleNotFoundError:
    print("ModuleNotFoundError: Attempting to import from parent directory.")
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from aces.ee_utils import EEUtils
    from aces.utils import TFUtils, Utils
    from aces.config import Config

# Before running this script, you need to authenticate to Google Cloud:
# https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python#before-you-begin

# ToDo: Is there a way to use seed value in the split?

__all__ = ["TrainingDataGenerator"]

class TrainingDataGenerator:
    """
    A class to generate training data for machine learning models.
    This class utilizes Apache Beam for data processing and Earth Engine for handling geospatial data.
    """
    def __init__(self, config, split_dir: str = "training", **kwargs):
        """
        Constructor for the TrainingDataGenerator class.

        Parameters:
        config (Config): The configuration object (aces.Config) containing the necessary settings.
        split_dir (str): The split directory. Default is training. Choices are training, testing, and validation.
        **kwargs: Additional keyword arguments to pass to the class.
        Additional keyword arguments:
        - test_ratio (float): The test ratio. Default is 0.1.
        - validation_ratio (float): The validation ratio. Default is 0.2.
        - seed (int): The seed value for random number generation. Default is 100.
        - use_service_account (bool): Whether to use the service account for authentication. Default is False.
        - label (str or ee.Image): The label dataset to load from. Default is the Bhutan ACES 2 dataset.
        - image (str or ee.Image): The image dataset to load from. Default is the Bhutan ACES 2 dataset.
        """
        self.config = config
        self.output_bucket = self.config.GCS_BUCKET
        self.kernel_size = self.config.PATCH_SHAPE_SINGLE
        self.grace = 10
        self.scale = self.config.SCALE
        self.test_ratio = kwargs.get("test_ratio", 0.1)
        self.validation_ratio = kwargs.get("validation_ratio", 0.2)
        self.seed = kwargs.get("seed", 100)
        self.use_service_account = self.config.USE_SERVICE_ACCOUNT
        self.split_dir = split_dir
        print(f"split_dir: {self.split_dir}")

        self.sample_locations = ee.FeatureCollection(kwargs.get("sampled_locations",
                                                                "projects/servir-sco-assets/assets/Bhutan/ACES_2/paro_2021_all_class_samples_clipped_new"))
        self.sample_locations = self.sample_locations.randomColumn("random", self.seed)

        default_label = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/paro_2021_all_class_label").rename("class").unmask(0, False)
        label = kwargs.get("label", default_label)
        if isinstance(label, str):
            self.label = ee.Image(label)
        elif isinstance(label, ee.Image):
            self.label = label
        else:
            raise ValueError("Invalid label provided. Please provide a valid Earth Engine Image object or path to an Image object.")

        image = kwargs.get("image")
        if isinstance(image, str):
            self.image = ee.Image(image)
        elif isinstance(image, ee.Image):
            self.image = image
        else:
            raise ValueError("Invalid image provided. Please provide a valid Earth Engine Image object or path to an Image object.")

    def load_data(self) -> None:
        """
        Load the necessary data from Earth Engine and prepare it for use.
        """
        EEUtils.initialize_session(use_highvolume=True, key=self.config.EE_SERVICE_CREDENTIALS if self.use_service_account else None)

        self.training_sample_locations = self.sample_locations.filter(ee.Filter.gt("random", self.validation_ratio + self.test_ratio)) # > 0.4
        print("Training sample size:", self.training_sample_locations.size().getInfo())
        self.validation_sample_locations = self.sample_locations.filter(ee.Filter.lte("random", self.validation_ratio)) # <= 0.2
        print("Validation sample size:", self.validation_sample_locations.size().getInfo())
        self.test_sample_locations = self.sample_locations.filter(ee.Filter.And(ee.Filter.gt("random", self.validation_ratio),
                                                               ee.Filter.lte("random", self.validation_ratio + self.test_ratio))) # > 0.2 and <= 0.4
        print("Test sample size:", self.test_sample_locations.size().getInfo())
        self.sample_size = self.sample_locations.size().getInfo()
        print("Sample size:", self.sample_size)
        self.sample_locations_list = self.sample_locations.toList(self.sample_size + self.grace)

        if self.config.USE_S1:
            self.config.FEATURES.extend(["vv_asc_before", "vh_asc_before", "vv_asc_during", "vh_asc_during",
                                         "vv_desc_before", "vh_desc_before", "vv_desc_during", "vh_desc_during"])

        if self.config.USE_ELEVATION:
            self.config.FEATURES.extend(["elevation", "slope"])

        self.image = self.image.select(self.config.FEATURES)
        self.image = self.image.addBands(label).toFloat()
        print("Image bands:", self.image.bandNames().getInfo())

        self.selectors = self.image.bandNames().getInfo()

        proj = ee.Projection("EPSG:4326").atScale(10).getInfo()

        # Get scales out of the transform.
        self.scale_x = proj["transform"][0]
        self.scale_y = -proj["transform"][4]

    def generate_training_neighborhood_data(self) -> None:
        """
        Use Apache Beam to generate training, validation, and test patch data from the loaded data.
        """
        from datetime import datetime

        export_kwargs = { "bucket": self.output_bucket, "selectors": self.selectors }

        extra_info = ""
        if self.config.USE_S1:
            extra_info += "_s1"

        if self.config.USE_ELEVATION:
            extra_info += "_elevation"

        training_file_prefix = f"{self.config.DATA_OUTPUT_DIR}/training_{int((1. - self.test_ratio - self.validation_ratio) * 100.)}/training{extra_info}__{self.kernel_size}x{self.kernel_size}"
        testing_file_prefix = f"{self.config.DATA_OUTPUT_DIR}/testing_{int(self.test_ratio * 100.)}/testing{extra_info}__{self.kernel_size}x{self.kernel_size}"
        validation_file_prefix = f"{self.config.DATA_OUTPUT_DIR}/validation_{int(self.validation_ratio * 100.)}/validation{extra_info}__{self.kernel_size}x{self.kernel_size}"

        def _generate_data(image, data, selectors, use_service_account, prefix):
            if "validation" in prefix:
                description = "Validation"
            elif "testing" in prefix:
                description = "Testing"
            elif "training" in prefix:
                description = "Training"
            else:
                description = "Unknown"

            print(f"{description} Data")

            beam_options = PipelineOptions([], direct_num_workers=0, direct_running_mode="multi_processing", runner="DirectRunner")
            with beam.Pipeline(options=beam_options) as pipeline:
                _ = (
                    pipeline
                    | "Create range" >> beam.Create(range(0, data.size().getInfo(), 1))
                    | "Yield sample points" >> beam.Map(EEUtils.beam_yield_sample_points_with_index, data.toList(data.size()), use_service_account)
                    | "Get patch" >> beam.Map(EEUtils.beam_sample_neighbourhood, image, self.config, use_service_account)
                    | "Write training data" >> beam.Map(EEUtils.beam_export_collection_to_cloud_storage, start_training=True,
                                                        **{**export_kwargs, "file_prefix": f"{prefix}_{datetime.now().strftime('%Y%m-%d%H-%M-%S_')}",
                                                           "description": f"{description}_{datetime.now().strftime('%Y%m-%d%H-%M-%S_')}",
                                                           "selectors": selectors})
                )

        if self.split_dir == "training":
            _generate_data(self.image, self.training_sample_locations, self.selectors, self.use_service_account, training_file_prefix)
        elif self.split_dir == "validation":
            _generate_data(self.image, self.validation_sample_locations, self.selectors, self.use_service_account, validation_file_prefix)
        elif self.split_dir == "testing":
            _generate_data(self.image, self.test_sample_locations, self.selectors, self.use_service_account, testing_file_prefix)
        else:
            print("Invalid split name specified. Choices are training, validation, and testing. Exiting..")
            exit(1)

    def generate_training_patch_data(self) -> None:
        """
        Use Apache Beam to generate training, validation, and test patch data from the loaded data.
        """
        beam_options = PipelineOptions([], direct_num_workers=0, direct_running_mode="multi_processing", runner="DirectRunner")
        with beam.Pipeline(options=beam_options) as pipeline:
            training_data, validation_data, test_data = (
                pipeline
                | "Create range" >> beam.Create(range(0, self.sample_size, 1))
                | "Yield sample points" >> beam.Map(EEUtils.beam_yield_sample_points, self.sample_locations_list, self.use_service_account)
                | "Get patch" >> beam.Map(EEUtils.beam_get_training_patches, self.image, self.selectors, self.scale, self.kernel_size, self.use_service_account)
                | "Filter patches" >> beam.Filter(Utils.filter_good_patches)
                | "Serialize" >> beam.Map(TFUtils.beam_serialize)
                | "Split dataset" >> beam.Partition(Utils.split_dataset, 3, validation_ratio=self.validation_ratio, test_ratio=self.test_ratio)
            )

            extra_info = ""
            if self.config.USE_S1:
                extra_info += "_s1"

            if self.config.USE_ELEVATION:
                extra_info += "_elevation"

            training_file_prefix = f"gs://{self.config.GCS_BUCKET}/{self.config.DATA_OUTPUT_DIR}/training_{int((1. - self.test_ratio - self.validation_ratio) * 100.)}/training{extra_info}__{self.kernel_size}x{self.kernel_size}"
            testing_file_prefix = f"gs://{self.config.GCS_BUCKET}/{self.config.DATA_OUTPUT_DIR}/testing_{int(self.test_ratio * 100.)}/testing{extra_info}__{self.kernel_size}x{self.kernel_size}"
            validation_file_prefix = f"gs://{self.config.GCS_BUCKET}/{self.config.DATA_OUTPUT_DIR}/validation_{int(self.validation_ratio * 100.)}/validation{extra_info}__{self.kernel_size}x{self.kernel_size}"


            # Write the datasets to TFRecord files in the output bucket
            training_data | "Write training data" >> beam.io.WriteToTFRecord(training_file_prefix, file_name_suffix=".tfrecord.gz")
            validation_data | "Write validation data" >> beam.io.WriteToTFRecord(testing_file_prefix, file_name_suffix=".tfrecord.gz")
            test_data | "Write test data" >> beam.io.WriteToTFRecord(validation_file_prefix, file_name_suffix=".tfrecord.gz")

    def generate_training_patch_seed_data(self) -> None:
        """
        Use Apache Beam to generate training, validation, and test patch data from the loaded data.
        """
        def _generate_data_seed(image, data, selectors, scale, kernel_size, use_service_account, output_path) -> None:
            beam_options = PipelineOptions([], direct_num_workers=0, direct_running_mode="multi_processing", runner="DirectRunner")
            with beam.Pipeline(options=beam_options) as pipeline:
                _ = (
                    pipeline
                    | "Create range" >> beam.Create(range(0, data.size().getInfo(), 1))
                    | "Yield sample points" >> beam.Map(EEUtils.beam_yield_sample_points, data.toList(data.size()), use_service_account)
                    | "Get patch" >> beam.Map(EEUtils.beam_get_training_patches, image, selectors, scale, kernel_size, use_service_account)
                    | "Filter patches" >> beam.Filter(Utils.filter_good_patches)
                    | "Serialize" >> beam.Map(TFUtils.beam_serialize)
                    | "Write training data" >> beam.io.WriteToTFRecord(output_path, file_name_suffix=".tfrecord.gz")
                )

        extra_info = ""
        if self.config.USE_S1:
            extra_info += "_s1"

        if self.config.USE_ELEVATION:
            extra_info += "_elevation"

        training_file_prefix = f"gs://{self.config.GCS_BUCKET}/{self.config.DATA_OUTPUT_DIR}/training_{int((1. - self.test_ratio - self.validation_ratio) * 100.)}/training{extra_info}__{self.kernel_size}x{self.kernel_size}"
        testing_file_prefix = f"gs://{self.config.GCS_BUCKET}/{self.config.DATA_OUTPUT_DIR}/testing_{int(self.test_ratio * 100.)}/testing{extra_info}__{self.kernel_size}x{self.kernel_size}"
        validation_file_prefix = f"gs://{self.config.GCS_BUCKET}/{self.config.DATA_OUTPUT_DIR}/validation_{int(self.validation_ratio * 100.)}/validation{extra_info}__{self.kernel_size}x{self.kernel_size}"


        datasets = [
            {"name": "training", "locations": self.training_sample_locations, "output_path": training_file_prefix},
            {"name": "testing", "locations": self.test_sample_locations, "output_path": testing_file_prefix},
            {"name": "validation", "locations": self.validation_sample_locations, "output_path": validation_file_prefix}
        ]

        for dataset in datasets:
            if dataset["name"] == self.split_dir:
                print(f"{dataset['name'].capitalize()} output path:", dataset["output_path"])
                _generate_data_seed(
                    self.image, dataset["locations"], self.selectors, self.scale, self.kernel_size,
                    self.use_service_account, dataset["output_path"]
                )

    def generate_training_point_data(self) -> None:
        """
        Use Apache Beam to generate training, validation, and test point data from the loaded data.
        """
        training_sample_points = EEUtils.sample_image(self.image, self.training_sample_locations, **self.config.__dict__)
        validation_sample_points = EEUtils.sample_image(self.image, self.validation_sample_locations, **self.config.__dict__)
        test_sample_points = EEUtils.sample_image(self.image, self.test_sample_locations, **self.config.__dict__)

        extra_info = ""
        if self.config.USE_S1:
            extra_info += "_s1"

        if self.config.USE_ELEVATION:
            extra_info += "_elevation"

        training_file_prefix = f"{self.config.DATA_OUTPUT_DIR}/training_{int((1. - self.test_ratio - self.validation_ratio) * 100.)}/training{extra_info}"
        testing_file_prefix = f"{self.config.DATA_OUTPUT_DIR}/testing_{int(self.test_ratio * 100.)}/testing{extra_info}"
        validation_file_prefix = f"{self.config.DATA_OUTPUT_DIR}/validation_{int(self.validation_ratio * 100.)}/validation{extra_info}"

        datasets = [
            {"name": "training", "sample_points": training_sample_points, "output_path": training_file_prefix},
            {"name": "testing", "sample_points": test_sample_points, "output_path": testing_file_prefix},
            {"name": "validation", "sample_points": validation_sample_points, "output_path": validation_file_prefix}
        ]

        export_kwargs = { "bucket": self.output_bucket, "selectors": self.selectors }

        for dataset in datasets:
            print(f"{dataset['name'].capitalize()} output path:", dataset["output_path"])
            EEUtils.export_collection_data(dataset["sample_points"], export_type="cloud", start_training=True, **{**export_kwargs, "file_prefix": dataset["output_path"], "description": dataset["name"].capitalize()})

    def run_patch_generator(self) -> None:
        """
        Run the patch training data generation process.
        """
        self.load_data()
        self.generate_training_patch_data()

    def run_patch_generator_seed(self) -> None:
        """
        Run the patch training data generation process.
        """
        self.load_data()
        self.generate_training_patch_seed_data()

    def run_point_generator(self) -> None:
        """
        Run the point training data generation process.
        """
        self.load_data()
        self.generate_training_point_data()

    def run_neighborhood_generator(self) -> None:
        """
        Run the neighborhood training data generation process.
        """
        self.load_data()
        self.generate_training_neighborhood_data()


if __name__ == "__main__":
    print("Program started..")

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-m", "--mode", help="""Which mode to run? The available modes are: patch, point, neighborhood (or neighbourhood), patch_seed. \n
                        use as python generate_training_data.py --mode patch \n
                        or python generate_training_data.py --m point""")

    parser.add_argument("--config", help="`.env` file to load config from")

    parser.add_argument("--split_directory", help="The split directory. Default is training. Choices are training, testing, and validation \n")

    parser.add_argument("--sample_data", help="The sampled dataset to load from. Please provide the path only to the ee.FeatureCollection to load from \n")

    parser.add_argument("--label_data", help="The label dataset to load from. Please either provide the path or directly the ee.Image Object to load from \n")

    parser.add_argument("--image_data", help="The image dataset to load from. Please either provide the path or directly the ee.Image Object to load from \n")

    # Read arguments from command line
    mode = "neighborhood"
    if parser.parse_args().mode:
        mode = parser.parse_args().mode
    else:
        print("No mode specified, defaulting to neighborhood.")

    split_dir = "training"
    if parser.parse_args().split_directory:
        split_dir = parser.parse_args().split_directory
    else:
        print("No split directory specified, defaulting to `training`.")

    # sample locations
    sample_locations = "projects/servir-sco-assets/assets/Bhutan/ACES_2/paro_2021_all_class_samples_clipped_new"
    if parser.parse_args().sample_data:
        sample_locations = parser.parse_args().sample_data
    else:
        print("No dataset specified, defaulting to what we have here.")

    EEUtils.initialize_session(use_highvolume=True)
    # label dataset
    if parser.parse_args().label_data:
        label = parser.parse_args().label_data
    else:
        print("No label dataset specified, defaulting to what we have here.")
        label = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/paro_2021_all_class_label").rename("class").unmask(0, False)
        # other = label.remap([0, 1], [1, 0]).rename(["other"])

    # config file
    config = "../config.env"
    if parser.parse_args().config:
        config = parser.parse_args().config
    else:
        print("No config file specified, defaulting to `config.env`.")

    config = Config(config)


    # image
    composite_after = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Paro_Rice_Composite_2021/composite_after")
    composite_before = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Paro_Rice_Composite_2021/composite_before")
    composite_during = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Paro_Rice_Composite_2021/composite_during")

    evi_before = EEUtils.calculate_evi(composite_before)
    evi_during = EEUtils.calculate_evi(composite_during)
    evi_after = EEUtils.calculate_evi(composite_after)

    composite_before = composite_before.addBands(evi_before)
    composite_during = composite_during.addBands(evi_during)
    composite_after = composite_after.addBands(evi_after)

    original_bands = composite_before.bandNames().getInfo()
    lowercase_bands = [band.lower() for band in original_bands]

    composite_before = composite_before.select(original_bands, lowercase_bands)
    composite_after = composite_after.select(original_bands, lowercase_bands)
    composite_during = composite_during.select(original_bands, lowercase_bands)

    composite_before = composite_before.regexpRename("$(.*)", "_before")
    composite_after = composite_after.regexpRename("$(.*)", "_after")
    composite_during = composite_during.regexpRename("$(.*)", "_during")

    sentinel1_asc_before_composite = ee.Image("projects/servir-sco-assets/assets/Bhutan/Sentinel1Ascending2021/s1AscBefore")
    sentinel1_asc_during_composite = ee.Image("projects/servir-sco-assets/assets/Bhutan/Sentinel1Ascending2021/s1AscDuring")

    sentinel1_desc_before_composite = ee.Image("projects/servir-sco-assets/assets/Bhutan/Sentinel1Descending2021/s1DescBefore")
    sentinel1_desc_during_composite = ee.Image("projects/servir-sco-assets/assets/Bhutan/Sentinel1Descending2021/s1DescDuring")

    elevation = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/elevationParo")
    slope = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/slopeParo")

    include_after = False
    if include_after:
        image = composite_before.addBands(composite_during).addBands(composite_after).toFloat()
    else:
        image = composite_before.addBands(composite_during).toFloat()

    if config.USE_S1:
        image = image.addBands(sentinel1_asc_before_composite).addBands(sentinel1_asc_during_composite)\
                     .addBands(sentinel1_desc_before_composite).addBands(sentinel1_desc_during_composite).toFloat()

    if config.USE_ELEVATION:
        image = image.addBands(elevation).addBands(slope).toFloat()

    # image dataset
    if parser.parse_args().image_data:
        image = parser.parse_args().image_data
    else:
        print("No image dataset specified, defaulting to what we have here.")
        image = image

    # more settings can be applied here
    generator = TrainingDataGenerator(config=config, split_dir=split_dir, sample_locations=sample_locations, image=image, label=label)
    if mode == "patch":
        generator.run_patch_generator()
    elif mode == "patch_seed":
        generator.run_patch_generator_seed()
    elif mode == "point":
        generator.run_point_generator()
    elif mode == "neighborhood" or mode == "neighbourhood":
        generator.run_neighborhood_generator()
    else:
        print("Invalid mode specified.")

    print("\nProgram completed.")
