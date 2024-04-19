# -*- coding: utf-8 -*-

import re, warnings
import ee
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List
from aces import Config
from aces.utils import Utils
import tensorflow as tf


__all__ = ["EEUtils"]


class EEUtils:
    """
    EEUtils: Earth Engine Utility Class

    This class provides utility functions to handle Earth Engine API information and make authenticated requests.
    """
    @staticmethod
    def get_credentials_by_service_account_key(key):
        """
        Helper function to retrieve credentials using a service account key.

        Parameters:
        key (str): The path to the service account key JSON file.

        Returns:
        ee.ServiceAccountCredentials: The authenticated credentials.
        """
        import json
        service_account = json.load(open(key))
        credentials = ee.ServiceAccountCredentials(service_account["client_email"], key)
        return credentials

    @staticmethod
    def initialize_session(use_highvolume : bool = False, key : Union[str, None] = None, project: str = None):
        """
        Initialize the Earth Engine session.
        If use_highvolume is True, the high-volume Earth Engine API will be used.
        If a project is provided, the session will be initialized with the project ID. Recommended to use project.
        If a key is provided, the service account key will be used.

        Parameters:
        use_highvolume (bool): Whether to use the high-volume Earth Engine API.
        key (str or None): The path to the service account key JSON file. If None, the default credentials will be used.
        project (str): The Google Cloud project ID to use for the session.
        """
        if key is None:
            if use_highvolume and project:
                ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project=project)
            elif use_highvolume:
                ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
            elif project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
        else:
            credentials = EEUtils.get_credentials_by_service_account_key(key)
            if use_highvolume and project:
                ee.Initialize(credentials, opt_url="https://earthengine-highvolume.googleapis.com", project=project)
            elif use_highvolume:
                ee.Initialize(credentials, opt_url="https://earthengine-highvolume.googleapis.com")
            elif project:
                ee.Initialize(credentials, project=project)
            else:
                ee.Initialize(credentials)

    @staticmethod
    def calculate_avg_min_max_statistics(image: ee.Image, geometry: ee.FeatureCollection, scale: int = 30) -> ee.Dictionary:
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
    def export_collection_data(collection: ee.FeatureCollection, export_type: Union[list, str]="cloud", start_training=True, **params) -> None:
        if isinstance(export_type, str):
            export_type = [export_type]

        for _type in export_type:
            if _type == "cloud":
                EEUtils._export_collection_to_cloud_storage(collection, start_training, **params)

            if _type == "asset":
                EEUtils._export_collection_to_asset(collection, start_training, **params)

            if _type == "drive":
                EEUtils._export_collection_to_drive(collection, start_training, **params)

            if _type not in ["cloud", "asset", "drive"]:
                raise NotImplementedError(f"Currently supported export types are: {', '.join(_type)}")

    @staticmethod
    def _export_collection_to_drive(collection, start_training, **kwargs) -> None:
        print("Exporting training data to Google Drive..")
        training_task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=kwargs.get("description", "myExportTableTask"),
            folder=kwargs.get("folder", "myFolder"),
            fileNamePrefix=kwargs.get("file_prefix", "myExportTableTask"),
            fileFormat=kwargs.get("file_format", "CSV"),
            selectors=kwargs.get("selectors", collection.first().propertyNames().getInfo()),
        )
        if start_training: training_task.start()

    @staticmethod
    def _export_collection_to_asset(collection, start_training, **kwargs) -> None:
        asset_id = kwargs.get("asset_id", "myAssetId")
        print(f"Exporting training data to {asset_id}..")
        training_task = ee.batch.Export.table.toAsset(
            collection=collection,
            description=kwargs.get("description", "myExportTableTask"),
            assetId=asset_id,
            selectors=kwargs.get("selectors", collection.first().propertyNames().getInfo()),
        )
        if start_training: training_task.start()

    @staticmethod
    def _export_collection_to_cloud_storage(collection, start_training, **kwargs) -> None:
        description = kwargs.get("description", "myExportTableTask")
        bucket = kwargs.get("bucket", "myBucket")
        file_prefix = kwargs.get("file_prefix") if kwargs.get("file_prefix") is not None else description
        print(f"Exporting training data to gs://{bucket}/{file_prefix}..")
        training_task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=description,
            fileNamePrefix=file_prefix,
            bucket=bucket,
            fileFormat=kwargs.get("file_format", "TFRecord"),
            selectors=kwargs.get("selectors", collection.first().propertyNames().getInfo()),
        )
        if start_training: training_task.start()

    @staticmethod
    def beam_export_collection_to_cloud_storage(collection_index, start_training, **kwargs) -> None:
        from aces.ee_utils import EEUtils
        import ee
        EEUtils.initialize_session(use_highvolume=True)

        collection = ee.FeatureCollection(collection_index[0])
        index = collection_index[1]

        description = kwargs.get("description", "myExportTableTask")
        bucket = kwargs.get("bucket", "myBucket")
        file_prefix = kwargs.get("file_prefix") if kwargs.get("file_prefix") is not None else description
        file_prefix = f"{file_prefix}_{index}"
        print(f"Exporting training data to gs://{bucket}/{file_prefix}..")
        training_task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=f"{description}__index_{index}",
            fileNamePrefix=file_prefix,
            bucket=bucket,
            fileFormat=kwargs.get("file_format", "TFRecord"),
            selectors=kwargs.get("selectors", collection.first().propertyNames().getInfo()),
        )
        if start_training: training_task.start()

    @staticmethod
    def export_image(image: ee.Image, export_type: str="asset", start_training=True, **params) -> None:
        if isinstance(export_type, str):
            export_type = [export_type]

        for _type in export_type:
            if _type == "cloud":
                EEUtils._export_image_to_cloud_storage(image, start_training, **params)

            if _type == "asset":
                EEUtils._export_image_to_asset(image, start_training, **params)

            if _type not in ["cloud", "asset"]:
                raise NotImplementedError(f"Currently supported export types are: {', '.join(_type)}")

    @staticmethod
    def _export_image_to_cloud_storage(image, start_training, **kwargs) -> None:
        description = kwargs.get("description", "myExportImageTask")
        bucket = kwargs.get("bucket", "myBucket")
        file_name_prefix = kwargs.get("file_name_prefix") if kwargs.get("file_name_prefix") is not None else description
        region = kwargs.get("region", None)
        if region is not None:
            if isinstance(region, ee.FeatureCollection):
                region = region.geometry()
            elif isinstance(region, ee.Geometry):
                region = region
            else:
                raise ValueError(f"region must be an ee.FeatureCollection or ee.Geometry object. Found {type(region)}")

        print(f"Exporting training data to gs://{bucket}/{file_name_prefix}..")

        params = {
            "image": image,
            "description": description,
            "fileNamePrefix": file_name_prefix,
            "bucket": kwargs.get("bucket", "myBucket"),
            "fileFormat": kwargs.get("file_format", "GeoTIFF"),
            "formatOptions": kwargs.get("format_options", None),
            "region": region,
            "scale": kwargs.get("scale", 1000),
            "maxPixels": kwargs.get("max_pixels", 1e10),
        }
        keys = Utils.convert_camel_to_snake(list(params.keys()))
        keys.remove("image")

        for key in list(kwargs.keys()):
            if key not in keys:
                warnings.warn(f"Parameter {key} not found in kwargs. Double check your parameter name (camelCase vs snake_case)")

        not_none_params = {k:v for k, v in params.items() if v is not None}
        training_task = ee.batch.Export.image.toCloudStorage(**not_none_params)
        if start_training: training_task.start()

    @staticmethod
    def _export_image_to_asset(image, start_training, **kwargs) -> None:
        asset_id = kwargs.get("asset_id", "")
        print(f"Exporting image to {asset_id}..")

        training_task = ee.batch.Export.image.toAsset(
            image=image,
            description=kwargs.get("description", "myExportImageTask"),
            assetId=asset_id,
            region=kwargs.get("region", None),
            scale=kwargs.get("scale", 30),
            maxPixels=kwargs.get("max_pixels", 1E13),
        )
        if start_training: training_task.start()

    @staticmethod
    def country_bbox(country_name, max_error=100):
        """Function to get a bounding box geometry of a country

        args:
            country_name (str): US-recognized country name
            max_error (float,optional): The maximum amount of error tolerated when
                performing any necessary reprojection. default = 100

        returns:
            ee.Geometry: geometry of country bounding box
        """

        all_countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        return all_countries.filter(ee.Filter.eq("country_na", country_name))\
                            .geometry(max_error).bounds(max_error)

    @staticmethod
    def get_image_collection_statistics(image_collection: ee.ImageCollection) -> ee.Image:
        reducers = ee.Reducer.mean() \
            .combine(reducer2=ee.Reducer.min(), sharedInputs=True) \
                .combine(reducer2=ee.Reducer.max(), sharedInputs=True) \
                    .combine(reducer2=ee.Reducer.stdDev(), sharedInputs=True) \
                        .combine(reducer2=ee.Reducer.percentile([25, 50, 75], ["Q1", "Q2", "Q3"]), sharedInputs=True)
        reducer = image_collection.reduce(reducer=reducers)
        return reducer.float()

    @staticmethod
    def calculate_planet_indices(image: ee.Image) -> ee.Image:
        ndvi = image.normalizedDifference(["N", "R"]).rename("NDVI")
        evi = image.expression (
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
                "NIR": image.select("N"),
                "RED": image.select("R"),
                "BLUE": image.select("B")
            }).rename("EVI")
        ndwi = image.normalizedDifference(["G", "N"]).rename("NDWI")
        savi = image.expression("((NIR - RED) / (NIR + RED + 0.5))*(1.5)", {
            "NIR": image.select("N"),
            "RED": image.select("R")
        }).rename("SAVI")
        msavi2 = image.expression("(( (2*NIR + 1) - sqrt( ((2*NIR + 1) * (2*NIR + 1)) - 8 * (NIR - R) ) )) / 2", {
            "NIR": image.select("N"),
            "R": image.select("R")
        }).rename("MSAVI2")

        mtvi2 = image.expression("( 1.5*(1.2*(NIR - GREEN) - 2.5*(RED - GREEN)) ) / ( sqrt( ((2*NIR + 1) * (2*NIR + 1)) - (6*NIR - 5*sqrt(RED)) - 0.5 ) )", {
            "NIR": image.select("N"),
            "RED": image.select("R"),
            "GREEN": image.select("G"),
        }).rename("MTVI2")

        vari = image.expression("(GREEN - RED) / (GREEN + RED - BLUE)", {
            "GREEN": image.select("G"),
            "RED": image.select("R"),
            "BLUE": image.select("B"),
        }).rename("VARI")

        tgi = image.expression("( (120*(RED - BLUE)) - (190*(RED - GREEN)) ) / 2", {
            "GREEN": image.select("G"),
            "RED": image.select("R"),
            "BLUE": image.select("B"),
        }).rename("TGI")

        return ndvi.addBands([evi, ndwi, savi, msavi2, mtvi2, vari, tgi]).float()

    @staticmethod
    def calculate_evi(image: ee.Image) -> ee.Image:
        evi = image.expression (
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
                "NIR": image.select("nir"),
                "RED": image.select("red"),
                "BLUE": image.select("blue")
            }).rename("EVI")
        return evi.float()

    @staticmethod
    def calculate_s1_indices(image: ee.Image) -> ee.Image:
        VV = image.select("VV").rename("vv")
        VH = image.select("VH").rename("vh")
        ratio = VV.divide(VH).rename("s1_ratio")
        ndratio = VV.subtract(VH).divide(VV.add(VH)).rename("s1_ndratio")
        return VV.addBands([VH, ratio, ndratio]).float()

    @staticmethod
    def generate_stratified_samples(image: ee.Image, region: ee.Geometry, numPoints: int = 500, classBand: str = None, scale: int=30, **kwargs) -> ee.FeatureCollection:
        # Add a latitude and longitude band.
        return image.addBands(ee.Image.pixelLonLat()).stratifiedSample(
            numPoints=numPoints,
            classBand=classBand if classBand else "label",
            scale=scale,
            region=region,
            seed=kwargs.get("seed", Config.SEED),
            classValues=kwargs.get("class_values", None),
            classPoints=kwargs.get("class_points", None),# [2000, 600, 600, 600]
        ).map(lambda f: f.setGeometry(ee.Geometry.Point([f.get("longitude"), f.get("latitude")])))

    @staticmethod
    def sample_image_by_collection(image: ee.Image, collection: ee.FeatureCollection, **kwargs: dict) -> ee.FeatureCollection:
        samples = image.sampleRegions(
            collection=collection,
            properties=kwargs.get("properties", collection.first().propertyNames().getInfo()),
            scale=kwargs.get("scale", None),
            geometries=kwargs.get("geometries", False),
            tileScale=kwargs.get("tile_scale", 1),
        )
        return samples

    @staticmethod
    def sample_image(image: ee.Image, region: ee.FeatureCollection, **kwargs: dict) -> ee.FeatureCollection:
        sample = image.sample(region=region,
                              scale=kwargs.get("SCALE") or kwargs.get("scale"),
                              seed=kwargs.get("SEED") or kwargs.get("seed"),
                              geometries=kwargs.get("geometries", False))
        return sample

    @staticmethod
    def beam_yield_sample_points_with_index(index, sample_locations: ee.List, use_service_account: bool = False) -> List:
        from aces.ee_utils import EEUtils
        from aces.config import Config
        import ee
        EEUtils.initialize_session(use_highvolume=True, key=Config.EE_SERVICE_CREDENTIALS if use_service_account else None)
        print(f"Yielding Index: {index} of {sample_locations.size().getInfo() - 1}")
        point = ee.Feature(sample_locations.get(index)).geometry().getInfo()
        return point["coordinates"], index

    @staticmethod
    def beam_yield_sample_points(index, sample_locations: ee.List, use_service_account: bool = False) -> List:
        from aces.ee_utils import EEUtils
        from aces.config import Config
        import ee
        EEUtils.initialize_session(use_highvolume=True, key=Config.EE_SERVICE_CREDENTIALS if use_service_account else None)
        print(f"Yielding Index: {index} of {sample_locations.size().getInfo() - 1}")
        point = ee.Feature(sample_locations.get(index)).geometry().getInfo()
        return point["coordinates"]

    @staticmethod
    def beam_sample_neighbourhood(coords_index, image, config: Union[Config, str] = "config.env", use_service_account: bool = False):
        from aces.ee_utils import EEUtils
        from aces.config import Config
        import ee

        if isinstance(config, str):
            config = Config(config)
        elif isinstance(config, Config):
            config = config
        else:
            raise ValueError("config must be of type Config or str")

        EEUtils.initialize_session(use_highvolume=True, key=config.EE_SERVICE_CREDENTIALS if use_service_account else None)

        coords = coords_index[0]
        index = coords_index[1]

        def get_kernel(kernel_size) -> ee.Kernel:
            eelist = ee.List.repeat(1, kernel_size)
            lists = ee.List.repeat(eelist, kernel_size)
            kernel = ee.Kernel.fixed(kernel_size, kernel_size, lists)
            return kernel

        def create_neighborhood(kernel) -> ee.Image:
            return image.neighborhoodToArray(kernel)

        def sample_data(image, points) -> ee.FeatureCollection:
            return image.sample(
                region=points,
                scale=config.SCALE,
                tileScale=16,
                geometries=False
            )

        image_kernel = get_kernel(config.PATCH_SHAPE_SINGLE)
        neighborhood = create_neighborhood(image_kernel)
        training_data = sample_data(neighborhood, ee.Geometry.Point(coords))
        return training_data, index


    @staticmethod
    def beam_get_training_patches(coords: List[float], image: ee.Image, bands: List[str] = [],
                                  scale: int = 5, patch_size: int = 128, use_service_account: bool = False) -> np.ndarray:
        """Get a training patch centered on the coordinates."""
        from aces.ee_utils import EEUtils
        from aces.config import Config
        import ee
        EEUtils.initialize_session(use_highvolume=True, key=Config.EE_SERVICE_CREDENTIALS if use_service_account else None)
        from google.api_core import exceptions, retry
        import requests
        import numpy as np
        from typing import List
        import io

        # @retry.Retry(timeout=10*60) # seconds
        @retry.Retry(deadline=10*60) # seconds
        def get_patch(image: ee.Image, region: ee.Geometry, bands: List[str], patch_size: int) -> np.ndarray:
            """Get the patch of pixels in the geometry as a Numpy array."""
            # Create the URL to download the band values of the patch of pixels.
            url = image.getDownloadURL({
                "region": region,
                "dimensions": [patch_size, patch_size],
                "format": "NPY",
                "bands": bands,
            })
            # Download the pixel data. If we get "429: Too Many Requests" errors,
            # it"s safe to retry the request.
            response = requests.get(url)
            if response.status_code == 429:
                # The retry.Retry library only works with `google.api_core` exceptions.
                raise exceptions.TooManyRequests(response.text)

            if response.status_code == 503:
                raise exceptions.ServiceUnavailable(response.text)

                # Still raise any other exceptions to make sure we got valid data.
            response.raise_for_status()
            # Load the NumPy file data and return it as a NumPy array.
            return np.load(io.BytesIO(response.content), allow_pickle=True)

        # @retry.Retry(timeout=10*60) # seconds
        @retry.Retry(deadline=10*60) # seconds
        def compute_pixel(image: ee.Image, region: ee.Geometry, bands: List[str], patch_size: int, scale_x: float, scale_y: float) -> np.ndarray:
            """Get the patch of pixels in the geometry as a Numpy array."""

            # Make a request object.
            request = {
                "expression": image,
                "fileFormat": "NPY",
                "bandIds": bands,
                "grid": {
                    "dimensions": {
                        "width": patch_size,
                        "height": patch_size
                    },
                    "affineTransform": {
                        "scaleX": scale_x,
                        "shearX": 0,
                        "translateX": coords[0],
                        "shearY": 0,
                        "scaleY": scale_y,
                        "translateY": coords[1]
                    },
                    "crsCode": "EPSG:4326",
                },
            }
            response = ee.data.computePixels(request)
            # Load the NumPy file data and return it as a NumPy array.
            return np.load(io.BytesIO(response.content), allow_pickle=True)

        point = ee.Geometry.Point(coords)
        region = point.buffer(scale * patch_size / 2, 1).bounds(1)
        return get_patch(image, region, bands, patch_size)
