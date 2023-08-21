# -*- coding: utf-8 -*-

try:
    from aces.ee_utils import EEUtils
    from aces.config import Config
except ModuleNotFoundError:
    print("ModuleNotFoundError: Attempting to import from parent directory.")
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from aces.ee_utils import EEUtils
    from aces.config import Config

import ee

EEUtils.initialize_session(use_highvolume=True)

country = "Bhutan"
region = "Paro"
year = 2021
sensor = "planet"

fc_country = ee.FeatureCollection(EEUtils.country_bbox(country))
admin_1 = ee.FeatureCollection("projects/servir-sco-assets/assets/Bhutan/BT_Admin_1")
paro = admin_1.filter(ee.Filter.eq("ADM1_EN", "Paro"))
punakha = admin_1.filter(ee.Filter.eq("ADM1_EN", "Punakha"))
punakha = punakha.map(lambda f: f.set("reduceId", 1))

monggar = admin_1.filter(ee.Filter.eq("ADM1_EN", "Monggar"))
monggar = monggar.map(lambda f: f.set("reduceId", 1))

name_region = {
    "Punakha": punakha,
    "Paro": paro,
    "Monggar": monggar
}

rice_zone = {
    "Punakha": {
        "min": 1000,
        "max": 2600,
    },
    "Paro ": {
        "min": 1500,
        "max": 2600,
    },
}

region_fc = name_region[region]

# smaller example region
# region_fc = ee.Geometry.Polygon(
#         [[[89.29777557572515, 27.54982094226756],
#           [89.29777557572515, 27.187608451375436],
#           [89.54908783158453, 27.187608451375436],
#           [89.54908783158453, 27.54982094226756]]], None, False);

composite_before_paro = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Paro_Rice_Composite_2021/composite_before")
composite_during_paro = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Paro_Rice_Composite_2021/composite_during")
composite_before_punakha = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Punakha_Composite_2021/composite_before")
composite_during_punakha = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/Punakha_Composite_2021/composite_during")

name_image = {
    "Punakha": [composite_before_punakha, composite_during_punakha],
    "Paro": [composite_before_paro, composite_during_paro],
}

composite_before = name_image[region][0]
composite_during = name_image[region][1]

if sensor == "planet":
    bands = ["red", "green", "blue", "nir"]

composite_before = composite_before.select(bands)
composite_during = composite_during.select(bands)

# var srtm = ee.Image("USGS/SRTMGL1_003")
# var slope = ee.Algorithms.Terrain(srtm).select("slope")

# var elevation_stats = statisticsForImage (srtm, paro, 90)

# srtm = srtm.unitScale(elevation_stats.get("elevation_min"), elevation_stats.get("elevation_max"))
# slope = slope.unitScale(0, 90)

composite_before = composite_before.regexpRename("$(.*)", "_before")
composite_during = composite_during.regexpRename("$(.*)", "_during")
image = composite_before.addBands(composite_during).toFloat()


dem = ee.Image("MERIT/DEM/v1_0_3") # ee.Image('USGS/SRTMGL1_003');
dem = dem.clip(fc_country)
riceZone = dem.gt(1000).And(dem.lte(2600))

image = image.clip(region_fc).updateMask(riceZone)

print("image", image.bandNames().getInfo())


# Specify patch and file dimensions.
formatOptions = {
  "patchDimensions": [Config.PATCH_SHAPE_SINGLE, Config.PATCH_SHAPE_SINGLE],
  "maxFileSize": 104857600,
  "compressed": True
}

if Config.KERNEL_BUFFER:
    formatOptions["kernelSize"] = Config.KERNEL_BUFFER

# Setup the task
image_export_options = {
    "description": Config.GCS_IMAGE_DIR.split("/")[-1],
    "file_name_prefix": f"{Config.GCS_IMAGE_DIR}/{Config.GCS_IMAGE_PREFIX}",
    "bucket": Config.GCS_BUCKET,
    "scale": Config.SCALE,
    "file_format": "TFRecord",
    # "region": image.geometry(),
    "format_options": formatOptions,
    "max_pixels": 1e13,
}

EEUtils.export_image(image, export_type=["cloud"], start_training=True, **image_export_options)
