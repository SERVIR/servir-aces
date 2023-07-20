# -*- coding: utf-8 -*-

try:
    from aces.ee_utils import EEUtils
    from aces.config import Config
except ModuleNotFoundError:
    print("ModuleNotFoundError: Attempting to import from parent directory.")
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from aces.ee_utils import EEUtils
    from aces.config import Config

import ee

EEUtils.initialize_session(use_highvolume=True)

country = "Bhutan"
region = "Punakha"
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

region_fc = name_region[region]

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
print("image", image.bandNames().getInfo())


# Specify patch and file dimensions.
formatOptions = {
  "patchDimensions": [128, 128],
  "maxFileSize": 104857600,
  "compressed": True
}

if Config.BUFFER_SIZE:
    formatOptions["kernelSize"] = Config.BUFFER_SIZE

# Setup the task
image_export_options = {
    "description": Config.GCS_IMAGE_DIR.split("/")[-1],
    "file_name_prefix": f"{Config.GCS_IMAGE_DIR}/{Config.GCS_IMAGE_PREFIX}",
    "bucket": Config.GCS_BUCKET,
    "scale": 5,
    "file_format": "TFRecord",
    "region": region_fc,
    "format_options": formatOptions,
}

EEUtils.export_image(image, export_type=["cloud"], start_training=True, **image_export_options)
