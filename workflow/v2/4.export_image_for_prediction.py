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
    "Paro": {
        "min": 1500,
        "max": 2600,
    },
}

region_fc = name_region[region]

# smaller example region
region_fc = ee.Geometry.Polygon(
        [[[89.25910506388209, 27.58540960195346],
          [89.25910506388209, 27.159794800895543],
          [89.58182845255396, 27.159794800895543],
          [89.58182845255396, 27.58540960195346]]], None, False);

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

composite_before = composite_before.regexpRename("$(.*)", "_before")
composite_during = composite_during.regexpRename("$(.*)", "_during")
image = composite_before.addBands(composite_during).toFloat()

config_file = "config.env"
config = Config(config_file)

if config.USE_ELEVATION:
    elevation = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/elevationParo")
    slope = ee.Image("projects/servir-sco-assets/assets/Bhutan/ACES_2/slopeParo")
    image = image.addBands(elevation).addBands(slope).toFloat()
    config.FEATURES.extend(["elevation", "slope"])


if config.USE_S1:
    sentinel1_asc_before_composite = ee.Image("projects/servir-sco-assets/assets/Bhutan/Sentinel1Ascending2021/s1AscBefore")
    sentinel1_asc_during_composite = ee.Image("projects/servir-sco-assets/assets/Bhutan/Sentinel1Ascending2021/s1AscDuring")
    sentinel1_desc_before_composite = ee.Image("projects/servir-sco-assets/assets/Bhutan/Sentinel1Descending2021/s1DescBefore")
    sentinel1_desc_during_composite = ee.Image("projects/servir-sco-assets/assets/Bhutan/Sentinel1Descending2021/s1DescDuring")

    image = image.addBands(sentinel1_asc_before_composite).addBands(sentinel1_asc_during_composite).addBands(sentinel1_desc_before_composite).addBands(sentinel1_desc_during_composite).toFloat()
    config.FEATURES.extend(["vv_asc_before", "vh_asc_before", "vv_asc_during", "vh_asc_during",
                            "vv_desc_before", "vh_desc_before", "vv_desc_during", "vh_desc_during"])

# dem = ee.Image("MERIT/DEM/v1_0_3") # ee.Image('USGS/SRTMGL1_003');
# dem = dem.clip(fc_country)
# riceZone = dem.gt(rice_zone[region]["min"]).And(dem.lte(rice_zone[region]["max"]))
# image = image.clip(region_fc).updateMask(riceZone)

image = image.select(config.FEATURES)
print("image", image.bandNames().getInfo())

# Specify patch and file dimensions.
formatOptions = {
  "patchDimensions": [config.PATCH_SHAPE_SINGLE, config.PATCH_SHAPE_SINGLE],
  "maxFileSize": 104857600,
  "compressed": True
}

if config.KERNEL_BUFFER:
    formatOptions["kernelSize"] = config.KERNEL_BUFFER

# Setup the task
image_export_options = {
    "description": "export_task_for_prediction",
    "file_name_prefix": f"{config.GCS_IMAGE_DIR}/{config.GCS_IMAGE_PREFIX}",
    "bucket": config.GCS_BUCKET,
    "scale": config.SCALE,
    "file_format": "TFRecord",
    "region": region_fc, # image.geometry(),
    "format_options": formatOptions,
    "max_pixels": 1e13,
}

print("image_export_options", image_export_options)

EEUtils.export_image(image, export_type=["cloud"], start_training=True, **image_export_options)
