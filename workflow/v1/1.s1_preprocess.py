# -*- coding: utf-8 -*-

try:
    from aces.ee_utils import EEUtils
except ModuleNotFoundError:
    print("ModuleNotFoundError: Attempting to import from parent directory.")
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from aces.ee_utils import EEUtils

try:
    import hydrafloods as hf
except ImportError:
    print("hydrafloods not installed, please install it from pip as `pip install hydrafloods`")

import ee


EEUtils.initialize_session(use_highvolume=True)

country = "Bhutan"
region = ee.FeatureCollection(EEUtils.country_bbox(country))

start = "2021-01-01"
end = "2021-02-01"

# projects/servir-sco-assets/assets/Bhutan/Sentinel1Ascending2021

s1 = hf.Sentinel1(region, start, end)
print(f"number of s1 images: {s1.n_images}")

dems = ee.ImageCollection("COPERNICUS/DEM/GLO30")

print(f"number of dem images: {dems.select('DEM').filterBounds(region).size().getInfo()}")

# Extract out the DEM bands from Copernicus DEM
dem = dems.select("DEM").filterBounds(region).mosaic().unmask(0)

s1_asc = hf.Sentinel1Asc(region, start, end)
# Apply a (psuedo-) terrain flattening algorithm to S1 data
# elevation (ee.Image): Input DEM to calculate slope corrections from
# buffer (int, optional): buffer in meters for layover/shadow mask. If zero then no buffer will be applied. default = 0
s1_asc_flat = s1_asc.apply_func(hf.slope_correction, elevation=dem, buffer=100)

# Apply a speckle filter algorithm to S1 data
s1_asc_filtered = s1_asc_flat.apply_func(hf.refined_lee)
print(f"number of s1_asc_filtered images: {s1_asc_filtered.n_images}")

s1_asc_filtered = s1_asc_filtered.collection.median().clip(region)
s1_asc_filtered = s1_asc_filtered.set("system:time_start", ee.Date(start).millis(), "system:time_end", ee.Date(end).advance(-1, "day").millis())

s1_desc = hf.Sentinel1Desc(region, start, end)
# Apply a (psuedo-) terrain flattening algorithm to S1 data
# elevation (ee.Image): Input DEM to calculate slope corrections from
# buffer (int, optional): buffer in meters for layover/shadow mask. If zero then no buffer will be applied. default = 0
s1_desc_flat = s1_desc.apply_func(hf.slope_correction, elevation=dem, buffer=100)

# Apply a speckle filter algorithm to S1 data
s1_desc_filtered = s1_desc_flat.apply_func(hf.refined_lee)
print(f"number of s1_desc_filtered images: {s1_asc.n_images}")

s1_desc_filtered = s1_desc_filtered.collection.median().clip(region)
s1_desc_filtered = s1_desc_filtered.set("system:time_start", ee.Date(start).millis(), "system:time_end", ee.Date(end).advance(-1, "day").millis());

export_kwargs = {
    "region": region.geometry().bounds().getInfo()["coordinates"],
    "scale": 10,
}

EEUtils.export_image(s1_asc_filtered, export_type="asset", **{**export_kwargs, "asset_id": f"projects/servir-sco-assets/assets/Bhutan/Sentinel1Ascending2021/Ascending_{start}"})
EEUtils.export_image(s1_desc_filtered, export_type="asset", **{**export_kwargs, "asset_id": f"projects/servir-sco-assets/assets/Bhutan/Sentinel1Ascending2021/Descending_{start}"})
