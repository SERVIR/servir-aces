# -*- coding: utf-8 -*-

try:
    from aces.ee_utils import EEUtils
except ModuleNotFoundError:
    print("ModuleNotFoundError: Attempting to import from parent directory.")
    import os, sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

    from aces.ee_utils import EEUtils

import ee

EEUtils.initialize_session(use_highvolume=True)

country = "Bhutan"
region = "Punakha"
year = 2021

dem = ee.Image("MERIT/DEM/v1_0_3")

# baseModule = require("users/biplovbhandari/ACES_2:main.js")

planet_asia = ee.ImageCollection("projects/planet-nicfi/assets/basemaps/asia")

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

# Rice growing zone
rice_zone = dem.gt(1000).And(dem.lte(2600))
ic = planet_asia.filterBounds(region_fc.geometry()).filterDate(f"{year}-01-01", f"{year}-12-30")

# nlcms
# class number to name mapping
# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# water_body, snow, glacier, forest, riverbed, built_up,cropland, bare_soil, bare_rock, grassland, 
nlcms = ee.Image(f"projects/servir-hkh/RLCMS/HKH/landcover/hkh_landcover-{year}").clip(region_fc)

# 
buildings  = ee.FeatureCollection("projects/sat-io/open-datasets/MSBuildings/Bhutan")
buildings_layer = ee.Image().byte().paint(buildings, 1).unmask(0)

forest = nlcms.eq(4)
builtUp = nlcms.eq(6)

# Mosaic: Apr to Oct
img_mar = planet_asia.filterDate(ee.Date.fromYMD(year, 3, 1), ee.Date.fromYMD(year, 3, 1).advance(1, "month").advance(-1, "day")).first()
img_apr = planet_asia.filterDate(ee.Date.fromYMD(year, 4, 1), ee.Date.fromYMD(year, 4, 1).advance(1, "month").advance(-1, "day")).first()
img_may = planet_asia.filterDate(ee.Date.fromYMD(year, 5, 1), ee.Date.fromYMD(year, 5, 1).advance(1, "month").advance(-1, "day")).first()
img_jun = planet_asia.filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 6, 1).advance(1, "month").advance(-1, "day")).first()
img_jul = planet_asia.filterDate(ee.Date.fromYMD(year, 7, 1), ee.Date.fromYMD(year, 7, 1).advance(1, "month").advance(-1, "day")).first()
img_aug = planet_asia.filterDate(ee.Date.fromYMD(year, 8, 1), ee.Date.fromYMD(year, 8, 1).advance(1, "month").advance(-1, "day")).first()
img_sep = planet_asia.filterDate(ee.Date.fromYMD(year, 9, 1), ee.Date.fromYMD(year, 9, 1).advance(1, "month").advance(-1, "day")).first()

rice_mosaic = ee.ImageCollection.fromImages(ee.List([img_apr, img_may, img_jun, img_jul, img_aug, img_sep])).median().mask(rice_zone)

none_masked_region = rice_mosaic.select(["B", "G"]).int32().reduceToVectors(
    reducer = ee.Reducer.firstNonNull(),
    geometry = region_fc,
    scale = 90,
    bestEffort = True,
    maxPixels = 1E13,
    tileScale = 16,
)

# Make the training dataset for k-means.
kMeans_training = rice_mosaic.sample(
    region = none_masked_region.geometry(),
    scale = 5,
    numPixels = 500,
    seed = 20,
    geometries = True,
)

# Instantiate the clusterer and train it.
# 7 has been an optimal class
num_k_class = 7
kMeans_clusterer = ee.Clusterer.wekaKMeans(num_k_class).train(kMeans_training)

# Cluster the input using the trained clusterer.
kMeans_cluster = ic.mosaic().cluster(kMeans_clusterer).updateMask(rice_zone)

# remap the clusters
from_class = [0, 1, 2, 3, 4, 5, 6]
# To class mapping
#  0 - cropland
#  1 - forest
#  2 - Built up
#  3 - water body, and others
# to_class = [0, 1, 2, 3, 0, 0, 1] # Paro # double check
to_class = [3, 1, 1, 0, 1, 1, 1] # Punakha

kMeans_cluster = kMeans_cluster.remap(from_class, to_class).rename("cluster")

kMeans_cluster = kMeans_cluster.where(nlcms.eq(1), ee.Image(3)) # class 1 // water
kMeans_cluster = kMeans_cluster.where(nlcms.eq(5), ee.Image(3)) # class 1 // river bed
kMeans_cluster = kMeans_cluster.where(nlcms.eq(4), ee.Image(1)) # class 4 // forest
kMeans_cluster = kMeans_cluster.where(nlcms.eq(6), ee.Image(2)) # class 6 // built-up

sample_kwargs = {
    "class_values": [0, 1, 2, 3],
    "class_points": [2000, 600, 600, 600],
}

stratified_points = EEUtils.generate_stratified_samples(
    kMeans_cluster, none_masked_region.geometry(),
    numPoints = 750,
    classBand = "cluster",
    **sample_kwargs
)

print("number of points per class", stratified_points.aggregate_histogram("cluster").getInfo())

export_kwargs = {
    "asset_id": "projects/servir-sco-assets/assets/Bhutan/ACES_2/stratifiedPoints_punakha_2021",
    "description": "stratifiedPoints_punakha_2021",
}

EEUtils.export_collection_data(stratified_points, export_type=["asset", "drive"], start_training=True, **export_kwargs)
