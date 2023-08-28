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
region = "Paro"
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
dem_punakha = dem.clip(punakha)
rice_zone1_punakha = dem_punakha.gt(1500).And(dem_punakha.lte(2600))
rice_zone2_punakha = dem_punakha.gt(1000).And(dem_punakha.lte(1500))
rice_zone_punakha = dem_punakha.gt(1000).And(dem_punakha.lte(2600))
rice_zone_punakha = rice_zone_punakha.selfMask()

punakha_image = punakha.reduceToImage(properties = ["reduceId"], reducer = ee.Reducer.first())

planet_asia = planet_asia.map(lambda img: img.multiply(0.0001).copyProperties(img, img.toDictionary().keys())
                         .set("system:time_start", img.get("system:time_start"), "system:time_end", img.get("system:time_end"))
                        )


# Export Planet Mosaic

# before growing season
# mar, apr, may
img_mar = planet_asia.filterDate(ee.Date.fromYMD(year, 3, 1), ee.Date.fromYMD(year, 3, 1).advance(1, "month").advance(-1, "day")).first()
img_apr = planet_asia.filterDate(ee.Date.fromYMD(year, 4, 1), ee.Date.fromYMD(year, 4, 1).advance(1, "month").advance(-1, "day")).first()
img_may = planet_asia.filterDate(ee.Date.fromYMD(year, 5, 1), ee.Date.fromYMD(year, 5, 1).advance(1, "month").advance(-1, "day")).first()
img_jun = planet_asia.filterDate(ee.Date.fromYMD(year, 6, 1), ee.Date.fromYMD(year, 6, 1).advance(1, "month").advance(-1, "day")).first()

before_collection = ee.ImageCollection([img_mar, img_apr, img_may])
print("collection_before", before_collection.size().getInfo())
before_composite = before_collection.median()

before_reducers = EEUtils.get_image_collection_statistics(before_collection)

before_indices = before_collection.map(EEUtils.calculate_planet_indices)
before_indices_reducers = EEUtils.get_image_collection_statistics(before_indices)
before_indices = before_indices.median()

before_composite = before_composite.select(["R", "G", "B", "N"], ["red", "green", "blue", "nir"]) \
                          .addBands(before_reducers).addBands(before_indices).addBands(before_indices_reducers)

print("before_composite", before_composite.bandNames().getInfo())


# during gwoing season
# july, aug, sept
img_jul = planet_asia.filterDate(ee.Date.fromYMD(year, 7, 1), ee.Date.fromYMD(year, 7, 1).advance(1, "month").advance(-1, "day")).first()
img_aug = planet_asia.filterDate(ee.Date.fromYMD(year, 8, 1), ee.Date.fromYMD(year, 8, 1).advance(1, "month").advance(-1, "day")).first()
img_sep = planet_asia.filterDate(ee.Date.fromYMD(year, 9, 1), ee.Date.fromYMD(year, 9, 1).advance(1, "month").advance(-1, "day")).first()

during_collection = ee.ImageCollection([img_jul, img_aug, img_sep])
print("during_collection", during_collection.size().getInfo())
during_composite = during_collection.median()

during_reducers = EEUtils.get_image_collection_statistics(during_collection)

during_indices = during_collection.map(EEUtils.calculate_planet_indices)
during_indices_reducers = EEUtils.get_image_collection_statistics(during_indices)
during_indices = during_indices.median()

during_composite = during_composite.select(["R", "G", "B", "N"], ["red", "green", "blue", "nir"]) \
    .addBands(during_reducers).addBands(during_indices).addBands(during_indices_reducers)

print("during_composite", during_composite.bandNames().getInfo())


# after growing season
# oct, nov, dec
img_oct = planet_asia.filterDate(ee.Date.fromYMD(year, 10, 1), ee.Date.fromYMD(year, 10, 1).advance(1, "month").advance(-1, "day")).first()
img_nov = planet_asia.filterDate(ee.Date.fromYMD(year, 11, 1), ee.Date.fromYMD(year, 11, 1).advance(1, "month").advance(-1, "day")).first()
img_dec = planet_asia.filterDate(ee.Date.fromYMD(year, 12, 1), ee.Date.fromYMD(year, 12, 1).advance(1, "month").advance(-1, "day")).first()

after_collection = ee.ImageCollection([img_oct, img_nov, img_dec])
print("after_collection", after_collection.size().getInfo())
after_composite = after_collection.median()

after_reducers = EEUtils.get_image_collection_statistics(after_collection)

after_indices = after_collection.map(EEUtils.calculate_planet_indices)
after_indices_reducers = EEUtils.get_image_collection_statistics(after_indices)
after_indices = after_indices.median()

after_composite = after_composite.select(["R", "G", "B", "N"], ["red", "green", "blue", "nir"]) \
    .addBands(after_reducers).addBands(after_indices).addBands(after_indices_reducers)

print("after_composite", after_composite.bandNames().getInfo())
print("region_fc.geometry().bounds().getInfo()['coordinates']", region_fc.geometry().bounds().getInfo()["coordinates"])
export_kwargs = {
    "region": region_fc.geometry().bounds().getInfo()["coordinates"],
    "scale": 5,
}

EEUtils.export_image(before_composite, export_type="asset",
                     **{**export_kwargs, "asset_id": f"projects/servir-sco-assets/assets/Bhutan/ACES_2/{region}_Composite_{year}/composite_before",
                     "description": f"{year}_composite_before"}
                    )
EEUtils.export_image(during_composite, export_type="asset",
                     **{**export_kwargs, "asset_id": f"projects/servir-sco-assets/assets/Bhutan/ACES_2/{region}_Composite_{year}/composite_during",
                        "description": f"{year}_composite_during"}
                    )
EEUtils.export_image(after_composite, export_type="asset",
                     **{**export_kwargs, "asset_id": f"projects/servir-sco-assets/assets/Bhutan/ACES_2/{region}_Composite_{year}/composite_after",
                        "description": f"{year}_composite_after"}
                    )
