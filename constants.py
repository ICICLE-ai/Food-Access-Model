from food_access_model.distance import Distance


# Radius that households look for stores on first search iteration.
# Units => meters.
SEARCHRADIUS_DISTANCE = Distance.from_meters(500)
# Backward-compatible numeric constant for existing call sites.
SEARCHRADIUS = int(SEARCHRADIUS_DISTANCE.meters)

#crs geometry for map (this is web mercator)
CRS = "3857"

#File paths
HOUSEHOLDSFILEPATH = "data/households.csv"
STORESFILEPATH = "data/stores.csv"
COUNTYDATAFILEPATH = "data/county_data.csv"
GEODATAFILEPATH = "data/tract_boundaries.zip"
