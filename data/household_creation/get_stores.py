import osmnx as ox
import pandas as pd
import pyproj
from shapely.geometry import Point
import shapely
from database_connection import engine


place_name = "Franklin County, Ohio, USA"

#helper method to switch x and y in a shapely Point
def swap_xy(x, y):
    return y, x

features = ox.features.features_from_place(place_name,tags = {"shop":["convenience",'supermarket',"butcher","wholesale","farm",'greengrocer',"health_food",'grocery']})

for index,row in features.iterrows():
    feature_geo = row["geometry"].centroid
    feature_geo = shapely.ops.transform(swap_xy, feature_geo)
    project = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:4326'), # source coordinate system
        pyproj.Proj('epsg:3857')) # destination coordinate system
    feature_geo = shapely.ops.transform(project.transform, feature_geo)
    features.loc[index,"geometry"] = feature_geo

# Save the DataFrame to a CSV file
features.to_sql('stores', engine, if_exists='replace', index=False)