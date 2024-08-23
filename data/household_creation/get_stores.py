import osmnx as ox
import pandas as pd
import pyproj
from shapely.geometry import Point,Polygon
import shapely
from database_connection import engine


place_name = "Franklin County, Ohio, USA"

#helper method to switch x and y in a shapely Point
def swap_xy(x, y):
    return y, x

features = ox.features.features_from_point((39.959813,-83.00514),tags = {"shop":["convenience",'supermarket',"butcher","wholesale","farm",'greengrocer',"health_food",'grocery']},dist=5000)
for index,row in features.iterrows():
    if not isinstance(row["geometry"],Point):
        features.loc[index, "geometry"] = row["geometry"].centroid
features = features.to_crs("epsg:3857")

# Save the DataFrame to a CSV file
#features.to_sql('stores', engine, if_exists='replace', index=False)
features.to_csv("data/household_creation/features.csv")