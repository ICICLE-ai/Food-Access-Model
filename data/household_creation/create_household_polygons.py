import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LineString
import shapely
import random
import pyproj
import csv
import rtree
import time
import math
from shapely.strtree import STRtree
import rasterio
from household_constants import(
    income_ranges,
    size_index_dict,
    workers_index_dict
)
from get_census_data import data


# Create a list of all stores so that we can test if households overlap with them
map_elements = list()
stores = pd.read_csv("data/household_creation/features.csv")
for index,row in stores.iterrows():
    point = shapely.wkt.loads(row["geometry"])
    polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
    map_elements.append(polygon.buffer(20))

housing_areas = []
roads_df = pd.read_csv("data/household_creation/roads.csv", low_memory=False)
for index,row in roads_df.iterrows():
    if (row["highway"] == "residential") or (row["highway"] == "living_street") or (row["service"] == "alley"):
        housing_areas.append(shapely.wkt.loads(row["geometry"]).buffer(30))
        map_elements.append(shapely.wkt.loads(row["geometry"]).buffer(2))
    elif (row["highway"] == "motorway"):
        map_elements.append(shapely.wkt.loads(row["geometry"]).buffer(75))
    elif (row["highway"] == "trunk"):
        map_elements.append(shapely.wkt.loads(row["geometry"]).buffer(50))
    elif (row["highway"] == "primary"):
        map_elements.append(shapely.wkt.loads(row["geometry"]).buffer(10))
    elif (row["highway"] == "secondary"):
        map_elements.append(shapely.wkt.loads(row["geometry"]).buffer(10))
    elif isinstance(shapely.wkt.loads(row["geometry"]), LineString):
        map_elements.append(shapely.wkt.loads(row["geometry"]))
map_elements_index = STRtree(map_elements)

with rasterio.open('data/household_creation/county_raster.tif') as src:
    band1 = src.read(1)  # Read the first band
    raster_crs = src.crs  # Get the CRS of the raster
    transform_affline = src.transform # Get the affine transformation of the raster

#helper method to switch x and y in a shapely Point
def swap_xy(x, y):
    return y, x

houses = list()
houses_index = rtree.index.Index()
households = pd.DataFrame(columns = ["id","polygon","income","household_size","vehicles","number_of_workers"])
total_count = 0
housing_areas_count = 0
for housing_area in housing_areas:
    housing_areas_count+=1
    print(str(round(housing_areas_count/len(housing_areas)*100)) + "%")
    count = 0
    # Get the exterior coordinates of the polygon
    exterior_coords = list(housing_area.exterior.coords)
    # Create LineStrings for each edge
    edges = [LineString([exterior_coords[i], exterior_coords[i+1]]) 
            for i in range(len(exterior_coords) - 1)]
        
    for edge in edges:
        length = edge.length
        coord1 = edge.coords[0]
        coord2 = edge.coords[1]
        vector_direction = (coord2[0] - coord1[0], coord2[1] - coord1[1])
        temp = (vector_direction[0])*(vector_direction[0]) + (vector_direction[1])*(vector_direction[1])
        vector_magnitude = math.sqrt(temp)
        normalized_vector = (0,0)
        if vector_magnitude != 0:
            normalized_vector = (vector_direction[0]/vector_magnitude,vector_direction[1]/vector_magnitude)
        for i in range(int(vector_magnitude/20)+1):
            location = Point(coord1[0]+normalized_vector[0]*i*30,coord1[1]+normalized_vector[1]*i*30)

            house = Polygon(((location.x+10, location.y+10),
                            (location.x, location.y+20),
                            (location.x-10, location.y+10),
                            (location.x+10, location.y+10),
                            (location.x-10, location.y+10),
                            (location.x-10, location.y-5),
                            (location.x-3, location.y-5),
                            (location.x-3, location.y+3),
                            (location.x+3, location.y+3),
                            (location.x+3, location.y-5),
                            (location.x-3, location.y-5),
                            (location.x+10, location.y-5)))
            
            
            valid = True
            intersecting_map_elements_indexes = map_elements_index.query(house)
            if len(intersecting_map_elements_indexes) != 0:
                for index in intersecting_map_elements_indexes:
                    if map_elements[index].intersects(house):
                        valid=False
                        continue
            if not valid:
                continue

            
            intersecting_houses_indexes = houses_index.intersection(house.bounds)
            if len(list(intersecting_houses_indexes)) != 0:
                continue

            houses.append(house)
            houses_index.add(total_count,house.bounds)
            households.loc[total_count] = {
                "id":total_count,
                "polygon":house,
                "income":0,
                "vehicles":0,
                "household_size":0,
                "number_of_workers":0
                }
            total_count+=1

households.to_csv('data/households.csv', index=False)
print(households)