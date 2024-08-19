import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
import shapely
import random
import pyproj
import csv
import time
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
stores = pd.read_csv("data/stores.csv")
for index,row in stores.iterrows():
    lat = row["latitude"]
    lon = row["longitude"]
    point = Point(lat,lon)
    project = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:4326'), # source coordinate system
        pyproj.Proj('epsg:3857')) # destination coordinate system
    point = shapely.ops.transform(project.transform, point)  # apply projection
    polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
    map_elements.append(polygon.buffer(20))  # apply projection

housing_areas = []
with open('data/household_creation/roads.csv', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == "geometry":
            continue
        if (row[1] == "residential"):
            housing_areas.append(shapely.wkt.loads(row[0]).buffer(100))
            map_elements.append(shapely.wkt.loads(row[0]))
        elif (row[1] == "motorway") or (row[1] == "motorway_link"):
            map_elements.append(shapely.wkt.loads(row[0]).buffer(100))
        elif (row[1] == "primary"):
            map_elements.append(shapely.wkt.loads(row[0]).buffer(10))
        else:
            map_elements.append(shapely.wkt.loads(row[0]).buffer(1))
map_elements_index = STRtree(map_elements)


housing_areas_index = STRtree(housing_areas)
temp_indexes = housing_areas_index.query(Polygon(((-9231087.591381988,4855583.605717118),(-9230359.092447992,4859016.425386268),(-9231386.539835587,4861464.9938469855),(-9237985.24736839,4863046.919173137))))
housing_areas = [housing_areas[i] for i in temp_indexes]
# Open the raster file and read the first band
#with rasterio.open('data/household_creation/county_raster.tif') as src:
#    band1 = src.read(1)  # Read the first band
#    raster_crs = src.crs  # Get the CRS of the raster
#    transform_affline = src.transform # Get the affine transformation of the raster

#helper method to switch x and y in a shapely Point
def swap_xy(x, y):
    return y, x

houses = list()
houses_index = STRtree(houses)
households = pd.DataFrame(columns = ["id","polygon","income","household_size","vehicles","number_of_workers"])
total_count = 0
housing_areas_count = 0
for housing_area in housing_areas:
    housing_areas_count+=1
    print(housing_areas_count/len(housing_areas))
    count = 0
    while count<10:
        min_x, min_y, max_x, max_y = housing_area.bounds
        location = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if housing_area.contains(location):
            house = Polygon(((location.x+20, location.y+20),
                            (location.x, location.y+40),
                            (location.x-20, location.y+20),
                            (location.x+20, location.y+20),
                            (location.x-20, location.y+20),
                            (location.x-20, location.y-10),
                            (location.x-5, location.y-10),
                            (location.x-5, location.y+5),
                            (location.x+5, location.y+5),
                            (location.x+5, location.y-10),
                            (location.x-5, location.y-10),
                            (location.x+20, location.y-10)))
            
            intersecting_map_elements_indexes = map_elements_index.query(house)
            if len(intersecting_map_elements_indexes) != 0:
                count+=1
                continue
            
            intersecting_houses_indexes = houses_index.query(house)
            if len(intersecting_houses_indexes) != 0:
                count+=1
                continue

            houses.append(house)
            houses_index = STRtree(houses)
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