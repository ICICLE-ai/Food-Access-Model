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
stores = pd.read_csv("data/household_creation/features.csv")
for index,row in stores.iterrows():
    point = shapely.wkt.loads(row["geometry"])
    polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
    map_elements.append(polygon.buffer(20))

housing_areas = []
with open('data/household_creation/roads.csv', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == "geometry":
            continue
        if (row[1] == "residential") or (row[1] == "living_street"):
            housing_areas.append(shapely.wkt.loads(row[0]).buffer(30))
            map_elements.append(shapely.wkt.loads(row[0]))
        elif (row[1] == "motorway"):
           map_elements.append(shapely.wkt.loads(row[0]).buffer(75))
        elif (row[1] == "trunk"):
            map_elements.append(shapely.wkt.loads(row[0]).buffer(50))
        elif (row[1] == "primary"):
            map_elements.append(shapely.wkt.loads(row[0]).buffer(5))
        else:
            map_elements.append(shapely.wkt.loads(row[0]))
map_elements_index = STRtree(map_elements)


housing_areas_index = STRtree(housing_areas)
temp_indexes = housing_areas_index.query(Polygon(((-9241087.591381988,4855583.605717118),(-9240359.092447992,4859016.425386268),(-9241386.539835587,4861464.9938469855),(-9247985.24736839,4863046.919173137))))
housing_areas = [housing_areas[i] for i in temp_indexes]
# Open the raster file and read the first band
with rasterio.open('data/household_creation/county_raster.tif') as src:
    band1 = src.read(1)  # Read the first band
    raster_crs = src.crs  # Get the CRS of the raster
    transform_affline = src.transform # Get the affine transformation of the raster

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
    print(str(round(housing_areas_count/len(housing_areas)*100)) + "%")
    count = 0
    while count<(((housing_area.area)/400)*10):
        min_x, min_y, max_x, max_y = housing_area.bounds
        location = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if housing_area.contains(location):
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
            
            
            intersecting_map_elements_indexes = map_elements_index.query(house)
            if len(intersecting_map_elements_indexes) != 0:
                count+=1
                continue
            
            intersecting_houses_indexes = houses_index.query(house)
            if len(intersecting_houses_indexes) != 0:
                count+=1
                continue

            in_housing_area = True
            points = [
                (location.x+10, location.y+10),
                (location.x-10, location.y+10),
                (location.x-10, location.y-5),
                (location.x+10, location.y-5)
            ]
            for i in range(len(points)):

                # Convert the transformed coordinates to row and column indices
                row, col = ~transform_affline * (points[i][0], points[i][1])

                # Convert to integers (row and column indices must be integers)
                row = int(row)
                col = int(col)

                # Extract the value from the NumPy array at the calculated indices
                value = band1[col, row]
                if (value <= 22) or (value>28):
                    in_housing_area = False
            if not in_housing_area:
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