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

roads = []
with open('data/household_creation/roads.csv', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == "geometry":
            continue
        if (row[1] == "residential"):
            roads.append(shapely.wkt.loads(row[0]).buffer(100))
            map_elements.append(shapely.wkt.loads(row[0]))
        elif (row[1] == "motorway") or (row[1] == "motorway_link"):
            map_elements.append(shapely.wkt.loads(row[0]).buffer(100))
        elif (row[1] == "primary"):
            map_elements.append(shapely.wkt.loads(row[0]).buffer(10))
        else:
            map_elements.append(shapely.wkt.loads(row[0]).buffer(1))
            
# Open the raster file and read the first band
with rasterio.open('data/household_creation/county_raster.tif') as src:
    band1 = src.read(1)  # Read the first band
    raster_crs = src.crs  # Get the CRS of the raster
    transform_affline = src.transform # Get the affine transformation of the raster

#helper method to switch x and y in a shapely Point
def swap_xy(x, y):
    return y, x

households = pd.DataFrame(columns = ["id","polygon","income","household_size","vehicles","number_of_workers"])


"""
# Function to generate a random point within a polygon
def get_house_polygons(tract_polygon,tract_elements,housing_areas):
    housing_area = 0
    for area in housing_areas:
        housing_area+=area.area

    num_houses = int(housing_area/30000)
    print(num_houses)
    houses = list()
    for i in range(num_houses):
        min_x, min_y, max_x, max_y = tract_polygon.bounds
        min_x += 10
        min_y += 10
        max_x = max_x-10
        max_y = max_y-10
        count = 0
        while True:
            if count==500:
                break
            location = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if tract_polygon.contains(location):
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
                count+=1
                in_housing_area = False
                for housing_area in housing_areas:
                    touches = housing_area.contains(location)
                    if touches:
                        in_housing_area = True
                        break
                if  not in_housing_area:
                    continue
                
                not_touching_other_element = True
                for element in tract_elements:
                    touches = house.intersects(element)
                    if touches:
                        not_touching_other_element = False
                        break
                if not not_touching_other_element:
                    continue

                for element in houses:
                    touches = house.intersects(element)
                    if touches:
                        not_touching_other_element = False
                        break

                if not not_touching_other_element:
                    continue

                houses.append(house)
                break
        print(count)
    return houses
"""   
""" 
# Function to generate a random point within a polygon
def get_house_polygons(tract_polygon):
    housing_areas = []
    roads_count = 0
    for i in range(len(roads)):
        if roads[roads_count].intersects(tract_polygon):
            housing_areas.append(roads.pop(roads_count))
        else:
            roads_count+=1
    housing_area_elements = list()
    buffered_tract_polygon = tract_polygon.buffer(200)
    for polygon in map_elements:
        if buffered_tract_polygon.intersects(polygon):
            housing_area_elements.append(polygon)
    houses = list()
    housing_areas_count = 0
    while housing_areas_count < len(housing_areas):
        housing_area = housing_areas[housing_areas_count]
        min_x, min_y, max_x, max_y = housing_area.bounds
        min_x += 0
        min_y += 0
        max_x = max_x-0
        max_y = max_y-0
        count = 0
        while count<500:
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
                count+=1
                not_touching_other_element = True
                for element in housing_area_elements:
                    touches = house.intersects(element)
                    if touches:
                        not_touching_other_element = False
                        break
                if not not_touching_other_element:
                    continue

                for element in houses:
                    touches = house.intersects(element)
                    if touches:
                        not_touching_other_element = False
                        break
                if not not_touching_other_element:
                    continue

                nlcd_transformer = pyproj.Transformer.from_crs(raster_crs, 'EPSG:3857')

                in_housing_area = True
                points = [
                    (location.x+20, location.y+20),
                    (location.x-20, location.y+20),
                    (location.x-20, location.y-10),
                    (location.x+20, location.y-10)
                ]
                for i in range(len(points)):
                    # Transform the point from EPSG:3857 to the raster's CRS
                    x_raster, y_raster = nlcd_transformer.transform(points[i][0], points[i][1])

                    # Convert the transformed coordinates to row and column indices
                    row, col = ~transform_affline * (x_raster, y_raster)

                    # Convert to integers (row and column indices must be integers)
                    row = int(row)
                    col = int(col)

                    # Extract the value from the NumPy array at the calculated indices
                    value = band1[col, row]
                    if value <= 22:
                        in_housing_area = False
                if not in_housing_area:
                    continue
                houses.append(house)
                map_elements.append(house)
                housing_area_elements.append(house)
                break
        housing_areas_count+=1
    return houses
"""
# Function to generate a random point within a polygon
def get_house_polygons(tract_polygon):
    houses = list()

    return houses

total_count = 0
count = 0
start = time.time()
for index,row in data.iterrows():
    if ((row['tract_y']>4000)&(row['tract_y']<6000)):
        start = time.time()
        
        count+=1
        print(count)
        #Create household polygon
        tract_polygon = Polygon(row["geometry"])
        tract_polygon = shapely.ops.transform(swap_xy, tract_polygon)
        project = pyproj.Transformer.from_proj(
            pyproj.Proj('epsg:4326'), # source coordinate system
            pyproj.Proj('epsg:3857')) # destination coordinate system
        tract_polygon = shapely.ops.transform(project.transform, tract_polygon)  # apply

        #Iterate through each tract and create households
        #Get amount of people in each tract at each income levela
        income_weights = np.array(row["10k to 15k":"150k to 200k"]).astype(int)
        if sum(income_weights)==0:
            continue

        # Make a list of incomes to distribute to households
        total_households = int(row["total households in tract"])
        distributed_incomes = []
        for i in range(14):
            uniform_list = []
            if i != 14:
                uniform_list = np.random.uniform(income_ranges[i][0],income_ranges[i][1],income_weights[i])
            else:
                uniform_list = np.random.uniform(200000,200000,income_weights[i])
            distributed_incomes.extend(uniform_list.astype(int))

        vehicle_weights = [
                            int(row["1 Person(s) 0 Vehicle(s)"]),
                            int(row["1 Person(s) 1 Vehicle(s)"]),
                            int(row["1 Person(s) 2 Vehicle(s)"]),
                            int(row["1 Person(s) 3 Vehicle(s)"]),
                            int(row["1 Person(s) 4+ Vehicle(s)"]),
                            int(row["2 Person(s) 0 Vehicle(s)"]),
                            int(row["2 Person(s) 1 Vehicle(s)"]),
                            int(row["2 Person(s) 2 Vehicle(s)"]),
                            int(row["2 Person(s) 3 Vehicle(s)"]),
                            int(row["2 Person(s) 4+ Vehicle(s)"]),
                            int(row["3 Person(s) 0 Vehicle(s)"]),
                            int(row["3 Person(s) 1 Vehicle(s)"]),
                            int(row["3 Person(s) 2 Vehicle(s)"]),
                            int(row["3 Person(s) 3 Vehicle(s)"]),
                            int(row["3 Person(s) 4+ Vehicle(s)"]),
                            int(row["4+ Person(s) 0 Vehicle(s)"]),
                            int(row["4+ Person(s) 1 Vehicle(s)"]),
                            int(row["4+ Person(s) 2 Vehicle(s)"]),
                            int(row["4+ Person(s) 3 Vehicle(s)"]),
                            int(row["4+ Person(s) 4+ Vehicle(s)"]),
                            int(row["0 Worker(s) 0 Vehicle(s)"]),
                            int(row["0 Worker(s) 1 Vehicle(s)"]),
                            int(row["0 Worker(s) 2 Vehicle(s)"]),
                            int(row["0 Worker(s) 3 Vehicle(s)"]),
                            int(row["0 Worker(s) 4+ Vehicle(s)"]),
                            int(row["1 Worker(s) 0 Vehicle(s)"]),
                            int(row["1 Worker(s) 1 Vehicle(s)"]),
                            int(row["1 Worker(s) 2 Vehicle(s)"]),
                            int(row["1 Worker(s) 3 Vehicle(s)"]),
                            int(row["1 Worker(s) 4+ Vehicle(s)"]),
                            int(row["2 Worker(s) 0 Vehicle(s)"]),
                            int(row["2 Worker(s) 1 Vehicle(s)"]),
                            int(row["2 Worker(s) 2 Vehicle(s)"]),
                            int(row["2 Worker(s) 3 Vehicle(s)"]),
                            int(row["2 Worker(s) 4+ Vehicle(s)"]),
                            int(row["3+ Worker(s) 0 Vehicle(s)"]),
                            int(row["3+ Worker(s) 1 Vehicle(s)"]),
                            int(row["3+ Worker(s) 2 Vehicle(s)"]),
                            int(row["3+ Worker(s) 3 Vehicle(s)"]),
                            int(row["3+ Worker(s) 4+ Vehicle(s)"])
                        ]
        vehicle_weights = [0 if item == -666666666 else item for item in vehicle_weights]

        worker_weights = [
                            int(row["1 Person(s) 0 Worker(s)"]),
                            int(row["1 Person(s) 1 Worker(s)"]),
                            int(row["2 Person(s) 0 Worker(s)"]),
                            int(row["2 Person(s) 1 Worker(s)"]),
                            int(row["2 Person(s) 2 Worker(s)"]),
                            int(row["3 Person(s) 0 Worker(s)"]),
                            int(row["3 Person(s) 1 Worker(s)"]),
                            int(row["3 Person(s) 2 Worker(s)"]),
                            int(row["3 Person(s) 3 Worker(s)"]),
                            int(row["4+ Person(s) 0 Worker(s)"]),
                            int(row["4+ Person(s) 1 Worker(s)"]),
                            int(row["4+ Person(s) 2 Worker(s)"]),
                            int(row["4+ Person(s) 3+ Worker(s)"]),
                        ]
        worker_weights = [0 if item == -666666666 else item for item in worker_weights]

        household_size_weights = [
                            int(row["Median Income for 1 Person(s)"]),
                            int(row["Median Income for 2 Person(s)"]),
                            int(row["Median Income for 3 Person(s)"]),
                            int(row["Median Income for 4 Person(s)"]),
                            int(row["Median Income for 5 Person(s)"]),
                            int(row["Median Income for 6 Person(s)"]),
                            int(row["Median Income for 7+ Person(s)"])
                        ]
        household_size_weights = [0 if item == -666666666 else item for item in household_size_weights]
        
        house_polygons = get_house_polygons(tract_polygon)
        for household_num in range(len(house_polygons)):

            location = Point()
            polygon = Polygon()
            polygon = house_polygons[household_num]

            income_range = random.choices(income_ranges,weights = income_weights)
            income = random.randint(int(income_range[0][0]/1000),int(income_range[0][1]/1000))*1000

            #This is stupid - literally just hardcoded
            household_size = random.choices([1,2,3,4,5,6,7],weights=[1,1,1,1,0,0,0])[0]

            num_workers = 0
            if household_size == 1:
                if sum(worker_weights[:2])==0:
                    num_workers = 0
                num_workers = random.choices([0,1], weights=worker_weights[:2], k=1)[0]
            if household_size == 2:
                if sum(worker_weights[2:5])==0:
                    num_workers = 0
                num_workers = random.choices([0,1,2], weights=worker_weights[2:5], k=1)[0]
            if household_size == 3:
                if sum(worker_weights[5:9])==0:
                    num_workers = 0
                num_workers = random.choices([0,1,2,3], weights=worker_weights[5:9], k=1)[0]
            if household_size >= 4:
                if sum(worker_weights[9:])==0:
                    num_workers = 0
                num_workers = random.choices([0,1,2,3], weights=worker_weights[9:], k=1)[0]
            
            size_indexes = None
            if household_size<4:
                size_indexes = size_index_dict[household_size]
            else:
                size_indexes = size_index_dict[4]
            workers_indexes = workers_index_dict[num_workers]
            vehicle_combined_weights = None
            if num_workers != 3:
                vehicle_combined_weights = np.array(vehicle_weights[(size_indexes[0]):(size_indexes[1])])+np.array(vehicle_weights[(workers_indexes[0]):(workers_indexes[1])])
            else:
                vehicle_combined_weights = np.array(vehicle_weights[(size_indexes[0]):(size_indexes[1])])+np.array(vehicle_weights[(workers_indexes[0]):])
            vehicles = random.choices([0,1,2,3,4],weights=vehicle_combined_weights)[0]

            households.loc[total_count] = {
                "id":total_count,
                "polygon":polygon,
                "income":income,
                "vehicles":vehicles,
                "household_size":household_size,
                "number_of_workers":num_workers
                }
            total_count+=1
        print("Time: "+str(time.time()-start))


households.to_csv('data/households.csv', index=False)
print(households)