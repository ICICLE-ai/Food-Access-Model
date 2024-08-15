import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point
import shapely
import random
import pyproj
import csv
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
            roads.append(shapely.wkt.loads(row[0]).buffer(40))
        else:
            map_elements.append(shapely.wkt.loads(row[0]))

#helper method to switch x and y in a shapely Point
def swap_xy(x, y):
    return y, x

households = pd.DataFrame(columns = ["id","polygon","income","household_size","vehicles","number_of_workers"])



# Function to generate a random point within a polygon
def get_house_polygons(tract_polygon,tract_elements,housing_areas):
    housing_area = 0
    for area in housing_areas:
        housing_area+=area.area

    num_houses = int(housing_area/5000)
    houses = list()
    for i in range(num_houses):
        min_x, min_y, max_x, max_y = tract_polygon.bounds
        min_x += 10
        min_y += 10
        max_x = max_x-10
        max_y = max_y-10
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
            
            not_touching = True
            for element in tract_elements:
                touches = house.intersects(element)
                if touches:
                    not_touching = False
                    break

            for element in houses:
                touches = house.intersects(element)
                if touches:
                    not_touching = False
                    break

            if not_touching:
                houses.append(house)
                break
    return houses
            

#Iterate through each tract and create households
total_count = 0
for index,row in data.iterrows():
    tract_elements = list()
    if ((row['tract_y']>5000)&(row['tract_y']<6000)):
        #Create household polygon
        tract_polygon = Polygon(row["geometry"])
        tract_polygon = shapely.ops.transform(swap_xy, tract_polygon)
        project = pyproj.Transformer.from_proj(
            pyproj.Proj('epsg:4326'), # source coordinate system
            pyproj.Proj('epsg:3857')) # destination coordinate system
        tract_polygon = shapely.ops.transform(project.transform, tract_polygon)  # apply

        #Get amount of people in each tract at each income level
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

        for polygon in map_elements:
            if polygon.intersects(tract_polygon):
                tract_elements.append(polygon)

        housing_areas = []
        residential_area = 0
        for polygon in roads:
            if polygon.intersects(tract_polygon):
                residential_area+=polygon.area
                housing_areas.append(polygon)

        house_polygons = get_house_polygons(tract_polygon,tract_elements,housing_areas)
        map_elements.extend(house_polygons)

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
                num_workers = random.choices([0,1], weights=worker_weights[:2], k=1)[0]
            if household_size == 2:
                num_workers = random.choices([0,1,2], weights=worker_weights[2:5], k=1)[0]
            if household_size == 3:
                num_workers = random.choices([0,1,2,3], weights=worker_weights[5:9], k=1)[0]
            if household_size >= 4:
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
            tract_elements.append(polygon)
        map_elements.extend(tract_elements)



households.to_csv('data/households.csv', index=False)
print(households)