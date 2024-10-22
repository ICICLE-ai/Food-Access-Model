import osmnx as ox
import psycopg2
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
import rtree
import time
import math
import pandas as pd
import requests
from zipfile import ZipFile
import tempfile
import shapely
import googlemaps
from datetime import datetime
import os
import numpy as np
import geopandas
import random
from psycopg2 import extras
import shapely.geometry as geometry
from pyproj import Transformer
from io import BytesIO
import csv
from household_constants import(
    households_variables_dict,
    households_key_list,
    FIBSCODE,
    YEAR,
    income_ranges,
    size_index_dict,
    workers_index_dict
)


place_name = "Franklin County, Ohio, USA"

county_code = FIBSCODE[2:]
state_code = FIBSCODE[:2]

center_point = (39.949614, -82.999420)
dist = 400

from config import APIKEY, GOOGLEAPIKEY, USER, PASS, NAME, HOST, PORT

#Read csvs into pandas dataframes
#For loop runs a census API pull for each loop iteration
#this is neccessary because we can only pull 50 variables at a time and we have >50
county_data = pd.DataFrame()
for count in range(int(len(households_key_list)/50)+1):
    variables = ""
    #put variables from above into census readable lists
    if ((count+1)*50) > len(households_key_list):
        variables = ",".join(households_key_list[(50*count):])
    elif count == 0:
        if (int(len(households_key_list)/50)+1) == 1:
            variables = ",".join(households_key_list[:])
        else:
            variables = ",".join(households_key_list[:(50*(count+1)-1)])
    else:
        variables = ",".join(households_key_list[(50*count):(50*(count+1)-1)])
    #Pull data and add to dataframe
    url = f"https://api.census.gov/data/{YEAR}/acs/acs5?get=NAME,{variables}&for=tract:*&in=state:{state_code}&in=county:{county_code}&key={APIKEY}"
    response = requests.request("GET", url)
    if len(county_data != 0):
        county_data = pd.merge(pd.DataFrame(response.json()[1:], columns=response.json()[0]), county_data, on='NAME', how='inner')
    else:
        county_data = pd.DataFrame(response.json()[1:], columns=response.json()[0])


# Load in geographical tract data
tract_url = f"https://www2.census.gov/geo/tiger/TIGER{YEAR}/TRACT/tl_{YEAR}_{state_code}_tract.zip"
response = requests.request("GET", tract_url)
# Use BytesIO to handle the zip file in memory
with ZipFile(BytesIO(response.content)) as zip_ref:
    # Create a temporary directory to extract the zip file
    with tempfile.TemporaryDirectory() as tmpdirname:
        zip_ref.extractall(tmpdirname)
        
        # Find the shapefile or GeoJSON file in the extracted contents
        for root, dirs, files in os.walk(tmpdirname):
            for file in files:
                if file.endswith(".shp") or file.endswith(".geojson"):
                    file_path = os.path.join(root, file)
                    # Load the file into a GeoDataFrame
                    geodata = (geopandas.read_file(file_path)).to_crs("epsg:3857")


#Merge geographical dataframe (containing shapely ploygons) with census data
county_geodata = geodata[geodata['COUNTYFP'] == county_code]
county_geodata = county_geodata.rename(columns={"TRACTCE":"tract_y"})
county_geodata["tract_y"] = county_geodata["tract_y"].astype(int)
county_data["tract_y"] = county_data["tract_y"].astype(int)
data = pd.merge(county_geodata, county_data, on = "tract_y", how="inner")
data.rename(columns=households_variables_dict, inplace = True)
data = data.to_crs("epsg:3857")
tract_index = STRtree(data["geometry"])

# Connect to the PostgreSQL database
connection = psycopg2.connect(
    host=HOST,
    database=NAME,
    user=USER,
    password=PASS,
    port=PORT
)
cursor = connection.cursor()

# Execute the drop table command for roads
cursor.execute('DROP TABLE IF EXISTS roads;')

# Execute the drop table command for stores
cursor.execute('DROP TABLE IF EXISTS food_stores;')

# SQL query to create the 'roads' table
create_roads_query = '''
CREATE TABLE roads (
    name TEXT,
    highway VARCHAR(30),
    length NUMERIC,
    geometry TEXT,
    service VARCHAR(30)
);
'''

# SQL query to create the 'roads' table
create_food_stores_query = '''
CREATE TABLE food_stores (
    shop VARCHAR(15),
    geometry TEXT,
    name VARCHAR(50)
);
'''

# Execute the create table command
#cursor.execute(create_roads_query)

# Execute the create table command
cursor.execute(create_food_stores_query)

place_name = "Franklin County, Ohio, USA"
map_elements = list()
housing_areas = list()

#Get road network from open street maps
G = ox.graph_from_point(center_point,dist=dist, network_type='all',retain_all=True)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

#convert to epsg:3857
gdf_edges = gdf_edges.to_crs("epsg:3857")
if "service" in gdf_edges.columns:
    gdf_edges = gdf_edges[["name","highway","length","geometry","service"]]
else:
    gdf_edges = gdf_edges[["name","highway","length","geometry"]]

# Insert data into the table using a SQL query
for index,row in gdf_edges.iterrows():
    if (row["highway"] == "residential") or (row["highway"] == "living_street"):
        housing_areas.append(row["geometry"].buffer(30))
        map_elements.append((row["geometry"]).buffer(2))
    elif "service" in gdf_edges.columns:
        if (row["service"])=="alley":
            housing_areas.append(row["geometry"].buffer(30))
            map_elements.append((row["geometry"]).buffer(2))
    elif (row["highway"] == "motorway"):
        map_elements.append((row["geometry"]).buffer(100))
    elif (row["highway"] == "trunk"):
        map_elements.append((row["geometry"]).buffer(50))
    elif (row["highway"] == "primary"):
        map_elements.append((row["geometry"]).buffer(10))
    elif (row["highway"] == "secondary"):
        map_elements.append((row["geometry"]).buffer(10))
    elif isinstance((row["geometry"]), LineString):
        map_elements.append((row["geometry"]))

gdf_edges["length"] = gdf_edges["length"].astype(int)
gdf_edges["geometry"] = gdf_edges["geometry"].astype(str)
roads_query = "INSERT INTO roads (name,highway,length,geometry,service) VALUES %s"
data_tuples = list(gdf_edges.itertuples(index=False, name=None))
#extras.execute_values(cursor, roads_query, data_tuples)

#Get food stores
features = ox.features.features_from_point(center_point,dist=dist,tags = {"shop":["convenience",'supermarket',"butcher","wholesale","farm",'greengrocer',"health_food",'grocery']})
features = features.to_crs("epsg:3857")
features = features[["shop","geometry","name"]]

#Insert food stores into postgres database
store_tuples = list()
food_stores_query = "INSERT INTO food_stores (shop,geometry,name) VALUES %s"
for index,row in features.iterrows():
    if not isinstance(row["geometry"],Point):
        point = row["geometry"].centroid
        polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
        map_elements.append(polygon.buffer(20))
        store_tuples.append((str(row["shop"]),str(polygon),str(row["name"])))
    else:
        point = row["geometry"]
        polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
        map_elements.append(polygon.buffer(20))
        store_tuples.append((str(row["shop"]),str(polygon),str(row["name"])))

map_elements_index = STRtree(map_elements)
extras.execute_values(cursor, food_stores_query, store_tuples)

# SQL query to create the 'households' table
create_households_query = '''
CREATE TABLE households (
    id NUMERIC,
    polygon TEXT,
    income NUMERIC,
    household_size NUMERIC,
    vehicles NUMERIC,
    number_of_workers NUMERIC,
    walking_time TEXT,
    biking_time TEXT,
    transit_time TEXT,
    driving_time TEXT
);
'''

# Execute the drop table command
cursor.execute('DROP TABLE IF EXISTS households;')

# Execute the create table command
cursor.execute(create_households_query)

household_query = "INSERT INTO households (id,polygon,income,household_size,vehicles,number_of_workers,walking_time,biking_time,transit_time,driving_time) VALUES %s"

connection.commit()
cursor.close()
connection.close()

houses = list()
houses_index = rtree.index.Index()
total_count = 0
house_tuples = list()
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
            intersecting_map_elements_indexes = map_elements_index.query(Polygon(((location.x+10, location.y+20),
                                                                        (location.x-10, location.y+20),
                                                                        (location.x-10, location.y-5),
                                                                        (location.x+10, location.y-5))))
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
            
            tract_row_numbers = tract_index.query(house.centroid)
            tract_row = 0
            for row_number in tract_row_numbers:
                if (data.loc[row_number,"geometry"]).contains(house.centroid):
                    tract_row = row_number
                    break
            tract = data.loc[tract_row]
            #Iterate through each tract and create households
            #Get amount of people in each tract at each income levela
            income_weights = np.array(tract["10k to 15k":"150k to 200k"]).astype(int)
            if sum(income_weights)==0:
                continue

            # Make a list of incomes to distribute to households
            total_households = int(tract["total households in tract"])
            distributed_incomes = []
            for i in range(14):
                uniform_list = []
                if i != 14:
                    uniform_list = np.random.uniform(income_ranges[i][0],income_ranges[i][1],income_weights[i])
                else:
                    uniform_list = np.random.uniform(200000,200000,income_weights[i])
                distributed_incomes.extend(uniform_list.astype(int))

            vehicle_weights = [
                                int(tract["1 Person(s) 0 Vehicle(s)"]),
                                int(tract["1 Person(s) 1 Vehicle(s)"]),
                                int(tract["1 Person(s) 2 Vehicle(s)"]),
                                int(tract["1 Person(s) 3 Vehicle(s)"]),
                                int(tract["1 Person(s) 4+ Vehicle(s)"]),
                                int(tract["2 Person(s) 0 Vehicle(s)"]),
                                int(tract["2 Person(s) 1 Vehicle(s)"]),
                                int(tract["2 Person(s) 2 Vehicle(s)"]),
                                int(tract["2 Person(s) 3 Vehicle(s)"]),
                                int(tract["2 Person(s) 4+ Vehicle(s)"]),
                                int(tract["3 Person(s) 0 Vehicle(s)"]),
                                int(tract["3 Person(s) 1 Vehicle(s)"]),
                                int(tract["3 Person(s) 2 Vehicle(s)"]),
                                int(tract["3 Person(s) 3 Vehicle(s)"]),
                                int(tract["3 Person(s) 4+ Vehicle(s)"]),
                                int(tract["4+ Person(s) 0 Vehicle(s)"]),
                                int(tract["4+ Person(s) 1 Vehicle(s)"]),
                                int(tract["4+ Person(s) 2 Vehicle(s)"]),
                                int(tract["4+ Person(s) 3 Vehicle(s)"]),
                                int(tract["4+ Person(s) 4+ Vehicle(s)"]),
                                int(tract["0 Worker(s) 0 Vehicle(s)"]),
                                int(tract["0 Worker(s) 1 Vehicle(s)"]),
                                int(tract["0 Worker(s) 2 Vehicle(s)"]),
                                int(tract["0 Worker(s) 3 Vehicle(s)"]),
                                int(tract["0 Worker(s) 4+ Vehicle(s)"]),
                                int(tract["1 Worker(s) 0 Vehicle(s)"]),
                                int(tract["1 Worker(s) 1 Vehicle(s)"]),
                                int(tract["1 Worker(s) 2 Vehicle(s)"]),
                                int(tract["1 Worker(s) 3 Vehicle(s)"]),
                                int(tract["1 Worker(s) 4+ Vehicle(s)"]),
                                int(tract["2 Worker(s) 0 Vehicle(s)"]),
                                int(tract["2 Worker(s) 1 Vehicle(s)"]),
                                int(tract["2 Worker(s) 2 Vehicle(s)"]),
                                int(tract["2 Worker(s) 3 Vehicle(s)"]),
                                int(tract["2 Worker(s) 4+ Vehicle(s)"]),
                                int(tract["3+ Worker(s) 0 Vehicle(s)"]),
                                int(tract["3+ Worker(s) 1 Vehicle(s)"]),
                                int(tract["3+ Worker(s) 2 Vehicle(s)"]),
                                int(tract["3+ Worker(s) 3 Vehicle(s)"]),
                                int(tract["3+ Worker(s) 4+ Vehicle(s)"])
                            ]
            vehicle_weights = [0 if item == -666666666 else item for item in vehicle_weights]

            worker_weights = [
                                int(tract["1 Person(s) 0 Worker(s)"]),
                                int(tract["1 Person(s) 1 Worker(s)"]),
                                int(tract["2 Person(s) 0 Worker(s)"]),
                                int(tract["2 Person(s) 1 Worker(s)"]),
                                int(tract["2 Person(s) 2 Worker(s)"]),
                                int(tract["3 Person(s) 0 Worker(s)"]),
                                int(tract["3 Person(s) 1 Worker(s)"]),
                                int(tract["3 Person(s) 2 Worker(s)"]),
                                int(tract["3 Person(s) 3 Worker(s)"]),
                                int(tract["4+ Person(s) 0 Worker(s)"]),
                                int(tract["4+ Person(s) 1 Worker(s)"]),
                                int(tract["4+ Person(s) 2 Worker(s)"]),
                                int(tract["4+ Person(s) 3+ Worker(s)"]),
                            ]
            worker_weights = [0 if item == -666666666 else item for item in worker_weights]

            household_size_weights = [
                                int(tract["Median Income for 1 Person(s)"]),
                                int(tract["Median Income for 2 Person(s)"]),
                                int(tract["Median Income for 3 Person(s)"]),
                                int(tract["Median Income for 4 Person(s)"]),
                                int(tract["Median Income for 5 Person(s)"]),
                                int(tract["Median Income for 6 Person(s)"]),
                                int(tract["Median Income for 7+ Person(s)"])
                            ]
            household_size_weights = [0 if item == -666666666 else item for item in household_size_weights]
            income_range = random.choices(income_ranges,weights = income_weights)
            income = random.randint(int(income_range[0][0]/1000),int(income_range[0][1]/1000))*1000

            #This is stupid - literally just hardcoded
            household_size = random.choices([1,2,3,4,5,6,7],weights=[1,1,1,1,0,0,0])[0]

            num_workers = 0
            if household_size == 1:
                if sum(worker_weights[:2])==0:
                    num_workers = 0
                else:
                    num_workers = random.choices([0,1], weights=worker_weights[:2], k=1)[0]
            if household_size == 2:
                if sum(worker_weights[2:5])==0:
                    num_workers = 0
                else:
                    num_workers = random.choices([0,1,2], weights=worker_weights[2:5], k=1)[0]
            if household_size == 3:
                if sum(worker_weights[5:9])==0:
                    num_workers = 0
                else:
                    num_workers = random.choices([0,1,2,3], weights=worker_weights[5:9], k=1)[0]
            if household_size >= 4:
                if sum(worker_weights[9:])==0:
                    num_workers = 0
                else:
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
            nearest_store = None
            store_distance = 100000000
            #TODO get nearest SPM and nearest CSPM
            for store in store_tuples:
                store = shapely.wkt.loads(store[1])
                if store.distance(house) <= store_distance:
                    nearest_store = store
            
            # Initialize the Google Maps client with your API key
            gmaps = googlemaps.Client(key=GOOGLEAPIKEY)

            # Define the origin and destination as (latitude, longitude) pairs
            # Define the source CRS and target CRS (e.g., from EPSG:4326 to EPSG:3857)
            source_crs = "EPSG:3857" # WGS84 (lat/lon)
            target_crs = "EPSG:4326" # Web Mercator (meters)

            # Create a transformer object
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

            # Transform the polygon by transforming each coordinate
            house_4326 = [transformer.transform(x, y) for x, y in house.exterior.coords]

            store_4326 = [transformer.transform(x, y) for x, y in nearest_store.exterior.coords]

            # Create a new Shapely polygon with the transformed coordinates
            house_4326 = geometry.Polygon(house_4326)
            store_4326 = geometry.Polygon(store_4326)
            origin = (float(house_4326.centroid.y), float(house_4326.centroid.x))
            destination = (float(store_4326.centroid.y), float(store_4326.centroid.x))
            walking_time = gmaps.directions(origin,
                                                destination,
                                                mode="walking",
                                                departure_time=datetime.now())[0]["legs"][0]["duration"]["text"]
            biking_time = gmaps.directions(origin,
                                                destination,
                                                mode="bicycling",
                                                departure_time=datetime.now())[0]["legs"][0]["duration"]["text"]
            transit_time = gmaps.directions(origin,
                                                destination,
                                                mode="transit",
                                                departure_time=datetime.now())[0]["legs"][0]["duration"]["text"]
            driving_time = gmaps.directions(origin,
                                                destination,
                                                mode="driving",
                                                departure_time=datetime.now())[0]["legs"][0]["duration"]["text"]
            house_tuples.append((total_count,str(house),income,household_size,vehicles,num_workers,walking_time,biking_time,transit_time,driving_time))
            total_count+=1


# Connect to the PostgreSQL database
connection = psycopg2.connect(
    host=HOST,
    database=NAME,
    user=USER,
    password=PASS,
    port=PORT
)
cursor = connection.cursor()

extras.execute_values(cursor, household_query, house_tuples)

# Close the cursor and connection
connection.commit()
cursor.close()
connection.close()