import osmnx as ox
import psycopg2
from config import USER, PASS
import shapely
from shapely.geometry import Point, Polygon, LineString
from shapely.strtree import STRtree
import rtree
import time
import math
import pandas as pd
import requests
from zipfile import ZipFile
import tempfile
import os
import geopandas
from io import BytesIO
from data.household_constants import(
    households_variables_dict,
    households_key_list,
    FIBSCODE,
    YEAR
)


place_name = "Franklin County, Ohio, USA"

county_code = FIBSCODE[2:]
state_code = FIBSCODE[:2]

from config import APIKEY

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
                    geodata = geopandas.read_file(file_path)


#Merge geographical dataframe (containing shapely ploygons) with census data
geodata.crs = 'EPSG:3857'
county_geodata = geodata[geodata['COUNTYFP'] == county_code]
county_geodata = county_geodata.rename(columns={"TRACTCE":"tract_y"})
county_geodata["tract_y"] = county_geodata["tract_y"].astype(int)
county_data["tract_y"] = county_data["tract_y"].astype(int)
data = pd.merge(county_geodata, county_data, on = "tract_y", how="inner")
data.rename(columns=households_variables_dict, inplace = True)

# Connect to the PostgreSQL database
connection = psycopg2.connect(
    host="localhost",
    database="FASS_DB",
    user=USER,
    password=PASS
)
cursor = connection.cursor()

# SQL query to create the 'roads' table
create_roads_query = '''
CREATE TABLE roads (
    name TEXT,
    highway VARCHAR(30),
    length VARCHAR(20),
    geometry TEXT,
    service VARCHAR(30)
);
'''

# SQL query to create the 'roads' table
create_food_stores_query = '''
CREATE TABLE food_stores (
    shop VARCHAR(15),
    geometry VARCHAR(50),
    amenity VARCHAR(20),
    name VARCHAR(50)
);
'''

# Execute the drop table command for roads
cursor.execute('DROP TABLE IF EXISTS roads;')

# Execute the drop table command for stores
cursor.execute('DROP TABLE IF EXISTS food_stores;')

# Execute the create table command
cursor.execute(create_roads_query)

# Execute the create table command
cursor.execute(create_food_stores_query)

place_name = "Franklin County, Ohio, USA"
map_elements = list()
housing_areas = list()

#Get road network from open street maps
G = ox.graph_from_point((39.959813,-83.00514),dist=5000, network_type='all',retain_all=True)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

#convert to epsg:3857
gdf_edges = gdf_edges.to_crs("epsg:3857")
gdf_edges = gdf_edges[["name","highway","length","geometry","service"]]

# Insert data into the table using a SQL query
roads_query = "INSERT INTO roads (name,highway,length,geometry,service) VALUES (%s, %s, %s, %s, %s)"
for index,row in gdf_edges.iterrows():
    if (row["highway"] == "residential") or (row["highway"] == "living_street") or (row["service"] == "alley"):
        housing_areas.append(row["geometry"].buffer(30))
        map_elements.append((row["geometry"]).buffer(2))
    elif (row["highway"] == "motorway"):
        map_elements.append((row["geometry"]).buffer(75))
    elif (row["highway"] == "trunk"):
        map_elements.append((row["geometry"]).buffer(50))
    elif (row["highway"] == "primary"):
        map_elements.append((row["geometry"]).buffer(10))
    elif (row["highway"] == "secondary"):
        map_elements.append((row["geometry"]).buffer(10))
    elif isinstance((row["geometry"]), LineString):
        map_elements.append((row["geometry"]))
    cursor.execute(roads_query, (row["name"],row["highway"],str(row["length"]),str(row["geometry"]),row["service"]))

#Get food stores
features = ox.features.features_from_point((39.959813,-83.00514),dist=5000,tags = {"shop":["convenience",'supermarket',"butcher","wholesale","farm",'greengrocer',"health_food",'grocery']})
features = features.to_crs("epsg:3857")
features = features[["shop","geometry","amenity","name"]]

#Insert food stores into postgres database

food_stores_query = "INSERT INTO food_stores (shop,geometry,amenity,name) VALUES (%s, %s, %s, %s)"
for index,row in features.iterrows():
    if not isinstance(row["geometry"],Point):
        point = row["geometry"].centroid
        polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
        map_elements.append(polygon.buffer(20))
        cursor.execute(food_stores_query, (row["shop"],str(row["geometry"].centroid),row["amenity"],row["name"]))
    else:
        point = row["geometry"]
        polygon = Polygon(((point.x, point.y+50),(point.x+50, point.y-50),(point.x-50, point.y-50)))
        map_elements.append(polygon.buffer(20))
        cursor.execute(food_stores_query, (row["shop"],str(row["geometry"]),row["amenity"],row["name"]))

map_elements_index = STRtree(map_elements)

# SQL query to create the 'households' table
create_households_query = '''
CREATE TABLE households (
    id VARCHAR(7),
    polygon TEXT,
    income VARCHAR(6),
    household_size VARCHAR(1),
    vehicles VARCHAR(1),
    number_of_workers VARCHAR(1)

);
'''

# Execute the drop table command
cursor.execute('DROP TABLE IF EXISTS households;')

# Execute the create table command
cursor.execute(create_households_query)

household_query = "INSERT INTO households (id,polygon,income,household_size,vehicles,number_of_workers) VALUES (%s, %s, %s, %s, %s, %s)"



houses = list()
houses_index = rtree.index.Index()
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
            
            cursor.execute(household_query, (total_count,str(house),0,0,0,0))
            total_count+=1

# Close the cursor and connection
connection.commit()
cursor.close()
connection.close()