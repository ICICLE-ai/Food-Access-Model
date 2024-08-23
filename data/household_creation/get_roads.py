import osmnx as ox
import psycopg2
from config import USER, PASS
import psycopg2

# Connect to the PostgreSQL database
connection = psycopg2.connect(
    host="localhost",
    database="FASS_DB",
    user=USER,
    password=PASS
)
cursor = connection.cursor()

# SQL query to create the 'roads' table
create_table_query = '''
CREATE TABLE roads (
    name VARCHAR(100),
    highway VARCHAR(100),
    length VARCHAR(20),
    geometry TEXT,
    service VARCHAR(100)
);
'''
# SQL query to drop the 'roads' table if it exists
drop_table_query = 'DROP TABLE IF EXISTS roads;'

# Execute the drop table command
cursor.execute(drop_table_query)

# Execute the create table command
cursor.execute(create_table_query)

# Commit the transaction to save changes
connection.commit()

place_name = "Franklin County, Ohio, USA"

#Get road network from open street maps
G = ox.graph_from_point((39.959813,-83.00514),dist=5000, network_type='all',retain_all=True)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
print(gdf_edges)

#convert to epsg:3857
gdf_edges = gdf_edges.to_crs("epsg:3857")
gdf_edges = gdf_edges[["name","highway","length","geometry","service"]]

# Save the DataFrame to a CSV file
#gdf_edges.to_csv('data/household_creation/roads.csv', index=False)

# Insert data into the table using a SQL query
insert_query = "INSERT INTO roads (name,highway,length,geometry,service) VALUES (%s, %s, %s, %s, %s)"
for index,row in gdf_edges.iterrows():
    cursor.execute(insert_query, (row["name"],row["highway"],str(row["length"]),str(row["geometry"]),row["service"]))

# Commit the transaction
connection.commit()

# Close the cursor and connection
cursor.close()
connection.close()