from shapely.geometry import LineString
import osmnx as ox
import shapely
import pyproj
import csv
import pandas as pd

place_name = "Franklin County, Ohio, USA"

#helper method to switch x and y in a shapely Point
def swap_xy(x, y):
    return y, x

G = ox.graph_from_place(place_name, network_type='all')
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
roads = [["geometry","type"]]
for index,row in gdf_edges.iterrows():
    road = LineString(row["geometry"])
    road = shapely.ops.transform(swap_xy, road)
    project = pyproj.Transformer.from_proj(
        pyproj.Proj('epsg:4326'), # source coordinate system
        pyproj.Proj('epsg:3857')) # destination coordinate system
    road = shapely.ops.transform(project.transform, road)
    roads.append([road,row["highway"]])

# Convert the 2D list to a DataFrame
df = pd.DataFrame(roads[1:], columns=roads[0])

# Save the DataFrame to a CSV file
df.to_csv('data/household_creation/roads.csv', index=False)