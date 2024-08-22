import osmnx as ox

place_name = "Franklin County, Ohio, USA"

#Get road network from open street maps
G = ox.graph_from_point((39.959813,-83.00514),dist=5000, network_type='all',retain_all=True)
gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)
print(gdf_edges)

#convert to epsg:3857
gdf_edges = gdf_edges.to_crs("epsg:3857")
gdf_edges = gdf_edges[["osmid","name","highway","length","geometry","service"]]

# Save the DataFrame to a CSV file
gdf_edges.to_csv('data/household_creation/roads.csv', index=False)