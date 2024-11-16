#!/usr/bin/env python
# coding: utf-8

# ### Import libraries

# In[1]:


# for data
import pandas as pd  
import numpy as np  

# for plotting
import matplotlib.pyplot as plt 
import seaborn as sns  
import folium  
from folium import plugins
import plotly.express as px  

# for graph and routing
import osmnx as ox  
import networkx as nx 
import geopandas as gpd
# for advanced routing
# from ortools.constraint_solver import pywrapcp  
# from ortools.constraint_solver import routing_enums_pb2

import re
import random


# ### import chicago data

# In[68]:


data_merged_routes_diff500=pd.read_csv(r"C:\Users\cclin\Downloads\Chicago_routes_filter.csv")


# ### Complete the traffic related edge attributes in G  

# In[67]:


# create a osmnx graph for Chicago city

place = 'Chicago, Illinois, USA'
G = ox.graph_from_place(place, network_type='drive')

#add speed and travel_time attributes to the edges.
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)


# In[71]:


# create geo-dataframe for both nodes and edges
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)


# In[72]:


# define a function to remove the list object
def remove_list_obj (col):
    col=col.apply(lambda x: tuple(x) if isinstance(x, list) else x)
    return col
edges=edges.apply(remove_list_obj)


# In[73]:


# Extract centroids of each road segment
edges['centroid'] = edges['geometry'].centroid
edges['x'] = edges['centroid'].x
edges['y'] = edges['centroid'].y


# In[74]:


#Use the centroids' coordinates to cluster road segments into regions.
from sklearn.cluster import KMeans

# Define the number of clusters (regions)
n_clusters = 29  

# Apply K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
edges['region'] = kmeans.fit_predict(edges[['x', 'y']])


# In[76]:


# Iterate through the edges in edges gdf and update G
edges=edges.reset_index()
for _, row in edges.iterrows():
    u, v, region = row['u'], row['v'], row['region']
    for key in G[u][v]:
        G[u][v][key]['region'] = region


# In[77]:


data_merged_routes_diff500_chopped=data_merged_routes_diff500.drop_duplicates(['shortest_route'])


# In[78]:


#add Geometry from edge GeoDataFrame to G for plot.
for idx, row in edges.iterrows():
    u, v, key = row['u'], row['v'], row['key']
    G[u][v][key]['geometry'] = row['geometry']


# In[79]:


#add area from edge geodataframe to G for a better concat in the following
for _, row in edges.iterrows():
    u, v, key = row['u'], row['v'], row['key']
    area_value = row['area']
    G[u][v][key]['area'] = area_value


# In[80]:


# based on the understanding of the columns, fill na by the appropriate values in edges gdf
edges['access']=edges['access'].fillna('yes')
edges['junction']=edges['junction'].fillna('no')
edges['bridge']=edges['bridge'].fillna('no')
edges['tunnel']=edges['tunnel'].fillna('no')


# ##### fill na values in lanes of edges gdf

# In[81]:


# convert the tuple of lanes into a single value by average
edges['lanes'] = edges['lanes'].apply(
    lambda x: sum(int(v) for v in x if str(v).isdigit()) / len([v for v in x if str(v).isdigit()]) if isinstance(x, tuple) else 
              int(x) if isinstance(x, str) and x.isdigit() else  
              x
)


# Analysis: based on the values of lanes distribution in each highway type, we can use 'median' to fill na in the types of
# - busway
# - primary
# - secondary
# - trunk
# 
# and use 'mode' to fill na in the types of:
# - motorway_link
# - primary_link
# - residential
# - secondary_link
# - tertiary
# - tertiary_link
# - trunk_link
# - unclassified

# In[82]:


# Compute median for specific highways and apply directly
median_ls = ['busway', 'primary', 'secondary', 'trunk']

edges['lanes'] = edges.groupby('highway')['lanes'].transform(
    lambda group: group.fillna(group.median()) if group.name in median_ls else group
)


# In[83]:


# Compute mode for specific highways and apply directly
mode_ls = ['motorway_link', 'primary_link', 'residential', 'secondary_link', 'tertiary', 'tertiary_link', 'trunk_link', 'unclassified']
edges['lanes'] = edges.groupby('highway')['lanes'].transform(
    lambda group: group.fillna(group.mode().iloc[0]) if group.name in mode_ls else group
)


# In[84]:


# fill na by some default values based on above analysis for the lanes with all na values
edges.loc[edges['highway'] == ('living_street', 'residential'), 'lanes'] = edges.loc[edges['highway'] == ('living_street', 'residential'), 'lanes'].fillna(1)
edges.loc[edges['highway'] == ('motorway_link', 'tertiary'), 'lanes'] = edges.loc[edges['highway'] == ('motorway_link', 'tertiary'), 'lanes'].fillna(1)
edges.loc[edges['highway'] == ('residential', 'living_street'), 'lanes'] = edges.loc[edges['highway'] == ('residential', 'living_street',), 'lanes'].fillna(1)
edges.loc[edges['highway'] == ('secondary', 'tertiary'), 'lanes'] = edges.loc[edges['highway'] == ('secondary', 'tertiary',), 'lanes'].fillna(2)
edges.loc[edges['highway'] == ('tertiary', 'residential'), 'lanes'] = edges.loc[edges['highway'] == ('tertiary','residential'), 'lanes'].fillna(2)
edges.loc[edges['highway'] == ('unclassified', 'residential'), 'lanes'] = edges.loc[edges['highway'] == ('unclassified','residential'), 'lanes'].fillna(2)
edges.loc[edges['highway'] == ('unclassified', 'tertiary'), 'lanes'] = edges.loc[edges['highway'] == ('unclassified','tertiary'), 'lanes'].fillna(2)
edges.loc[edges['highway'] == 'emergency_bay',  'lanes'] = edges.loc[edges['highway'] == 'emergency_bay', 'lanes'].fillna(1)
edges.loc[edges['highway'] == 'living_street',  'lanes'] = edges.loc[edges['highway'] == 'living_street', 'lanes'].fillna(1)


# In[86]:


#update lanes in G 
for _, row in edges.iterrows():
    u, v, key = row['u'], row['v'], row['key']
    lanes_value = row['lanes']
    G[u][v][key]['lanes'] = lanes_value


# In[87]:


#update access in G 
for _, row in edges.iterrows():
    u, v, key = row['u'], row['v'], row['key']
    access_value = row['access']
    G[u][v][key]['access'] = access_value


# In[88]:


#update junction in G 
for _, row in edges.iterrows():
    u, v, key = row['u'], row['v'], row['key']
    junction_value = row['junction']
    G[u][v][key]['junction'] = junction_value


# In[89]:


#update bridge in G 
for _, row in edges.iterrows():
    u, v, key = row['u'], row['v'], row['key']
    bridge_value = row['bridge']
    G[u][v][key]['bridge'] = bridge_value


# In[90]:


#update tunnel in G 
for _, row in edges.iterrows():
    u, v, key = row['u'], row['v'], row['key']
    tunnel_value = row['tunnel']
    G[u][v][key]['tunnel'] = tunnel_value


# In[91]:


# define a function to extract the road segments details from each shortest route
def extract_road_segments_details(row):
    if isinstance(row['shortest_route'], str):
        shortest_route_str=row['shortest_route']
        nodes_str=re.findall(r'\d+', shortest_route_str)
        nodes_int=[int(x) for x in nodes_str]
    else:
        nodes_int=row['shortest_route']
    road_segments = []
    for u, v in zip(nodes_int[:-1], nodes_int[1:]):
        # Check all edges between u and v
        edgs = G[u][v]
        # If there's only one edge, take it
        if len(edgs) == 1:
            edge_data = list(edgs.values())[0]
        else:
            # For multiple edges, select the one with the minimum 'length'
            edge_data = min(edgs.values(), key=lambda x: x.get('length', np.inf))
        edge_data['u'] = u
        edge_data['v'] = v
        road_segments.append((u, v, edge_data))
    
    
    edge_data_ls=[]
    free_travel_time_sec_ls=[]
    for segment in road_segments:
        u,v,edge_data=segment
        edge_data_ls.append(edge_data)
        free_travel_time_sec_ls.append(edge_data.get('travel_time', np.nan))
    free_travel_time_total_sec=sum(free_travel_time_sec_ls)
    return (free_travel_time_total_sec,  edge_data_ls)


# In[92]:


# application
data_merged_routes_diff500_chopped[['free_travel_time_total_sec', 
                                                               'edge_data_ls', 
                                                              ]]=data_merged_routes_diff500_chopped.apply(lambda row:
                                                                                                                          pd.Series(extract_road_segments_details(row)),
                                                                                                                          axis=1)


# In[94]:


# combine all segments into a list
shortest_routes_segs_ls=[]
for x in data_merged_routes_diff500_chopped['edge_data_ls'].values:
    shortest_routes_segs_ls.extend(x)


# In[95]:


# use chuncks to deal with memory error
chunk_size = 100000  # Adjust based on available memory
chunks = []

for i in range(0, len(shortest_routes_segs_ls), chunk_size):
    chunk = pd.DataFrame(shortest_routes_segs_ls[i:i + chunk_size])
    chunks.append(chunk)

shortest_routes_segs_df = pd.concat(chunks, ignore_index=True)


# In[96]:


# remove list object 
shortest_routes_segs_df=shortest_routes_segs_df.apply(remove_list_obj)


# In[97]:


shortest_routes_segs_df_group=shortest_routes_segs_df.drop_duplicates()


# ### Check the road segment types

# - The 'reversed' column in the edges GeoDataFrame typically indicates whether the direction of a road segment (edge) has been reversed relative to its default representation in the road network.
# - 'bridge' represents whether a segment is a bridge. True or yes means it is a bridge, otherwise, being false or NaN means it is not a bridge.
# - 'ref' contains reference identifiers for the segment.
# - 'tunnel' represents whether a road segment is a tunnel. NaN means the edge is not a tunnel.
# - 'access' describes access restrictions. If yes or NaN means no restrictions.
# - 'junction' means whether a road is a part of a junction/intersection. NaN means the edge is not part of a junction.

# To group edges based on their similarity for learning traffic patterns and performing route optimization, we should choose columns that are the most relevant attributes affecting traffic:
# - region: Represents spatial clustering. Useful for capturing regional traffic behavior differences.
# - highway: Traffic behavior differs significantly by road type.
# - lanes: The number of lanes directly affects traffic capacity and congestion levels.
# - speed_kph: Speed limits are crucial for understanding expected travel times and traffic flow.
# - access: Indicates access restrictions, which could affect traffic density (e.g., private, permissive).
# - junction: Important for identifying special road segments like roundabouts or complex intersections where traffic behavior differs.
# - bridge and tunnel: These attributes often indicate constrained segments where traffic might behave differently due to structural limitations.

# - secondary:  medium sized roads that connect smaller towns or districts. less traffic than primary but more than tertiary.
# - tertiary: smaller roads that connect villages or neighborhoods. moderate traffic.
# - primary: major roads connect larger towns or cities. significant traffic flow but less than motorway and trunks.
# - motorway_link: short connecting roads between motorways and other types of roads. Often used to enter or leave a motorway.
# - motorway: high speed roads with limited access. Designed for long-distance and high-volume traffic.
# - trunk: major roads, connecting cities or regions, but not motorway. Important for long-distance travel.
# - emergency_bay: dedicated areas on motorway for emergency.
# - living_street: roads primarily for pedestrain traffic, with restricted vehical access and with very low speed limit.
# - ['residential', 'tertiary']: Could indicate a road that transitions between residential and tertiary classifications.

# ### convert the existing trip level dataset into road segment level dataset

# In[ ]:


# add the 'free_travel_time_total_sec' and 'edge_data_ls' columns to the original trip level dataset
data_merged_routes_diff500_chopped_subcols=data_merged_routes_diff500_chopped[['free_travel_time_total_sec', 
                                                               'edge_data_ls', 'shortest_route']]
trips_segs_merge=pd.merge(data_merged_routes_diff500, data_merged_routes_diff500_chopped_subcols, on='shortest_route')


# #####  Distribute Trip Travel Time to Segments
# Since trip-level data provides total travel time, we can estimate segment-level travel times by distributing the trip-level travel time across the matched road segments propotionally based on their free flow information. Specifically, based on the given speed limit and length, calculate the free-flow travel time ratio of each segment over the whole route, and then multiply this ratio to the actual total trip time to infer the actual travel time for each segment.

# In[ ]:


def create_seg_level_df (row):
    total_free_travel_time=row['free_travel_time_total_sec']
    X=pd.DataFrame(row['edge_data_ls'])[[ 'lanes',  'highway', 
        'speed_kph', 'travel_time', 'region',  'u', 'v', 'junction',
       'bridge', 'tunnel', 'access']]
    X['inferred_travel_time_sec']=X['travel_time']/total_free_travel_time*row['trip_seconds']
    X[['Unnamed: 0', 'trip_start_timestamp', 'trip_end_timestamp', 
       'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wpgt', 'pres']]=row[['Unnamed: 0', 'trip_start_timestamp', 'trip_end_timestamp', 
       'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wpgt', 'pres']]
    return X


# In[ ]:


chunk_size = 100  # Adjust based on available memory

# Open the file in append mode and write chunks incrementally
for i in range(0, len(trips_segs_merge), chunk_size):
    chunk = trips_segs_merge.iloc[i:i + chunk_size]
    chunk_result = pd.concat(chunk.apply(create_seg_level_df, axis=1).to_list(), ignore_index=True)
    
    # Write chunk results to file
    chunk_result.to_csv(r"C:\Users\cclin\Downloads\segment_level_data.csv", mode='a', index=False, header=(i == 0));  

