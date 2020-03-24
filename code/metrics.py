import math
import random
import networkx as nx
import numpy as np

from edits import *


## EDGES
# Count edges
def count_edges(G):
    for node in G:
        G.nodes[node]['edges'] = len(G.edges(node))
    return G



# Compute strikes
def compute_strikes(G):
    
    for n, edge in enumerate(G.edges):
        
        n0 = edge[0]
        n1 = edge[1]  
        
        G.edges[edge]['strike'] = strike_between_nodes(G, n0, n1)
        
    return G






# Strike in radius calculation
def distance_between_nodes(G, n0, n1):
    y0, x0 = G.nodes[n0]['pos']
    y1, x1 = G.nodes[n1]['pos']     
    return math.sqrt((x0 -x1)**2+(y0-y1)**2)



def distance_between_nodes2(G, n0, n1):
    x0 = G.nodes[n0]['x']
    y0 = G.nodes[n0]['y']
    x1 = G.nodes[n1]['x']
    y1 = G.nodes[n1]['y']     
    return math.sqrt((x0 -x1)**2+(y0-y1)**2)


def strike_between_nodes(G, n0, n1):
    x0, y0 = G.nodes[n0]['pos']
    x1, y1 = G.nodes[n1]['pos']

    return math.degrees(math.atan((x0-x1)/(y0-y1)))
        
    
def nodes_of_max_dist(G, nodes):
    
    if len(nodes) < 2:
        print('Only 2 nodes in neighborhood')
        return
    
    threshold = 0
    for n0 in nodes:
        for n1 in nodes:
            d = distance_between_nodes(G, n0, n1)
            if d > threshold:
                threshold = d
                pair = (n0, n1)
    return pair    
        


def calculate_strikes_in_radius(G, radius = 10):
    
    N = nx.number_of_nodes(G)
    
    for node in G:
        
        G.nodes[node]['strike'] = float("nan")
    
        nodes_within_radius = []
        
        for other in G:
        
            if distance_between_nodes(G, node, other) < radius:
            
                nodes_within_radius.append(other)
        
        if len(nodes_within_radius) > 0:
                        
            (nA, nB) = nodes_of_max_dist(G, nodes_within_radius)
        
            G.nodes[node]['strike'] = strike_between_nodes(G, nA, nB)
           
        print(str(node) + ' of ' + str(N))
        
    return G






# Strike in neighborhood calculation
def new_nodes_of_max_dist(G, nodes):
    threshold = 0
    for n0 in nodes:
        for n1 in nodes:
            d = nx.shortest_path_length(G, source = n0, target = n1, weight='length')
            
            if d > threshold:
                threshold = d
                pair = (n0, n1)
    return pair   



def calculate_strikes_in_neighborhood(G, neighbors = 3):    
    
    for cc in sorted(nx.connected_components(G)):
    
        for node in cc:
            
            G_cc = G.subgraph(cc)
            
            dict_neighbors = nx.single_source_shortest_path_length(G_cc, node, cutoff=neighbors)
            
            list_of_nodes = [x for x in dict_neighbors]
                        
            (nA, nB) = new_nodes_of_max_dist(G, list_of_nodes)
        
            G.nodes[node]['strike'] = strike_between_nodes(G, nA, nB)
        
    return G



    


## EDGES
def compute_edge_length(G):
    for edge in G.edges:
        G.edges[edge]['length'] = distance_between_nodes2(G, edge[0], edge[1])
    return G

def total_length(G):    
    G = compute_edge_length(G)    
    length = 0
    for edge in G.edges:
        length = length + G.edges[edge]['length']
    return length




def fault_lengths(G):    
    noc = number_of_components(G)    
    lengths = np.zeros(noc)    
    for n in range(noc):    
        lengths[n] = total_length(select_component(G, component=n))    
    return lengths    
    
    
    
    
    
    





## COMPONENTS
def number_of_components(G):
    return len(sorted(nx.connected_components(G)))



def max_value_components(G, attribute):    
    max_values = np.zeros((number_of_components(G)))    
    for m, cc in enumerate(sorted(nx.connected_components(G))):        
        values = np.zeros((len(cc)))        
        for n, node in enumerate(cc):            
            values[n] = G.nodes[node][attribute]            
        max_values[m] = np.max(values)        
    return max_values




def mean_x_components(G):    
    mean_values = np.zeros((number_of_components(G)))    
    for m, cc in enumerate(sorted(nx.connected_components(G))):        
        values = np.zeros((len(cc)))        
        for n, node in enumerate(cc):            
            values[n] = G.nodes[node]['pos'][0]           
        mean_values[m] = np.mean(values)        
    return mean_values


def mean_y_components(G):    
    mean_values = np.zeros((number_of_components(G)))    
    for m, cc in enumerate(sorted(nx.connected_components(G))):        
        values = np.zeros((len(cc)))        
        for n, node in enumerate(cc):            
            values[n] = G.nodes[node]['pos'][1]           
        mean_values[m] = np.mean(values)        
    return mean_values






## EXTRACT VALUES FROM IMAGE
def extract_attribute(G, image, name):    
    for node in G:
        y,x = G.nodes[node]['pos']    
        G.nodes[node][name] = image[x,y]
    return G
    
    







## EXTRACT STRAIN PROFILE FROM NETWORK
def strain_profile(G, attribute):
    
    y = np.zeros(len(G.nodes))
    strain_rate = np.zeros(len(G.nodes))
    
    
    for n, node in enumerate(G):        
        y[n] = G.nodes[node]['pos'][1]
        strain_rate[n] = G.nodes[node][attribute]
    
    
    y_min = np.min(y)
    y_max = np.max(y)    
    y_resample = np.arange(y_min, y_max)
    
            
    def find_max_y(y, y_n, attribute):        
        index = np.where(y == y_n)[0]        
        return np.max(attribute[index])
           
    
    strain_rate_resample = np.zeros_like(y_resample)
    for n, y_n in enumerate(y_resample):        
        strain_rate_resample[n] = find_max_y(y, y_n, strain_rate)
    
    
    return y_resample, strain_rate_resample









## CALCULATE NETWORK ATTRIBUTE
def max_value_nodes(G, attribute):    
    values = np.zeros((len(G.nodes)))
    for n, node in enumerate(G):              
        values[n] = G.nodes[node][attribute]            
    return np.max(values)



def min_value_nodes(G, attribute):    
    values = np.zeros((len(G.nodes)))
    for n, node in enumerate(G):              
        values[n] = G.nodes[node][attribute]            
    return np.min(values)




def mean_value_nodes(G, attribute):    
    values = np.zeros((len(G.nodes)))
    for n, node in enumerate(G):              
        values[n] = G.nodes[node][attribute]            
    return np.mean(values)





def mean_value_edges(G, attribute):    
    values = np.zeros((len(G.edges)))
    for n, edge in enumerate(G.edges):              
        values[n] = G.edges[edge][attribute]            
    return np.mean(values)





def average_x(G):    
    values = np.zeros((len(G.nodes)))
    for n, node in enumerate(G):              
        values[n] = G.nodes[node]['pos'][1]            
    return np.average(values)






def average_y(G):    
    values = np.zeros((len(G.nodes)))
    for n, node in enumerate(G):              
        values[n] = G.nodes[node]['pos'][0]            
    return np.average(values)













  

