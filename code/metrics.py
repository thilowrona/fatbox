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
        
        n0 = edge[1]
        n1 = edge[0]  
        
        G.edges[edge]['strike'] = strike_between_nodes_pix(G, n0, n1)
        
    return G


def compute_xy(G, scale_factor):
    for node in G:
        G.nodes[node]['x'] = G.nodes[node]['pos'][0]*scale_factor
        G.nodes[node]['y'] = G.nodes[node]['pos'][1]*scale_factor
    return G




# Strike in radius calculation
def distance_between_nodes_pix(G, n0, n1):
    y0, x0 = G.nodes[n0]['pos']
    y1, x1 = G.nodes[n1]['pos']     
    return math.sqrt((x0 -x1)**2+(y0-y1)**2)




def distance_between_nodes(G, n0, n1):
    x0 = G.nodes[n0]['x']
    y0 = G.nodes[n0]['y']
    x1 = G.nodes[n1]['x']
    y1 = G.nodes[n1]['y']     
    return math.sqrt((x0 -x1)**2+(y0-y1)**2)


def strike_between_nodes(G, n0, n1):
    x0 = G.nodes[n0]['x']
    y0 = G.nodes[n0]['y']
    x1 = G.nodes[n1]['x']
    y1 = G.nodes[n1]['y']
    
    if (x0-x1)<0:
        if (y0-y1)<0:
            strike = math.degrees(math.atan((x0-x1)/(y0-y1)))
        elif (y0-y1)>0:
            strike = math.degrees(math.atan((x0-x1)/(y0-y1))) + 180
        else:
            strike = 90
            
    if (x0-x1)>0:
        if (y0-y1)>0:
            strike = math.degrees(math.atan((x0-x1)/(y0-y1)))
        elif (y0-y1)<0:
            strike = math.degrees(math.atan((x0-x1)/(y0-y1))) + 180
        else:
            strike = 90
            
    if (x0-x1) == 0:
        if (y0-y1)<0:
            strike = 0
        elif (y0-y1)>0:
            strike = 180
            
    return strike
    
    

def strike_between_nodes_pix(G, n0, n1):
    x0 = G.nodes[n0]['pos'][0]
    y0 = G.nodes[n0]['pos'][1]
    x1 = G.nodes[n1]['pos'][0]
    y1 = G.nodes[n1]['pos'][1]
    
    if (x0-x1)<0:
        if (y0-y1)<0:
            strike = math.degrees(math.atan((x0-x1)/(y0-y1)))
        elif (y0-y1)>0:
            strike = math.degrees(math.atan((x0-x1)/(y0-y1))) + 180
        else:
            strike = 90
            
    if (x0-x1)>0:
        if (y0-y1)>0:
            strike = math.degrees(math.atan((x0-x1)/(y0-y1)))
        elif (y0-y1)<0:
            strike = math.degrees(math.atan((x0-x1)/(y0-y1))) + 180
        else:
            strike = 90
            
    if (x0-x1) == 0:
        if (y0-y1)<0:
            strike = 0
        elif (y0-y1)>0:
            strike = 180
        else:
            strike = 0
            
    return strike


def strike_between_nodes_xz(G, n0, n1):
    x0 = G.nodes[n0]['x']
    y0 = G.nodes[n0]['z']
    x1 = G.nodes[n1]['x']
    y1 = G.nodes[n1]['z']

    return math.degrees(math.atan((x0-x1)/(y0-y1)))


        
    
def nodes_of_max_dist(G, nodes):
    
    if len(nodes) < 2:
        print('Only 2 nodes in neighborhood')
        return
    
    threshold = 0
    for n0 in nodes:
        for n1 in nodes:
            d = distance_between_nodes_pix(G, n0, n1)
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
        
            if distance_between_nodes_pix(G, node, other) < radius:
            
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
def max_value_edges(G, attribute):    
    values = np.zeros((len(G.edges)))
    for n, edge in enumerate(G.edges()):              
        values[n] = G.edges[edge][attribute]            
    return np.max(values)



def min_value_edges(G, attribute):    
    values = np.zeros((len(G.edges)))
    for n, edge in enumerate(G.edges()):              
        values[n] = G.edges[edge][attribute]            
    return np.min(values)



def compute_edge_length(G):
    for edge in G.edges:
        G.edges[edge]['length'] = distance_between_nodes_pix(G, edge[0], edge[1])
    return G



def total_length(G):    
    G = compute_edge_length(G)    
    length = 0
    for edge in G.edges:
        length = length + G.edges[edge]['length']
    return length



def select_components(G, components):
    H = G.copy()
    if type(components) != list:
        selected_nodes = [n[0] for n in H.nodes(data=True) if n[1]['component'] == components]
    else:
        selected_nodes = [n[0] for n in H.nodes(data=True) if n[1]['component'] in components]      
    H = H.subgraph(selected_nodes)
    return H

def component_lengths(G):
    values = np.zeros((number_of_components(G)))      
    for m, cc in enumerate(sorted(nx.connected_components(G))):     
        G_sub = G.copy()        
        G_sub = G_sub.subgraph(cc)    
        length = 0
        for edge in G_sub.edges:
            length = length + G_sub.edges[edge]['length']    
        values[m] = length
    return values












def fault_lengths(G):    
    
    faults = return_faults(G)
    
    lengths = np.zeros(len(faults))
    
    for n, fault in enumerate(faults):
        nodes = [node for node in G if G.nodes[node]==fault]      
        G_sub = G.subgraph(nodes)        
        lengths[n] = total_length(G_sub)    
    return lengths
    
    
  
    
    
    
import math

def dip(x0,z0,x1,z1):
    if (x0-x1) == 0:
        value = 90
    else:
        value = math.degrees(math.atan((z0-z1)/(x0-x1)))
        if value == -0:
            value = 0
    
    return value


def edge_dip(G):
    for edge in G.edges():
        G.edges[edge]['dip'] = dip(G.nodes[edge[0]]['x'], G.nodes[edge[0]]['z'],
                                    G.nodes[edge[1]]['x'], G.nodes[edge[1]]['z']
                                    )
    return G 
    


def calculate_dip(G):
    
    for edge in G.edges():
        G.edges[edge]['dip'] = dip(G.nodes[edge[0]]['x'], G.nodes[edge[0]]['z'],
                                    G.nodes[edge[1]]['x'], G.nodes[edge[1]]['z']
                                    )
    
    for node in G.nodes():
        lst = []
        for edge in G.edges(node):
            lst.append(G.edges[edge]['dip'])
        G.nodes[node]['dip'] = sum(lst) / len(lst)
        
    return G





## COMPONENTS
def number_of_components(G):
    return len(sorted(nx.connected_components(G)))

def number_of_faults(G):
    return len(return_faults(G))




def max_value_components_nodes(G, attribute):    
    max_values = np.zeros((number_of_components(G)))    
    for m, cc in enumerate(sorted(nx.connected_components(G))):        
        values = np.zeros((len(cc)))        
        for n, node in enumerate(cc):            
            values[n] = G.nodes[node][attribute]            
        max_values[m] = np.max(values)        
    return max_values




def max_value_faults_nodes(G, attribute):
    faults = return_faults(G)
    max_values = np.zeros((len(faults))) 
    for n, fault in enumerate(faults):
        nodes = [node for node in G if G.nodes[node]['fault']==fault]
        values = np.zeros((len(nodes))) 
        for m, node in enumerate(nodes):            
            values[m] = G.nodes[node][attribute]            
        max_values[n] = np.max(values)   
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
    (x_max, y_max) = image.shape   
    for node in G:
        y,x = G.nodes[node]['pos']
        if x >= x_max or y>=y_max:
            G.nodes[node][name] = float('nan')        
        else:
            G.nodes[node][name] = image[int(x),int(y)]
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




def sum_value_nodes(G, attribute):    
    values = np.zeros((len(G.nodes)))
    for n, node in enumerate(G):              
        values[n] = G.nodes[node][attribute]            
    return np.sum(values)


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


def get_limits(G):
    x_min = min_value_nodes(G, 'x')
    x_max = max_value_nodes(G, 'x')
    y_min = min_value_nodes(G, 'x')
    y_max = max_value_nodes(G, 'x')
    return x_min, x_max, y_min, y_max





def return_components(G):
    return sorted([G.nodes[next(iter(c))]['component'] for c in sorted(nx.connected_components(G))])


def return_faults(G):
    faults = set()
    for node in G:
        faults.add(G.nodes[node]['fault'])
    return list(faults)
    
    









def calculate_crop(G, edge):
  
  x = np.zeros((len(G.nodes)))
  z = np.zeros((len(G.nodes)))
  
  for n, node in enumerate(G):              
      (x[n], z[n]) = G.nodes[node]['pos']            
     
  x_max = np.max(x)
  x_min = np.min(x)
  dx    = x_max-x_min
  
  x_max = int(x_max + 0.1 * dx)
  x_min = int(x_min - 0.1 * dx)
  
  
  z_max = np.max(z)
  z_min = np.min(z)
  dz    = z_max-z_min
  
  z_max = int(z_max + edge * dz)
  z_min = int(z_min - edge * dz)
  
  if z_min < 0:
      z_min = 0

  return (x_min, x_max), (z_min, z_max)



























def calculate_edge_mid_point(G, edge):
    
    # point locations
    x = int((G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0])/2)
    y = int((G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1])/2)

    return (x, y)




def calculate_mid_points(G):
    
    H = nx.Graph()   
    
    for edge in G.edges:
    
        node = (edge)
        H.add_node(node)
        H.nodes[node]['pos'] = calculate_edge_mid_point(G, edge)
        H.nodes[node]['component']  = -1
        H.nodes[node]['edge'] = (edge)
    
    return H








# calculate direction

# def calculate_edge_direction(G, normalize=True):

#     for edge in G.edges:
               
#         # calculate vector coordinates              
#         dx = G.nodes[edge[0]]['pos'][0] - G.nodes[edge[1]]['pos'][0]
#         dy = G.nodes[edge[0]]['pos'][1] - G.nodes[edge[1]]['pos'][1]
            
#         # normalize coordinates
#         if normalize:
#             dx = dx/(abs(dx)+abs(dy))
#             dy = dy/(abs(dx)+abs(dy))

#         G.edges[edge]['direction'] = (dx, dy)

#     return G









def calculate_direction(G, cutoff, normalize=True):

    for node in G.nodes:
            
        length = nx.single_source_shortest_path_length(G, node, cutoff=cutoff)
        keys = [keys for keys,values in length.items() if values == max(length.values())]
        
        if len(keys) >  2:
            (node_0, node_1) = keys[:2]

        if len(keys) == 2:
            (node_0, node_1) = keys
        if len(keys) == 1:
            node_0 = keys[0]
            
            length = nx.single_source_shortest_path_length(G, node, cutoff=cutoff-1)
            keys = [keys for keys,values in length.items() if values == max(length.values())]
            
            node_1 = keys[0]

    
        # extrac position
        pt_0 = G.nodes[node_0]['pos']
        pt_1 = G.nodes[node_1]['pos']  
                
        # calculate vector              
        dx = pt_0[0] - pt_1[0]
        dy = pt_0[1] - pt_1[1]
            
        # write to graph
        G.nodes[node]['dx'] = dx
        G.nodes[node]['dy'] = dy
        
    return G










def calculate_pickup_points(G, factor):

    H = nx.Graph() 
    
    for node in G.nodes:
        
        (x, y) = G.nodes[node]['pos']
        
        dx = G.nodes[node]['dx']
        dy = G.nodes[node]['dy']
        
        dx = factor * dx
        dy = factor * dy
    
        x_p = int(x - dy)
        y_p = int(y + dx)
    
        x_n = int(x + dy)
        y_n = int(y - dx)
    
        node_mid = (node, 0)
        H.add_node(node_mid)
        H.nodes[node_mid]['pos'] = (x, y)
        H.nodes[node_mid]['component']  = -1
    
        node_p = (node, 1)
        H.add_node(node_p)
        H.nodes[node_p]['pos'] = (x_p, y_p)
        H.nodes[node_p]['component']  = -2
    
        node_n = (node, 2)
        H.add_node(node_n)
        H.nodes[node_n]['pos'] = (x_n, y_n)
        H.nodes[node_n]['component']  = -3
    
    
        H.add_edge(node_n, node_p)
                
    return H





def filter_pickup_points(G, H):
    
    for node in G:           
        
        if H.nodes[(node, 1)]['pos'][1] < 0 or H.nodes[(node, 2)]['pos'][1] < 0:
            
            H.nodes[(node,0)]['v_x'] = 0
            H.nodes[(node,0)]['v_z'] = 0
    
            H.nodes[(node,1)]['v_x'] = 0
            H.nodes[(node,1)]['v_z'] = 0
    
            H.nodes[(node,2)]['v_x'] = 0
            H.nodes[(node,2)]['v_z'] = 0
            
    return H



def calculate_slip_rate(G, H, dim):
    if dim == 2:
        for node in H.nodes:
            if node[1] == 0:    # centre point
                
                if H.nodes[(node[0], 1)]['v_x'] == 0 or H.nodes[(node[0], 2)]['v_x'] == 0:
                    G.nodes[node[0]]['slip_rate_x'] =  0
                    G.nodes[node[0]]['slip_rate_z'] =  0
                    G.nodes[node[0]]['slip_rate']   =  0
                else:            
                    G.nodes[node[0]]['slip_rate_x'] =  abs((H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x']))
                    G.nodes[node[0]]['slip_rate_z'] =  abs((H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z']))
                    G.nodes[node[0]]['slip_rate']   =  math.sqrt(G.nodes[node[0]]['slip_rate_x']**2 + G.nodes[node[0]]['slip_rate_z']**2)
    if dim == 3:
        for node in H.nodes:
            if node[1] == 0:    # centre point   
                # Outside of the box
                if H.nodes[(node[0], 1)]['v_x'] == 0 or H.nodes[(node[0], 2)]['v_x'] == 0:
                    G.nodes[node[0]]['slip_rate_x'] =  0
                    G.nodes[node[0]]['slip_rate_y'] =  0
                    G.nodes[node[0]]['slip_rate_z'] =  0
                    G.nodes[node[0]]['slip_rate']   =  0        
                # Inside the box
                else:            
                    G.nodes[node[0]]['slip_rate_x'] =  abs((H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x']))
                    G.nodes[node[0]]['slip_rate_y'] =  abs((H.nodes[(node[0], 1)]['v_y'] - H.nodes[(node[0], 2)]['v_y']))
                    G.nodes[node[0]]['slip_rate_z'] =  abs((H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z']))
                    G.nodes[node[0]]['slip_rate']   =  math.sqrt(G.nodes[node[0]]['slip_rate_x']**2 + 
                                                                 G.nodes[node[0]]['slip_rate_y']**2 + 
                                                                 G.nodes[node[0]]['slip_rate_z']**2)        
        
    return G





def calculate_slip(G, H, dt, dim):
    if dim == 2:
        for node in H.nodes:
            if node[1] == 0:
                
                if H.nodes[(node[0], 1)]['v_x'] == 0 or H.nodes[(node[0], 2)]['v_x'] == 0:
                    G.nodes[node[0]]['slip_x'] =  0
                    G.nodes[node[0]]['slip_z'] =  0
                    G.nodes[node[0]]['slip']   =  0
                else:            
                    G.nodes[node[0]]['slip_x'] =  abs((H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x']))*dt
                    G.nodes[node[0]]['slip_z'] =  abs((H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z']))*dt
                    G.nodes[node[0]]['slip']   =  math.sqrt(G.nodes[node[0]]['slip_x']**2 + G.nodes[node[0]]['slip_z']**2)
                                                            
    if dim == 3:
        for node in H.nodes:
            if node[1] == 0:
                
                if H.nodes[(node[0], 1)]['v_x'] == 0 or H.nodes[(node[0], 2)]['v_x'] == 0:
                    G.nodes[node[0]]['slip_x'] =  0
                    G.nodes[node[0]]['slip_y'] =  0
                    G.nodes[node[0]]['slip_z'] =  0
                    G.nodes[node[0]]['slip']   =  0
                else:            
                    G.nodes[node[0]]['slip_x'] =  abs((H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x']))*dt
                    G.nodes[node[0]]['slip_y'] =  abs((H.nodes[(node[0], 1)]['v_y'] - H.nodes[(node[0], 2)]['v_y']))*dt
                    G.nodes[node[0]]['slip_z'] =  abs((H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z']))*dt
                    G.nodes[node[0]]['slip']   =  math.sqrt(G.nodes[node[0]]['slip_x']**2 + 
                                                            G.nodes[node[0]]['slip_y']**2 + 
                                                            G.nodes[node[0]]['slip_z']**2)
    return G






def split_graph_by_polarity(G):
    G_0 = G.copy()
    G_1 = G.copy()
    for node in G.nodes:
        if G.nodes[node]['polarity'] == 0:
            G_1.remove_node(node)
        else:
            G_0.remove_node(node)
    return G_0, G_1



def get_fault_labels(G):
    labels=set()
    for node in G:
        labels.add(G.nodes[node]['fault'])
    return sorted(list(labels))   
