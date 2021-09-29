# Packages
import math
import networkx as nx
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d



#==============================================================================
# This file contains a series of function to compute metrics of the fault 
# network (graph). This includes functions for: 
# (1) nodes
# (2) edges
# (3) components (i.e. connected nodes)
# (4) faults (i.e. one or more components)
# (5) the whole network 
#==============================================================================



#******************************************************************************
# (1) NODE METRICS
# A couple of functions to calculate node attributes
#******************************************************************************

def compute_node_values(G, attribute, mode):
    """ Calculate common node value, e.g. min, max, mean
    
    Parameters
    ----------
    G : nx.graph
        Graph contraining nodes
    attribute : string
        Node attribute used for calculation
    mode: string
        Type of value that's calculated. Options: min, max, mean, range, sum'
        
    Returns
    -------  
    value
        Float
    """

    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    assert (mode in ['min', 'max', 'mean', 'range', 'sum']), 'Node 0 is not in G'    

    # Get values
    values = np.array([G.nodes[node][attribute] for node in G])
    
    # Compute value
    if mode == 'min':
        return np.nanmin(values)
    
    elif mode == 'max':
        return np.nanmax(values)
    
    elif mode == 'mean':
        return np.nanmean(values)
    
    elif mode == 'range':
        return np.nanmax(values)-np.nanmin(values)

    elif mode == 'sum':
        return np.nansum(values)




def get_limits_xy(G):
    """ Calculate limits (x,y) of graph
    
    Parameters
    ----------
    G : nx.graph
        Graph contraining nodes
        
    Returns
    -------  
    x_min
        Float
    x_max
        Float
    y_min
        Float
    y_max
        Float
    """

    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    
    # Calculation
    x_min = compute_node_values(G, 'x', 'min')
    x_max = compute_node_values(G, 'x', 'max')
    y_min = compute_node_values(G, 'y', 'min')
    y_max = compute_node_values(G, 'y', 'max')
    
    return x_min, x_max, y_min, y_max




def distance_between_nodes(G, n0, n1, mode='pos'):
    """ Calculate distance (euclidean) between nodes
    
    Parameters
    ----------
    G : nx.graph
        Graph contraining nodes
    n0 : node 0
    n1 : node 1
    mode : Coordinates used to calculate distance (Default: 'pos', others: 'xy')
    
    Returns
    -------  
    distance
        Float
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    assert (n0 in G.nodes), 'Node 0 is not in G'
    assert (n1 in G.nodes), 'Node 1 is not in G'
    assert mode == 'pos' or mode == 'xy', "Invalid mode! Mode is neither 'pos' nor 'xy'"
    
    # Get coordinates
    if mode == 'pos':
        y0, x0   = G.nodes[n0]['pos']
        y1, x1   = G.nodes[n1]['pos']
        
        
    if mode == 'xy':
        x0 = G.nodes[n0]['x']
        y0 = G.nodes[n0]['y']
        x1 = G.nodes[n1]['x']
        y1 = G.nodes[n1]['y']       
        
    # Return euclidean distance        
    return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)



# Calculate polarity
def calculate_polarity(G):
    for cc in nx.connected_components(G):
        maximum = 0
        minimum = 1e6
        for node in cc:
            y_value = G.nodes[node]['y']
            
            if y_value < minimum:
                min_y = node
                minimum = y_value
                
            if y_value > maximum:
                max_y = node
                maximum = y_value
                
        if G.nodes[min_y]['x'] <= G.nodes[max_y]['x']:
            for node in cc:
                G.nodes[node]['polarity'] = 1
        else:
            for node in cc:
                G.nodes[node]['polarity'] =-1    
    return G




def compute_curvature(G, non, sigma):

  for node in G:
      
    neighbors = nx.single_source_shortest_path_length(G, node, cutoff=non)

    # Compute distances
    dm = np.zeros((len(neighbors), len(neighbors)))
    for n, neighbor in enumerate(neighbors):
      for m, another in enumerate(neighbors):
        dm[n,m] = nx.shortest_path_length(G, neighbor, another)

    maximum_distance = np.max(dm)

    # Find start and end node
    for n, neighbor in enumerate(neighbors):
      for m, another in enumerate(neighbors):
        if dm[n,m] == maximum_distance:
          start = neighbor
          end = another
          break

    # Compute path
    path = nx.shortest_path(G, start, end) 

    #print(dm)
    #print(maximum)
    #print(path)


    x = np.zeros(len(path))
    y = np.zeros(len(path))

    for n, pode in enumerate(path):
      x[n] = G.nodes[pode]['pos'][0]
      y[n] = G.nodes[pode]['pos'][1]


    ysmoothed = gaussian_filter1d(y, sigma=sigma)

    dx = np.gradient(x, x)  # first derivatives
    dy = np.gradient(y, x)

    d2x = np.gradient(dx, x)  # second derivatives
    d2y = np.gradient(dy, x)

    cur = np.abs(d2y) / (np.sqrt(1 + dy ** 2)) ** 1.5  # curvature

    G.nodes[node]['min_curv'] = np.nanmin(cur)
    G.nodes[node]['mean_curv'] = np.nanmean(cur)
    G.nodes[node]['max_curv'] = np.nanmax(cur)

  return G


#******************************************************************************
# (2) EDGE METRICS
# A couple of functions to calculate edge attributes
#******************************************************************************

def count_edges(G):
    """ Count the number of edges for each node
    
    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    G : nx.graph
        Graph
    """

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
        
    # Calculation
    for node in G:
        G.nodes[node]['edges'] = len(G.edges(node))
        
    return G



def compute_edge_length(G):
    """ Count the length of each edge
    
    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    G : nx.graph
        Graph with 'length' edge attribute
    """  
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
        
    # Calculation   
    for edge in G.edges:
        G.edges[edge]['length'] = distance_between_nodes(G, edge[0], edge[1])
        
    return G



def compute_edge_values(G, attribute, mode):
    """ Calculate common edge value, e.g. min, max, mean
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    attribute : string
        Node attribute used for calculation
    mode: string
        Type of value that's calculated. Options: min, max, mean, range, sum'
        
    Returns
    -------  
    value
        Float
    """

    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    assert (mode in ['min', 'max', 'mean', 'range', 'sum']), 'Node 0 is not in G'    

    # Get values
    values = np.array([G.edges[edge][attribute] for edge in G.edges])
    
    # Compute value
    if mode == 'min':
        return np.min(values)
    
    elif mode == 'max':
        return np.max(values)
    
    elif mode == 'mean':
        return np.mean(values)
    
    elif mode == 'range':
        return np.max(values)-np.min(values)

    elif mode == 'sum':
        return np.sum(values)


def strike(x1, y1, x2, y2):
  if (x2-x1)<0:
    strike = math.degrees(math.atan2((x2-x1),(y2-y1))) + 360
  else:
    strike = math.degrees(math.atan2((x2-x1),(y2-y1)))
  
  #Scale to [0, 180]
  if strike<=180:
    return strike
  else:
    return strike - 180


def calculate_strike(G, non):
    """ Compute dip of fault network
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    non: int
        Number of neighbors
        
    Returns
    -------  
    G : nx.graph
        Graph containing edges with 'strike' attribute
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    for node in G:
    
        
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=non)
        
        
        neighbors = sorted(neighbors.items())
        
        first = neighbors[0][0]
        last = neighbors[-1][0]
        
        # print(node)
        # print(neighbors)
        # print(first, last)

        
        
        x1 = G.nodes[first]['pos'][0]
        y1 = G.nodes[first]['pos'][1]
           
        x2 = G.nodes[last]['pos'][0]
        y2 = G.nodes[last]['pos'][1]
          
        
        G.nodes[node]['strike'] = strike(x1, y1, x2, y2)

    return G




def dip(x0, z0, x1, z1):
    """ Compute dip between two points: (x0, z0) (x1, z1)
    
    Parameters
    ----------
    x0 : float
        X-coordinate of point 0
    x1 : float
        X-coordinate of point 1
    z0 : float
        Z-coordinate of point 0
    z1 : float
        Z-coordinate of point 1
        
    Returns
    -------  
    value : float
        Dip between points
    """

    # Assertions

    
    # Calculation
    if (x0 - x1) == 0:
        value = 90
    else:
        value = math.degrees(math.atan((z0 - z1)/(x0 - x1)))
        if value == -0:
            value = 0

    return value






def calculate_dip(G, non):
    """ Compute dip of fault network
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    non: int
        Number of neighbors
        
    Returns
    -------  
    G : nx.graph
        Graph containing edges with 'strike' attribute
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    for node in G:
    
        
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=non)
        
        
        neighbors = sorted(neighbors.items())
        
        first = neighbors[0][0]
        last = neighbors[-1][0]
        
        # print(node)
        # print(neighbors)
        # print(first, last)

        
        
        x1 = G.nodes[first]['pos'][0]
        y1 = G.nodes[first]['pos'][1]
           
        x2 = G.nodes[last]['pos'][0]
        y2 = G.nodes[last]['pos'][1]
          
        
        G.nodes[node]['dip'] = dip(x1, y1, x2, y2)

    return G




def calculate_diff_dip(G, non):
    """ Compute dip difference between nodes of fault network
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    non: int
        Number of neighbors
        
    Returns
    -------  
    G : nx.graph
        Graph containing edges with 'strike' attribute
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'

    
    for node in G:
    
        neighbors = nx.single_source_shortest_path_length(G, node, cutoff=non)
        dips = [G.nodes[node]['dip'] for node in neighbors.keys()]
        G.nodes[node]['max_diff'] = np.max(np.diff(dips))
    
    return G







def strike_between_nodes(G, n0, n1):
    """ Compute strike between two nodes
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    n0: int
        Node 0
    n1 : int
        Node 1
        
    Returns
    -------  
    value : float
        Strike between two nodes
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    # Calculation    
    x0 = G.nodes[n0]['pos'][0]
    x1 = G.nodes[n1]['pos'][0]
    y0 = G.nodes[n0]['pos'][1]
    y1 = G.nodes[n1]['pos'][1]
    
    if (x1-x0)<0:
        strike = math.degrees(math.atan2((x1-x0),(y1-y0))) + 360
    else:
        strike = math.degrees(math.atan2((x1-x0),(y1-y0)))
    
    #Scale to [0, 180]
    if strike<=180:
        return strike
    else:
        return strike - 180





def nodes_of_max_dist(G, nodes):
    """ Compute pair of nodes which are furthest apart
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    nodes: list
        List of nodes
        
    Returns
    -------  
    pair : tuple
        Tuple of nodes
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    # Calculation 
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




def calculate_strikes_in_radius(G, radius=10):
    """ Compute pair of nodes which are furthest apart
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    radius: int, float
        Radius to calculate strike in
        
    Returns
    -------  
    G : nx.graph
        Graph containing edges
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    # Calculation 
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




def nodes_of_max_dist_2(G, nodes):
    """ Compute pair of nodes which are furthest apart
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    nodes: list
        List of nodes
        
    Returns
    -------  
    pair : tuple
        Tuple of nodes
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    # Calculation   
    threshold = 0
    for n0 in nodes:
        for n1 in nodes:
            d = nx.shortest_path_length(G, source=n0, target=n1, weight='length')
            if d > threshold:
                threshold = d
                pair = (n0, n1)
    return pair




def calculate_strikes_in_neighborhood(G, neighbors=3):
    """ Compute pair of nodes in neighborhood
    
    Parameters
    ----------
    G : nx.graph
        Graph containing edges
    neighbors: int
        Number of neighbors to include in calculation
        
    Returns
    -------  
    G : nx.graph
        Graph containing edges
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    # Calculation
    for cc in sorted(nx.connected_components(G)):
        for node in cc:

            G_cc = G.subgraph(cc)

            dict_neighbors = nx.single_source_shortest_path_length(G_cc, node, cutoff=neighbors)

            list_of_nodes = [x for x in dict_neighbors]

            (nA, nB) = nodes_of_max_dist_2(G, list_of_nodes)

            G.nodes[node]['strike'] = strike_between_nodes(G, nA, nB)

    return G









#******************************************************************************
# (3) COMPONENT METRICS
# A couple of functions to calculate attributes of components
#******************************************************************************

def get_component_labels(G):
    """ Get component labels of graph G
    
    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    list: int
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    
    return sorted([G.nodes[next(iter(cc))]['component'] for cc in nx.connected_components(G)])




def number_of_components(G):
    """ Count the number of components
    
    Parameters
    ----------
    G : nx.graph
        Graph
    
    Returns
    -------  
    value: int
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"
    
    return len(nx.connected_components(G))




def compute_component_node_values(G, attribute, mode):
    """ Calculate common comonent value, e.g. min, max, mean
    
    Parameters
    ----------
    G : nx.graph
        Graph contraining nodes
    attribute : string
        Node attribute used for calculation
    mode: string
        Type of value that's calculated. Options: min, max, mean, range, sum'
        
    Returns
    -------  
    array
        Float
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'     
    assert (mode in ['min', 'max', 'mean', 'range', 'sum']), 'Node 0 is not in G'    

    # Calculation
    components = sorted(nx.connected_components(G))
    values = np.zeros((len(components)))
    
    # Loop through components
    for m, cc in enumerate(components):
        node_values = np.zeros((len(cc)))
        for n, node in enumerate(cc):
            node_values[n] = G.nodes[node][attribute]
                          
            
        # Compute fault values
        if mode == 'min':
            values[n] = np.min(node_values)
        
        elif mode == 'max':
            values[n] = np.max(node_values)
        
        elif mode == 'mean':
            values[n] = np.mean(node_values)
        
        elif mode == 'range':
            values[n] = np.max(node_values)-np.min(node_values)
    
        elif mode == 'sum':
            values[n] = np.sum(node_values)         
            
    return values




def component_lengths(G):
    """ Calculate lengths of components
    
    Parameters
    ----------
    G : nx.graph
        Graph contraining nodes
        
    Returns
    -------  
    array
        Float
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'     

    # Calculation
    values = np.zeros((number_of_components(G)))
    for m, cc in enumerate(sorted(nx.connected_components(G))):
        G_sub = G.copy()
        G_sub = G_sub.subgraph(cc)
        length = 0
        for edge in G_sub.edges:
            length = length + G_sub.edges[edge]['length']
        values[m] = length
    return values








#******************************************************************************
# (4) FAULT METRICS
# A couple of functions to calculate fault attributes
#******************************************************************************

def get_fault_labels(G):
    """ Get fault labels (sorted list)
    
    Parameters
    ----------
    G : nx.graph
        Fault network
        
    Returns
    -------  
    list
        List (sorted)
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph' 
    
    # Collect labels
    labels = set()
    for node in G:
        labels.add(G.nodes[node]['fault'])
        
    return sorted(list(labels))





def number_of_faults(G):
    """ Calculate the number of faults in G
    
    Parameters
    ----------
    G : nx.graph
        Fault network
        
    Returns
    -------  
    value
        Int
    """

    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    
    
    return len(get_fault_labels(G))



def get_fault(G, n):
    """ Return fault n as subgraph of G
    
    Parameters
    ----------
    G : nx.graph
        Fault network
        
    n : int
        Fault number
        
    Returns
    -------  
    G : nx.graph
        Subgraph of fault n
    """

    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph' 
        
    return G.subgraph([node for node in G if G.nodes[node]['fault'] == n])



def fault_lengths(G):
    """ Calculate fault lengths
    
    Parameters
    ----------
    G : nx.graph
        Fault network
        
    Returns
    -------  
    array
        Int
    """

    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    

    # Calculation
    labels = get_fault_labels(G)
    lengths = np.zeros(len(labels))

    for n, label in enumerate(labels):
        fault = get_fault(G, label)
        lengths[n] = total_length(fault)
        
    return lengths



def compute_fault_values(G, attribute, mode):
    """ Calculate common fault value, e.g. min, max, mean
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    attribute : string
        Node attribute used for calculation
    mode: string
        Type of value that's calculated. Options: min, max, mean, range, sum'
        
    Returns
    -------  
    value
        Float
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'     
    assert (mode in ['min', 'max', 'mean', 'range', 'sum']), 'Node 0 is not in G'    

    # Calculation
    faults = get_fault_labels(G)
    values = np.zeros((len(faults)))
    
    # Loop through faults
    for n, fault in enumerate(faults):
        nodes = [node for node in G if G.nodes[node]['fault'] == fault]
        node_values = np.zeros((len(nodes)))
        for m, node in enumerate(nodes):
            node_values[m] = G.nodes[node][attribute]                        
            
        # Compute fault values
        if mode == 'min':
            values[n] = np.min(node_values)
        
        elif mode == 'max':
            values[n] = np.max(node_values)
        
        elif mode == 'mean':
            values[n] = np.mean(node_values)
        
        elif mode == 'range':
            values[n] = np.max(node_values)-np.min(node_values)
    
        elif mode == 'sum':
            values[n] = np.sum(node_values)         
            
    return values






#******************************************************************************
# (5) NETWORK METRICS
# A couple of functions to calculate network properties
#******************************************************************************

def total_length(G):
    """ Calculate network length
    
    Parameters
    ----------
    G : nx.graph
        Fault network
        
    Returns
    -------  
    value
        Int
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'    

    # Calculation
    G = compute_edge_length(G)
    
    length = 0
    for edge in G.edges:
        length = length + G.edges[edge]['length']
        
    return length




def extract_attribute(G, arr, attribute):
    """ Extract attribute from image to network
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    arr : np.array
        Array to extract attribute from
    attribute : string
        Name of attribute assigned to network
    
    Returns
    -------  
    G
        nx.graph
    """

    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    assert isinstance(arr, np.ndarray), 'arr is not a NumPy array'    
    assert isinstance(attribute, str), 'Name of attribute is not a string'

    # Calculation    
    (x_max, y_max) = arr.shape
    for node in G:
        y, x = G.nodes[node]['pos']
        if (x >= x_max) or (y >= y_max):
            G.nodes[node][attribute] = float('nan')
        else:
            G.nodes[node][attribute] = arr[int(x), int(y)]
    return G


def extract_attribute_periodic(G, arr, attribute):
    """ Extract attribute from image to network with periodic bc in y-direction
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    arr : np.array
        Array to extract attribute from
    attribute : string
        Name of attribute assigned to network
    
    Returns
    -------  
    G
        nx.graph
    """

    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    assert isinstance(arr, np.ndarray), 'arr is not a NumPy array'    
    assert isinstance(attribute, str), 'Name of attribute is not a string'

    # Calculation    
    (x_max, y_max) = arr.shape
    for node in G:
        y, x = G.nodes[node]['pos']
        if y > y_max:
            G.nodes[node][attribute] = arr[int(x), int(y-y_max)]
        if y < 0:
            G.nodes[node][attribute] = arr[int(x), int(y+y_max)]
        else:
            G.nodes[node][attribute] = arr[int(x), int(y)]
    return G    




def extract_profile(G, attribute):
    """ Extract profile from network
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    attribute : string
        Name of attribute assigned to network
    
    Returns
    -------  
    array, array
        np.array, np.array
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    assert isinstance(attribute, str), 'Name of attribute is not a string'

    # Calculation   
    y = np.zeros(len(G.nodes))
    values = np.zeros(len(G.nodes))

    # Get values
    for n, node in enumerate(G):
        y[n] = G.nodes[node]['pos'][1]
        values[n] = G.nodes[node][attribute]

    # Resample
    y_min = np.min(y)
    y_max = np.max(y)
    y_resample = np.arange(y_min, y_max)

    def find_max_y(y, y_n, attribute):
        index = np.where(y == y_n)[0]
        return np.max(attribute[index])

    values_resample = np.zeros_like(y_resample)
    for n, y_n in enumerate(y_resample):
        values_resample[n] = find_max_y(y, y_n, values)

    return y_resample, values_resample





def calculate_edge_mid_point(G, edge):
    """ Calculate mid point of edge
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    edge : tuple
        Edge in network
    
    Returns
    -------  
    array, array
        np.array, np.array
    """    
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    assert isinstance(edge, tuple), 'Edge is not a tuple'

    # Calculation       
    x = int((G.nodes[edge[0]]['pos'][0] + G.nodes[edge[1]]['pos'][0])/2)
    y = int((G.nodes[edge[0]]['pos'][1] + G.nodes[edge[1]]['pos'][1])/2)

    return (x, y)





def calculate_mid_points(G):
    """ Calculate mid points of network
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    
    Returns
    -------  
    H
        nx.graph
    """   
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'

    # Calculation  
    H = nx.Graph()

    for edge in G.edges:
        node = (edge)
        H.add_node(node)
        H.nodes[node]['pos'] = calculate_edge_mid_point(G, edge)
        H.nodes[node]['component'] = -1
        H.nodes[node]['edge'] = (edge)

    return H


def calculate_direction(G, cutoff, normalize=True):
    """ Calculate direction for entire network
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    cutoff : int, float
        Cutoff distance for direction
    normalize : bolean
        Normalize direction (default: True)
    
    Returns
    -------  
    G
        nx.graph
    """   
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'

    # Calculation
    for node in G.nodes:

        length = nx.single_source_shortest_path_length(G, node, cutoff=cutoff)
        keys = [keys for keys, values in length.items() if values == max(length.values())]

        if len(keys) > 2:
            (node_0, node_1) = keys[:2]

        if len(keys) == 2:
            (node_0, node_1) = keys
        if len(keys) == 1:
            node_0 = keys[0]

            length = nx.single_source_shortest_path_length(G, node, cutoff=cutoff - 1)
            keys = [keys for keys, values in length.items() if values == max(length.values())]

            node_1 = keys[0]

        # extrac position
        pt_0 = G.nodes[node_0]['pos']
        pt_1 = G.nodes[node_1]['pos']

        # calculate vector
        dx = pt_0[0] - pt_1[0]
        dy = pt_0[1] - pt_1[1]
        
        # normalize
        v_norm = np.array([dx,dy])/np.linalg.norm([dx, dy])
        dx = v_norm[0]
        dy = v_norm[1]
        
        # write to graph
        G.nodes[node]['dx'] = dx
        G.nodes[node]['dy'] = dy

    return G




def generate_pickup_points(G, factor):
    """ Generate pick up points for network
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    factor : int, float
        Factor for distance of pick up points from fault
    
    Returns
    -------  
    H
        nx.graph
    """ 
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'

    # Calculation
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
        H.nodes[node_mid]['component'] = -1

        node_p = (node, 1)
        H.add_node(node_p)
        H.nodes[node_p]['pos'] = (x_p, y_p)
        H.nodes[node_p]['component'] = -2

        node_n = (node, 2)
        H.add_node(node_n)
        H.nodes[node_n]['pos'] = (x_n, y_n)
        H.nodes[node_n]['component'] = -3

        H.add_edge(node_n, node_p)

    return H




def filter_pickup_points(G, H):
    """ Filter pick up points
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    H : nx.graph
        Pick up points
    
    Returns
    -------  
    H
        nx.graph
    """ 
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'

    # Calculation
    for node in G:

        if (
            H.nodes[(node, 1)]['pos'][1] < 0 or
            H.nodes[(node, 2)]['pos'][1] < 0
        ):

            H.nodes[(node, 0)]['v_x'] = 0
            H.nodes[(node, 0)]['v_z'] = 0

            H.nodes[(node, 1)]['v_x'] = 0
            H.nodes[(node, 1)]['v_z'] = 0

            H.nodes[(node, 2)]['v_x'] = 0
            H.nodes[(node, 2)]['v_z'] = 0

    return H



def calculate_slip_rate(G, H, dim):
    """ Calculate slip rate from pick up points
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    H : nx.graph
        Pick up points
    dim : int
        Dimension of network
    
    Returns
    -------  
    G
        nx.graph
    """ 
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    assert isinstance(H, nx.Graph), 'H is not a NetworkX graph'

    # Calculation
    if dim == 2:
        for node in H.nodes:
            if node[1] == 0:  # centre point
                G.nodes[node[0]]['slip_rate_x'] = abs(
                    H.nodes[(node[0], 1)]['v_x'] -
                    H.nodes[(node[0], 2)]['v_x']
                )
                G.nodes[node[0]]['slip_rate_z'] = abs(
                    H.nodes[(node[0], 1)]['v_z'] -
                    H.nodes[(node[0], 2)]['v_z']
                )
                G.nodes[node[0]]['slip_rate'] = math.sqrt(
                    G.nodes[node[0]]['slip_rate_x']**2 +
                    G.nodes[node[0]]['slip_rate_z']**2
                )
    if dim == 3:
        for node in H.nodes:
            if node[1] == 0:
                G.nodes[node[0]]['slip_rate_x'] = abs(
                    H.nodes[(node[0], 1)]['v_x'] -
                    H.nodes[(node[0], 2)]['v_x']
                )
                G.nodes[node[0]]['slip_rate_y'] = abs(
                    H.nodes[(node[0], 1)]['v_y'] -
                    H.nodes[(node[0], 2)]['v_y']
                )
                G.nodes[node[0]]['slip_rate_z'] = abs(
                        H.nodes[(node[0], 1)]['v_z'] -
                        H.nodes[(node[0], 2)]['v_z']
                )
                G.nodes[node[0]]['slip_rate'] = math.sqrt(
                    G.nodes[node[0]]['slip_rate_x']**2 +
                    G.nodes[node[0]]['slip_rate_y']**2 +
                    G.nodes[node[0]]['slip_rate_z']**2
                )

    return G





def calculate_slip(G, H, dt, dim):
    """ Calculate slip from pick up points
    
    Parameters
    ----------
    G : nx.graph
        Fault network
    H : nx.graph
        Pick up points
    dt : float
        Time step
    dim : int
        Dimension of network
    
    Returns
    -------  
    G
        nx.graph
    """ 
    
    # Assertions
    assert isinstance(G, nx.Graph), 'G is not a NetworkX graph'
    assert isinstance(H, nx.Graph), 'H is not a NetworkX graph'

    # Calculation
    if dim == 2:
        for node in H.nodes:
            if node[1] == 0:
                G.nodes[node[0]]['slip_x'] = abs(
                    H.nodes[(node[0], 1)]['v_x'] -
                    H.nodes[(node[0], 2)]['v_x']
                )*dt
                G.nodes[node[0]]['slip_z'] = abs(
                    H.nodes[(node[0], 1)]['v_z'] -
                    H.nodes[(node[0], 2)]['v_z']
                )*dt
                G.nodes[node[0]]['slip'] = math.sqrt(
                    G.nodes[node[0]]['slip_x']**2 +
                    G.nodes[node[0]]['slip_z']**2
                )

    if dim == 3:
        for node in H.nodes:
            if node[1] == 0:
                G.nodes[node[0]]['slip_x'] = abs(
                    H.nodes[(node[0], 1)]['v_x'] -
                    H.nodes[(node[0], 2)]['v_x']
                )*dt
                G.nodes[node[0]]['slip_y'] = abs(
                    H.nodes[(node[0], 1)]['v_y'] -
                    H.nodes[(node[0], 2)]['v_y']
                )*dt
                G.nodes[node[0]]['slip_z'] = abs(
                    H.nodes[(node[0], 1)]['v_z'] -
                    H.nodes[(node[0], 2)]['v_z']
                )*dt
                G.nodes[node[0]]['slip'] = math.sqrt(
                    G.nodes[node[0]]['slip_x']**2 +
                    G.nodes[node[0]]['slip_y']**2 +
                    G.nodes[node[0]]['slip_z']**2
                )
    return G




