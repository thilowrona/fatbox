import math
import random
import numpy as np
import networkx as nx

import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from metrics import *


import copy

import point_cloud_utils as pcu




import ngtpy

def add_edges_fast(G, dim, distance, max_conn):
    
    objects = []
    for node in G:
        objects.append([G.nodes[node]['pos'][0], G.nodes[node]['pos'][1]])


    ngtpy.create(b"tmp", dim)
    index = ngtpy.Index(b"tmp")
    index.batch_insert(objects)
    index.save()
    
    H = G.copy()
    
    for n, node in enumerate(G):        
        query = objects[n]        
        neighbors = index.search(query, max_conn)           
        for neighbor in neighbors:
            if neighbor[1] < distance:
                H.add_edge(node, neighbor[0])
                
    H = remove_self_edge(H)
                
    return H





def add_edges(G, N):   
 
    # def distance_between_nodes(G, node0, node1):
    #     (x0, y0) = G.nodes[node0]['pos']
    #     (x1, y1) = G.nodes[node1]['pos']
    #     return math.sqrt((x0-x1)**2+(y0-y1)**2)


    def find_closest(G, node):
        threshold = 1000000
        for other in G:
            d = distance_between_nodes_pix(G, node, other)
            if 0 < d < threshold:
                threshold = d
                index = other
        return index
  

    for node in G:
        print(str(node) + ' of ' + str(N))
        closest = find_closest(G, node)
        if (closest, node) not in G.edges:
            G.add_edge(node, closest)
            G.nodes[node]['edges'] = 1
            G.nodes[closest]['edges'] = 1

    def clostest_except(G, node, cn):
        index = float('nan')
        threshold = 1000000
        for other in G:
            if other not in cn:
                d = distance_between_nodes_pix(G, node, other)
                if 0 < d < threshold:
                    threshold = d
                    index = other
        return index, threshold


    for node in G:
        print(str(node) + ' of ' + str(N))
        if len(G.edges(node)) == 1:
            cn = nx.node_connected_component(G, node)
            index, threshold  = clostest_except(G, node, cn)
            if threshold < 2:
                G.add_edge(node, index)
                G.nodes[node]['edges'] = 1
                G.nodes[index]['edges'] = 1
    return G

































## COMPONENTS
def label_components(G):
    for label, cc in enumerate(sorted(nx.connected_components(G))): 
        for n in cc:
            G.nodes[n]['component'] = label
    return G


def select_components(G, components):
    H = G.copy()
    if type(components) != list:
        selected_nodes = [n[0] for n in H.nodes(data=True) if n[1]['component'] == components]
    else:
        selected_nodes = [n[0] for n in H.nodes(data=True) if n[1]['component'] in components]      
    H = H.subgraph(selected_nodes)
    return H



    




def remove_component(G, component):
    for cc in sorted(nx.connected_components(G)):
        for n in cc:
            if G.nodes[n]['component'] == component:
                G.remove_node(n)
    return G



def remove_small_components(G, minimum_size = 10):
    for cc in sorted(nx.connected_components(G)):
        if len(cc) < minimum_size:
            for n in cc:
                G.remove_node(n)
    return G







def find_neighbor_except(G, neighbor, node):
    if len(list(G.neighbors(neighbor))) != 2:
        return neighbor
    else:
        for nn in G.neighbors(neighbor):
            if nn != node:
                return nn
    



def find_new_neighbors(G, neighbors, origins):
    new_neighbors = [None]*3
    for k in range(3):
        new_neighbors[k] = find_neighbor_except(G, neighbors[k], origins[k])            
    return new_neighbors




def split(G, depth, tol):

    for node in G:
        if G.nodes[node]['edges'] == 3:
            true_neighbors = list(G.neighbors(node))

            
            
            origins = [node]*3
            neighbors = true_neighbors
            for n in range(depth):                     
                new_neighbors = find_new_neighbors(G, neighbors, origins)
                origins   = neighbors
                neighbors = new_neighbors
                
                                    
            strikes = np.zeros(3)
            
            for k in range(3):
                strikes[k] = strike_between_nodes_xz(G, node, new_neighbors[k])
            print(strikes)
            
            if abs(strikes[0]+strikes[1]) < tol:
                print(2)
                G.remove_edge(node, true_neighbors[2])    
            if abs(strikes[1]+strikes[2])  < tol:
                print(0)
                G.remove_edge(node, true_neighbors[0])    
            if abs(strikes[0]+strikes[2])  < tol:
                print(1)
                G.remove_edge(node, true_neighbors[1])
                
    return G









def connect_components(G, cc0, cc1, relabel=True):
    
    edge0 = []    
    for node in cc0:        
        if G.nodes[node]['edges'] == 1:            
            edge0.append(node)
        

    edge1 = []    
    for node in cc1:        
        if G.nodes[node]['edges'] == 1:            
            edge1.append(node)        
    
    value = 1000000

    for e0 in edge0:
        for e1 in edge1:
            distance = distance_between_nodes_pix(G, e0, e1)
            if distance < value:
                value = distance
                ep0 = e0
                ep1 = e1
                    
    G.add_edge(ep0, ep1)
    
    
    if relabel == True:
        for node in cc1:
            label = G.nodes[node]['component']
        
        for node in cc0:
            G.nodes[node]['component'] = label
    
    return G







def expand_network(G, relabel=True, vertical_shift = 960, distance = 5):
    
    def minimum_y(G, nodes):
        
        minimum = 1000000
        
        for node in nodes:
            y = G.nodes[node]['pos'][1]
            if y < minimum:
                minimum = y
                    
        return minimum



    def maximum_y(G, nodes): 
        
        maximum = -1000000
        
        for node in nodes:
            y = G.nodes[node]['pos'][1]
            if y > maximum:
                maximum = y
                    
        return maximum





    def add_y(G, nodes, value):
        
        for node in nodes:
            G.nodes[node]['pos'] = (G.nodes[node]['pos'][0], G.nodes[node]['pos'][1] + value)
            






    def distance_between_nodes(G, n0, n1):
        
        x0, y0 = G.nodes[n0]['pos']
        x1, y1 = G.nodes[n1]['pos']     
        
        return math.sqrt((x0 -x1)**2+(y0-y1)**2)







    def within_reach(G, cc0, cc1, value):
            
        min_dist = 10000000
        
        for n0 in cc0:
            for n1 in cc1:
                distance = distance_between_nodes(G, n0, n1)
                if distance < min_dist:
                    min_dist = distance            
                    
        if min_dist < value:
            return True
        else:
            return False







     
        
        
        
        return G        

    
        

    
    



    
    
    # Expand network upwards
    for n_cc, cc in enumerate(sorted(nx.connected_components(G))):
        
        # Component reaches lower, but not upper boundary        
        if minimum_y(G, cc) == 0 and maximum_y(G, cc) != 959:
            
            # Move component up
            print('Move component ' + str(n_cc) + ' up')
            add_y(G, cc, vertical_shift)
            
            # Connect to other component
            for n_cc_other, cc_other in enumerate(sorted(nx.connected_components(G))):
                
                # if it is within reach
                if within_reach(G, cc, cc_other, distance) and cc!=cc_other:
                    print('Connect component ' + str(n_cc) + ' to ' + str(n_cc_other))
                    G = connect_components(G, cc, cc_other, relabel)
              
                
                
    # Expand network downwards
    for n_cc, cc in enumerate(sorted(nx.connected_components(G))):                
            
        # Component reaches upper, but not lower boundary                
        if minimum_y(G, cc) != 0 and maximum_y(G, cc) == 959:
    
            # Move component down
            print('Move component ' + str(n_cc) + ' down')
            add_y(G, cc, -vertical_shift)
        
            # Connect to other component
            for n_cc_other, cc_other in enumerate(sorted(nx.connected_components(G))):    
        
                # if it is within reach
                if within_reach(G, cc, cc_other, distance) and cc!=cc_other:
                    print('Connect component ' + str(n_cc) + ' to ' + str(n_cc_other))
                    G = connect_components(G, cc, cc_other, relabel)
                    
    
    return G 





# Connect close components
def min_dist_comp(G, cc0, cc1):
    threshold = 1e6
    for n0 in cc0:
        for n1 in cc1:
            distance = distance_between_nodes_pix(G, n0, n1)
            if distance < threshold:
                threshold = distance
    return threshold
    

def connect_close_components(G, value):
    for label1, cc0 in enumerate(sorted(nx.connected_components(G))):
        for label2, cc1 in enumerate(sorted(nx.connected_components(G))):         
            if min_dist_comp(G, cc0, cc1) < value:
                G = connect_components(G, cc0, cc1)
    return G



def max_dist_comp(G, cc0, cc1):
    threshold = 0
    for n0 in cc0:
        for n1 in cc1:
            distance = distance_between_nodes2(G, n0, n1)
            if distance > threshold:
                threshold = distance
    return threshold





## SIMPLIFY GRAPH    
def simplify(G, degree):
    
    H = G.copy()

    for k in range(degree):
    
        for n, node in enumerate(list(nx.dfs_preorder_nodes(H))):
            
            # print(H.degree(node))
        
            if H.degree(node) == 2 and (n % 2) == 0:
            
                edges = list(H.edges(node))
        
                H.add_edge(edges[0][1], edges[1][1])
        
                H.remove_node(node)

    return H









## SIMILARITY
## COMPUTE CONNECTIONS
# Distance between two points
def distance_between_points(n0, n1):
    return math.sqrt((n0[1] -n1[1])**2+(n0[0] -n1[0])**2)


# Minimum distance between point and graph
def min_dist(point, graph):
    threshold = 1000000    
    for node in graph:
        if distance_between_points(point, graph.nodes[node]['pos']) < threshold:
            threshold = distance_between_points(point, graph.nodes[node]['pos'])    
    return threshold    


# Measure of similarity between two components
def similarity_between_components(G, H):

    # Determine short and long graph
    if len(G.nodes) > len(H.nodes):
        short = H
        long  = G 
    else:
        short = G
        long  = H    

    N = len(long.nodes)     
    distance = np.zeros((N))
    
    for n in range(N):                
        distance[n] = min_dist(long.nodes[random.choice(list(long))]['pos'], short)
    
    return np.average(distance)    



def get_components(G):
    components = []
    for cc in sorted(nx.connected_components(G)):
        components.append(G.nodes[cc.pop()]['component'])
    return components



def assign_components(G, components):
    for n, cc in enumerate(sorted(nx.connected_components(G))):
        for node in cc:
            G.nodes[node]['component'] = components[n]
    return G


def common_components(G, H):
    C_G = get_components(G)
    C_H = get_components(H)
    return list(set(C_G) & set(C_H))


def unique_components(G_0, G_1):
    G_0_components = set(get_components(G_0))
    G_1_components = set(get_components(G_1))

    return ([item for item in G_0_components if item not in G_1_components], 
            [item for item in G_1_components if item not in G_0_components])





def similarity_between_graphs(G, H, normalize=True):
        
    components_G = sorted(components(G))
    components_H = sorted(components(H))
            
    matrix = np.zeros((len(components_G), len(components_H)))
    
    for n, c_G in enumerate(components_G):
        cc_G = select_components(G, components=c_G)
            
        for m, c_H in enumerate(components_H):        
            cc_H = select_components(H, components=c_H)
    
            matrix[n,m] = similarity_between_components(cc_G, cc_H)
    
    if normalize:
        minimum = np.min(matrix)
        maximum = np.max(matrix)
        
        matrix = (matrix-minimum)/(maximum-minimum)    
    
    
    
    return matrix, components_G, components_H






            
    

# Compute connections from similarity
def similarity_to_connection(matrix, rows, columns, threshold):    
    connections = []
    for n, row in enumerate(matrix):
        if np.min(row) < threshold:
            index = np.argmin(row)      
            connections.append([columns[index], rows[n]])            
    return connections





    

def hausdorff_distance(G, H, normalize=True):
    
    hausdorff_dist = np.zeros((len(list(nx.connected_components(G))), len(list(nx.connected_components(H)))))
    
    for n, cc_0 in enumerate(sorted(nx.connected_components(G))):
        a = np.zeros((len(cc_0), 3))   
        for k, node in enumerate(cc_0):
            a[k,0] = G.nodes[node]['pos'][0]
            a[k,1] = G.nodes[node]['pos'][1]
    
        for m, cc_1 in enumerate(sorted(nx.connected_components(H))):
            b = np.zeros((len(cc_1), 3))   
            for l, node in enumerate(cc_1):
                b[l,0] = H.nodes[node]['pos'][0]
                b[l,1] = H.nodes[node]['pos'][1]
    
    
            # Compute each one sided squared Hausdorff distances
            hausdorff_a_to_b = pcu.hausdorff(a, b)
            hausdorff_b_to_a = pcu.hausdorff(b, a)
            
            # Take a max of the one sided squared  distances to get the two sided Hausdorff distance
            hausdorff_dist[n,m] = max(hausdorff_a_to_b, hausdorff_b_to_a)
    
    
    if normalize:
        minimum = np.min(hausdorff_dist)
        maximum = np.max(hausdorff_dist)
        
        hausdorff_dist = (hausdorff_dist-minimum)/(maximum-minimum)
    
    return hausdorff_dist


    
    
    
    
    
    

## RELABEL
def relabel(G, connections, count):

    sources = np.zeros((len(connections)),int)
    targets = np.zeros((len(connections)),int)
    
    for n, connection in enumerate(connections):
        sources[n] = connection[0]
        targets[n] = connection[1]
    
    highest_index = max(np.max(sources), count)
    
    
    components_old = get_components(G)    
    components_new = [None] * len(components_old)
    
    for n in range(len(components_old)):
        component = components_old[n]    
        if component in sources:
            index = np.where(sources==component)[0][0]
            components_new[n] = targets[index]
        else:
            components_new[n] = highest_index + 1
            highest_index += 1
    
    
    G = assign_components(G, components_new)
    
    return G, count









def dip_diff(d0,d1):
    if d0 >= 45:
        if d1 > 0:
            return abs(d1-d0)
        if d1 <= 0:
            return abs(d0+d1)
    
    if 0 < d0 < 45:
        if d1 >= 0:
            return abs(d1-d0)
        if d1 < 0:
            if d1 > -45:
                return d0 + abs(d1)
            if d1 <= -45:
                return 180 - d0 + d1
    
    if d0 == 0:
        return abs(d1)
    
    if 0 > d0 > -45:
        if d1 <= 0:
            return abs(d0-d1)
        if d1 > 0:
            if d1 > 45:
                return d0 + d1
            if d1 <= 45:
                return abs(d0) + d1

    if d0 <= -45:
        if d1 <= 0:
            return abs(d1-d0)
        if d1 > 0:
            if d1 > 45:
                return abs(d0+d1)         
            if d1 <= 45:
                return 180 + d0 - d1 



def dip_matrix(dips):
    N = dips.shape[0]
    diff = np.zeros((N,N))
    
    for n in range(N):
        for m in range(N):
            diff[n,m] = dip_diff(dips[n],dips[m])
    
    return diff








def remove_below(G, attribute, value):
    for node in G.nodes:
        if G.nodes[node][attribute] > value:
            G.nodes[node][attribute] = float('nan')
    return G










def remove_y_nodes(G):
    H = G.copy()
    for node in G:
        if H.nodes[node]['edges'] == 3:
            H.remove_node(node)
    return H




def remove_self_edge(G):
    for edge in G.edges:    
        if edge[0] == edge[1]:
            G.remove_edge(*edge)
    return G




def combine_graphs(G, H):

    highest_node = max(list(G.nodes)) + 1
    nodes_new = [node + highest_node for node in H.nodes]
    
    mapping = dict(zip(H.nodes, nodes_new))
    H = nx.relabel_nodes(H, mapping)
        
    F = nx.compose(G, H)

    return F









def get_nodes(G):    
    points = np.zeros((len(list(G)),4))    
    for n, node in enumerate(G):
        points[n,0] = node
        points[n,1] = G.nodes[node]['pos'][0]
        points[n,2] = G.nodes[node]['pos'][1]
        points[n,3] = G.nodes[node]['displacement']
    return points


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist)


def assign_nodes(G, points):
    for node in G:
        for point in points:
            if node == point[0]:
                G.nodes[node]['displacement'] = point[3]        
    return G