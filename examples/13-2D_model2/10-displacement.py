import matplotlib.pyplot as plt
plt.close("all")

import numpy as np
from numpy import genfromtxt
import meshio
import networkx as nx
import pickle

# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from image_processing import *
from edits import *
from metrics import *
from plots import *




# for time in range(0, 10400000, 200000):

time = 800000

print('time ' + str(time))

name = str(time).zfill(7)

data = genfromtxt('./csv/others/' + name + '.csv', delimiter=',')

data = np.flip(data, axis=0)

plastic_strain = data[:-1,0].reshape(600, 2500)
strain_rate    = data[:-1,1].reshape(600, 2500)
v_x            = data[:-1,2].reshape(600, 2500)
v_z            = data[:-1,3].reshape(600, 2500)


G = pickle.load(open('./graphs/extracted/graph_' + name + '.p', 'rb'))


# plt.matshow(v_z)
# plt.colorbar()









# G = select_component(G, 17)

# Unfreeze graph
G = nx.Graph(G)

# G = simplify(G, 2)

G = remove_self_edge(G)


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
            
        # normalize coordinates
        if normalize:
            dx = dx/(abs(dx)+abs(dy))
            dy = dy/(abs(dx)+abs(dy))
    
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




G = calculate_direction(G, 3)

H = calculate_pickup_points(G, 5)

H = extract_attribute(H, v_x, 'v_x')
H = extract_attribute(H, v_z, 'v_z')


def calculate_displacement(G, H, dt):
    for node in H.nodes:
        if node[1] == 0:
            G.nodes[node[0]]['heave'] =  (H.nodes[(node[0], 1)]['v_x'] - H.nodes[(node[0], 2)]['v_x'])*dt
            G.nodes[node[0]]['throw'] =  (H.nodes[(node[0], 1)]['v_z'] - H.nodes[(node[0], 2)]['v_z'])*dt
            G.nodes[node[0]]['displacement']    =  math.sqrt(G.nodes[node[0]]['heave']**2 + G.nodes[node[0]]['throw']**2)
    return G
        
        
G = calculate_displacement(G, H, dt=200000)






fig, ax = plt.subplots()
ax.matshow(v_x)
plot_components(G, ax)
plot_components(H, ax)






fig, ax = plt.subplots()
ax.matshow(v_x)
plot_attribute(G, 'displacement', ax)







# G = remove_below(G, 'displacement', value = 500)




fig, ax = plt.subplots()
ax.matshow(strain_rate)
plot_attribute(G, 'displacement', ax)


plt.figure()
cross_plot(G, 'displacement', 'x')
plt.gca().invert_yaxis()




