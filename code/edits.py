import math
import random
import numpy as np
import networkx as nx

import sys
sys.path.append('./fault_analysis/code/')

from metrics import *


import copy

# import point_cloud_utils as pcu









def array_to_points(arr):
    N = np.count_nonzero(arr)
    points = np.zeros((N, 2))
    (points[:,1], points[:,0]) = np.where(arr != 0)
    return points













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
















def remove_triangles(G):
    
    G = compute_edge_length(G)
    
    # Find triangles through edges        
    triangles = []        
    for node in G:
        for n in G.neighbors(node):
            for nn in G.neighbors(n):
                for nnn in G.neighbors(nn):
                    if node == nnn:
                        triangles.append((node, n, nn))                                               
                        
    triangles = set(tuple(sorted(t)) for t in triangles)   
    
                    
    # Remove triangles
    H = G.copy()
            
    for t in triangles:  
        # Find longest edge
        length = 0
        for edge in [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]:
            if G.edges[edge]['length']>length:
                length = G.edges[edge]['length']
                longest_edge = edge
                
        # Remove longest edge                                    
        if longest_edge in list(H.edges):
            H.remove_edge(*longest_edge)
                     
    return H





def remove_cycles(G):

    nodes_to_remove = set()
    
    # Find cycles
    cycles = nx.cycle_basis(G)
    
    for n, cycle in enumerate(cycles):
        
        # Find y-nodes (i.e. node with 3 edges)
        y_nodes = [node for node in cycle if G.degree(node)==3]    
        
        # If cycle has only one y-node, remove it (except the y-node itself)
        if len(y_nodes)==1:
            print('Cycle ' + str(n) + ' has only one y-node. Remove it')                
            for node in cycle:
                if node not in y_nodes:
                    nodes_to_remove.add(node)
                    
    G.remove_nodes_from(nodes_to_remove)
    
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




def split_triple_junctions(G, dos, plot=False):
    # This function splits up triple junctions (or y-nodes) based on their 
    # orientation, so that the two branches most closely aligned remain
    # connected and the splay is cut off.
    # 
    # Parameters
    # ---------------------------------------------------------------------
    # dos - depth of search or the number of nodes used to determine the 
    # orientation of each branch
    import matplotlib.pyplot as plt
    
    count = 0 
        
    for node in G:
        if G.degree[node] == 3:     # Find y-node
            
            count = count+1         # Increase counter
            
            true_neighbors = list(G.neighbors(node))    # Find 1st neighbors
            
            # Set up branches, i.e. list of nodes belonging to each
            # (rgb refers to the colors when plotting)
            branch_r = [true_neighbors[0]]
            branch_g = [true_neighbors[1]]
            branch_b = [true_neighbors[2]]
            
            # Set up set of eadges belonging to each branch                        
            edges_r = {(node, true_neighbors[0])}
            edges_g = {(node, true_neighbors[1])}
            edges_b = {(node, true_neighbors[2])}
            
            # Extract branches - it starts with the y-node (i.e. node) as
            # origins and the true neighbors as neighbors. Then we search
            # for the neighbors' neighbors (i.e. new_neighbors) ignoring
            # the origins. Next the neighbors become the origins and the
            # new neighbors become the neighbors, so that the search can go
            # one level deeper.
            origins = [node]*3
            neighbors = true_neighbors
            
            for n in range(dos):                     
                new_neighbors = find_new_neighbors(G, neighbors, origins)
                origins   = neighbors
                neighbors = new_neighbors
                
                # Add new neigbors to branch, if they're not already are
                if new_neighbors[0] not in branch_r:                
                    branch_r.append(new_neighbors[0])
                    
                if new_neighbors[1] not in branch_g:                
                    branch_g.append(new_neighbors[1])  
                    
                if new_neighbors[2] not in branch_b:                
                    branch_b.append(new_neighbors[2])
                    
    
                # Add new edges to set of edges, unless they're self-edges
                if origins[0] != neighbors[0]:
                    edges_r.add((origins[0], neighbors[0]))
                if origins[1] != neighbors[1]:
                    edges_g.add((origins[1], neighbors[1]))
                if origins[2] != neighbors[2]:
                    edges_b.add((origins[2], neighbors[2]))
                
                
            # Plot y-nodes with edges
            if plot:
                
                plt.figure(figsize=(12,12))
            
                nx.draw_networkx_nodes(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       nodelist=[node],
                                       node_color="yellow")
                    
                nx.draw_networkx_nodes(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       nodelist=branch_r,
                                       node_color="red")
                
                nx.draw_networkx_edges(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       edgelist=edges_r,
                                       edge_color="red")
                
                nx.draw_networkx_nodes(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       nodelist=branch_g,
                                       node_color="green")
                
                nx.draw_networkx_edges(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       edgelist=edges_g,
                                       edge_color="green")
                
                nx.draw_networkx_nodes(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       nodelist=branch_b,
                                       node_color="blue")
                
                nx.draw_networkx_edges(G, 
                                       pos = nx.get_node_attributes(G, 'pos'),
                                       edgelist=edges_b,
                                       edge_color="blue")
                
                plt.axis('equal')
                
                            
            
            
            # Calculate slope of each branch
            def slope(G, nodes):
                
                x = [G.nodes[node]['pos'][0] for node in nodes]
                y = [G.nodes[node]['pos'][1] for node in nodes]
                
                dx = x[0]-x[-1]
                dy = y[0]-y[-1]
                
                # If point cloud is vertical
                if abs(dy/dx) > 8:
                    slope = 1e16
                    x_pred = np.ones_like(x)*np.mean(x)
                    y_pred = y                   
                    
                # If point cloud is 'normal', fit linear function
                else:
                    slope = dy/dx
                    intercept = y[0]-dy/dx*x[0]
                    x_pred = x
                    y_pred = [slope*xn + intercept for xn in x_pred]
                  
                return x_pred, y_pred, slope
                
            
            x_r, y_r, slope_r = slope(G, branch_r)
            x_g, y_g, slope_g = slope(G, branch_g) 
            x_b, y_b, slope_b = slope(G, branch_b)
            
                            
            # Convert slopes to angles              
            angle_r = np.degrees(np.arctan(slope_r))
            angle_g = np.degrees(np.arctan(slope_g))
            angle_b = np.degrees(np.arctan(slope_b))
            
            
            # Plot linear approximation to check angles
            if plot:  
                plt.plot(x_r, y_r, 'r', linewidth=2) 
                plt.plot(x_g, y_g, 'g', linewidth=2)
                plt.plot(x_b, y_b, 'b', linewidth=2)
            
            
            # Calculate differences in angles (from -180 to +180 degrees)
            def differnce_between_angles(a0,a1):
            	return abs((((2*a1-2*a0+540)%360)-180)/2)
            
            diff_rg = differnce_between_angles(angle_r, angle_g)
            diff_rb = differnce_between_angles(angle_r, angle_b)
            diff_gb = differnce_between_angles(angle_g, angle_b)
            
            
            # Plot angles and differences as figure title
            if plot:
                plt.suptitle('Angles: Red: '     + str(round(angle_r)) +
                          ', Green: ' + str(round(angle_g)) +
                          ', Blue: '  + str(round(angle_b)))
                
                plt.title(' Minimum difference: RG: ' + str(round(diff_rg)) +
                          ', RB: ' + str(round(diff_rb)) +
                          ', GB: ' + str(round(diff_gb)))
            
            
            # Split y-node based on difference in angles
            # If angle between red and green branch is smallest, remove 
            # blue branch.
            if diff_rg < diff_rb and diff_rg < diff_gb:
                if plot:
                    nx.draw_networkx_edges(G, 
                                pos = nx.get_node_attributes(G, 'pos'),
                                edgelist=[(node, true_neighbors[2])],
                                edge_color="black",
                                width=10)
                G.remove_edge(node, true_neighbors[2])
                
            # If angle between red and blue branch is lowest, remove green
            # branch.
            elif diff_rb < diff_rg and diff_rb < diff_gb:
                if plot:
                    nx.draw_networkx_edges(G, 
                                pos = nx.get_node_attributes(G, 'pos'),
                                edgelist=[(node, true_neighbors[1])],
                                edge_color="black",
                                width=10)
                G.remove_edge(node, true_neighbors[1])
                
            # If angle between green and blue brach is smallest (and all 
            # equal cases), remove red branch.
            else:
                if plot:
                    nx.draw_networkx_edges(G, 
                            pos = nx.get_node_attributes(G, 'pos'),
                            edgelist=[(node, true_neighbors[0])],
                            edge_color="black",
                            width=10)
                G.remove_edge(node, true_neighbors[0])
            
            
            
            
            # Split y-node based on difference in angles based on threshold
            # threshold = 35
            
            # if diff_rg < threshold:
            #     nx.draw_networkx_edges(G, 
            #                         pos = nx.get_node_attributes(G, 'pos'),
            #                         edgelist=[(node, true_neighbors[2])],
            #                         edge_color="black",
            #                         width=10)
    
            # elif diff_rb < threshold:
            #     nx.draw_networkx_edges(G, 
            #                         pos = nx.get_node_attributes(G, 'pos'),
            #                         edgelist=[(node, true_neighbors[1])],
            #                         edge_color="black",
            #                         width=10)
    
            # elif diff_gb < threshold:
            #     nx.draw_networkx_edges(G, 
            #                         pos = nx.get_node_attributes(G, 'pos'),
            #                         edgelist=[(node, true_neighbors[0])],
            #                         edge_color="black",
            #                         width=10)
                
            # else:
            #     nx.draw_networkx_edges(G, 
            #             pos = nx.get_node_attributes(G, 'pos'),
            #             edgelist=[(node, true_neighbors[0]), (node, true_neighbors[1]), (node, true_neighbors[2])],
            #             edge_color="black",
            #             width=10)
                
          
            if plot:
                plt.show()
                
            # # Stop criteria to look at certain junctions
            # if count==29:
            #     break
        

    return G












# def split(G, depth, tol):
 
#      for node in G:
#          if G.degree[node] == 3:
#              true_neighbors = list(G.neighbors(node))
             
             
#              origins = [node]*3
#              neighbors = true_neighbors
#              for n in range(depth):                     
#                  new_neighbors = find_new_neighbors(G, neighbors, origins)
#                  origins   = neighbors
#                  neighbors = new_neighbors
                 
                 
                              
#              strikes = np.zeros(3)
             
#              for k in range(3):
#                  strikes[k] = strike_between_nodes_pix(G, node, new_neighbors[k])
             
#              print('Strikes')
#              print(strikes)    
             
#              # strike_difference = np.zeros(3)
#              # strike_difference[0] = abs(strikes[0]-strikes[1])
#              # strike_difference[1] = abs(strikes[1]-strikes[2])
#              # strike_difference[2] = abs(strikes[0]-strikes[2])
 
#              # print('Strike difference')                                       
#              # print(strike_difference)
             
#              # index = np.arange(3) 
#              # p = strike_difference.argsort()
#              # index = index[strike_difference.argsort()]
#              # strike_difference = strike_difference[strike_difference.argsort()]
 
 
             
#              # if strike_difference[0] < tol:
#              #     G.remove_edge(node, true_neighbors[index[0]]) 
                 
#              # if strike_difference[1] < tol:
#              #     G.remove_edge(node, true_neighbors[index[1]])                
                 
             
#              if abs(strikes[0]-strikes[1]) < tol:
#                  print(2)
#                  G.remove_edge(node, true_neighbors[2])    
#              if abs(strikes[1]-strikes[2])  < tol:
#                  print(0)
#                  G.remove_edge(node, true_neighbors[0])    
#              if abs(strikes[0]-strikes[2])  < tol:
#                  print(1)
#                  G.remove_edge(node, true_neighbors[1])
                 
#      return G






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
    for col in range(matrix.shape[0]):
        for row in range(matrix.shape[1]):
            if matrix[col, row] < threshold:
                connections.append([columns[row], rows[col]])        
    return connections





    
    
    
    
    
    

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





def get_displacement(G, dim):  
    if dim == 2:
        points = np.zeros((len(list(G)),6))    
        for n, node in enumerate(G):
            points[n,0] = node
            points[n,1] = G.nodes[node]['pos'][0]
            points[n,2] = G.nodes[node]['pos'][1]
            points[n,3] = G.nodes[node]['heave']
            points[n,4] = G.nodes[node]['throw']
            points[n,5] = G.nodes[node]['displacement']
    if dim == 3:
        points = np.zeros((len(list(G)),7))    
        for n, node in enumerate(G):
            points[n,0] = node
            points[n,1] = G.nodes[node]['pos'][0]
            points[n,2] = G.nodes[node]['pos'][1]
            points[n,3] = G.nodes[node]['heave']
            points[n,4] = G.nodes[node]['lateral']
            points[n,5] = G.nodes[node]['throw']
            points[n,6] = G.nodes[node]['displacement']
    return points



def get_slip_rate(G, dim): 
    if dim == 2:
        points = np.zeros((len(list(G)),6))    
        for n, node in enumerate(G):
            points[n,0] = node
            points[n,1] = G.nodes[node]['pos'][0]
            points[n,2] = G.nodes[node]['pos'][1]
            points[n,3] = G.nodes[node]['slip_rate_x']
            points[n,4] = G.nodes[node]['slip_rate_z']
            points[n,5] = G.nodes[node]['slip_rate']
    if dim == 3:
        points = np.zeros((len(list(G)),7))    
        for n, node in enumerate(G):
            points[n,0] = node
            points[n,1] = G.nodes[node]['pos'][0]
            points[n,2] = G.nodes[node]['pos'][1]
            points[n,3] = G.nodes[node]['slip_rate_x']
            points[n,4] = G.nodes[node]['slip_rate_y']
            points[n,5] = G.nodes[node]['slip_rate_z']
            points[n,6] = G.nodes[node]['slip_rate']
    return points


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist)


def assign_displacement(G, points, dim):
    if dim == 2:
        for node in G:
            for point in points:
                if node == point[0]:
                    G.nodes[node]['heave'] = point[3]
                    G.nodes[node]['throw'] = point[4]
                    G.nodes[node]['displacement'] = point[5]
    if dim == 3:
        for node in G:
            for point in points:
                if node == point[0]:
                    G.nodes[node]['heave']   = point[3]
                    G.nodes[node]['lateral'] = point[4]
                    G.nodes[node]['throw']   = point[5]
                    G.nodes[node]['displacement'] = point[6]   
    return G




def write_slip_to_displacement(G, dim):    
    if dim == 2:
        for node in G:
            G.nodes[node]['heave'] = G.nodes[node]['slip_x']
            G.nodes[node]['throw'] = G.nodes[node]['slip_z']
            G.nodes[node]['displacement'] = G.nodes[node]['slip']

    if dim == 3:
        for node in G:
            G.nodes[node]['heave']   = G.nodes[node]['slip_x']            
            G.nodes[node]['lateral'] = G.nodes[node]['slip_y']
            G.nodes[node]['throw']   = G.nodes[node]['slip_z']
            G.nodes[node]['displacement'] = G.nodes[node]['slip']
    return G
