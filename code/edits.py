import math
import random
import numpy as np
import networkx as nx

from graph_metrics import*




def remove_small_components(G, minimum_size = 10):
    for cc in sorted(nx.connected_components(G)):
        if len(cc) < minimum_size:
            for n in cc:
                G.remove_node(n)
    return G




## COMPONENTS
def label_components(G):
    for label, cc in enumerate(sorted(nx.connected_components(G))): 
        for n in cc:
            G.nodes[n]['component'] = label
    return G


def select_component(G, component=0):    
    selected_nodes = [n[0] for n in G.nodes(data=True) if n[1]['component'] == component]  
    c = G.subgraph(selected_nodes)
    return c























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
            
        return G






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







    def connect_components(G, cc0, cc1, relabel):
        
        edge0 = []    
        for node in cc0:        
            if G.nodes[node]['edges'] == 1:            
                edge0.append(node)
            
    
        edge1 = []    
        for node in cc1:        
            if G.nodes[node]['edges'] == 1:            
                edge1.append(node)        
        
        value = 100000
    
        for e0 in edge0:
            for e1 in edge1:
                distance = distance_between_nodes(G, e0, e1)
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














## SIMPLIFY GRAPH    
def simplify(G, degree):

    for k in range(degree):
    
        for n, node in enumerate(list(nx.dfs_preorder_nodes(G))):
        
            if G.degree(node) == 2 and (n % 2) == 0:
            
                edges = list(G.edges(node))
        
                G.add_edge(edges[0][1], edges[1][1])
        
                G.remove_node(node)

    return G









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





def components(G):
    components = []
    for node in G:
        if G.nodes[node]['component'] not in components:
            components.append(G.nodes[node]['component'])
    return components





def similarity_between_graphs(G, H):
    
    G_sim = simplify(G, 10)
    H_sim = simplify(H, 10)
    
    components_G = sorted(components(G_sim))
    components_H = sorted(components(H_sim))
            
    matrix = np.zeros((len(components_G), len(components_H)))
    
    for n, c_G in enumerate(components_G):
        cc_G = select_component(G_sim, component=c_G)
            
        for m, c_H in enumerate(components_H):        
            cc_H = select_component(H_sim, component=c_H)
    
            matrix[n,m] = similarity_between_components(cc_G, cc_H)
    
    return matrix, components_G, components_H






            
    

# Compute connections from similarity
def similarity_to_connection(matrix, rows, columns, threshold):    
    alist = []    
    for n, row in enumerate(matrix):           
        if np.min(row) < threshold:        
            m = np.argmin(row)
            alist.append([rows[n], columns[m]])            
    return alist





    
    
    
    
    
    
    
    

## RELABEL
# Compute the maximum component in graph
def max_comp_match(H):
    value = 0
    for node in H:
        if H.nodes[node]['match'] == True:
            if H.nodes[node]['component'] > value:
                value = H.nodes[node]['component']
    return value


# Relabel graph based on connections
def relabel(H, connections, count):

    # Set all nodes to unmatched
    for node in H:
        H.nodes[node]['match'] = False
        
    # Match components based on connections
    for component, cc in enumerate(sorted(nx.connected_components(H))): 
            
        for connection in connections:
            
            if component == connection[1]:
                
                for node in cc:
                    
                    H.nodes[node]['component'] = connection[0]
                    H.nodes[node]['match'] = True
                    
                    
    # Extract targets
    targets = []
    for connection in connections:
        targets.append(connection[1])
                        
    # Relabel unmatchted components
    for component, cc in enumerate(sorted(nx.connected_components(H))):
     
        if component not in targets:
    
            count = max(max_comp_match(H), count)
            
            for node in cc:        
                H.nodes[node]['component'] = count + 1
                H.nodes[node]['match'] = True

    return H, count









