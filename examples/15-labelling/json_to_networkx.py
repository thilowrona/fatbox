import matplotlib.pyplot as plt
plt.close("all")

from matplotlib.image import imread

import numpy as np
from numpy import genfromtxt
import meshio
import networkx as nx
import pickle



img = imread('outcrop.jpg')


fig, axs = plt.subplots(1, 1, figsize=(10,8))
axs.imshow(img)





import json

# read file
with open('polygons.json', 'r') as myfile:
    data=myfile.read()

# parse file
obj = json.loads(data)


group_id = set()

shapes = obj['shapes']
for shape in shapes:
    group_id.add(shape['group_id'])




G = nx.Graph()


node = 0
for shape in shapes:
    
    for group in group_id:
                
        points = shape['points']
        
        for m in range(len(points)):
            
            position = (int(points[m][0]), int(points[m][1]))
            
                
            G.add_node(node)
            G.nodes[node]['pos'] = position
    
            if m < len(points)-1:
                G.add_edge(node, node+1)                
            node += 1












import math

H = G.copy()

threshold = 25

for node in G:
    for another in G:
        
        if H.has_node(node) and H.has_node(another) and node != another:
        
        
            if math.sqrt((G.nodes[node]['pos'][0] - G.nodes[another]['pos'][0])**2  + 
                         (G.nodes[node]['pos'][1] - G.nodes[another]['pos'][1])**2) < threshold:
                
                neighbors = H.neighbors(another)
                
                for neighbor in neighbors:
                    H.add_edge(node, neighbor)
                
                H.remove_node(another)
            
            
            


            
    




nx.draw(H,
        pos = nx.get_node_attributes(H, 'pos'),
        node_size = 1,
        with_labels = True,
        ax = axs)







print(H.edges)







