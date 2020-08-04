import ngtpy
import random

# dim = 10
# objects = []
# for i in range(0, 100) :
#     vector = random.sample(range(100), dim)
#     objects.append(vector)

# query = objects[0]

# ngtpy.create(b"tmp", dim)
# index = ngtpy.Index(b"tmp")
# index.batch_insert(objects)
# index.save()

# result = index.search(query, 3)

# for i, o in enumerate(result) :
#     print(str(i) + ": " + str(o[0]) + ", " + str(o[1]))
#     object = index.get_object(o[0])
#     print(object)
    
    
    
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


    
time = 0

name = str(time).zfill(7)

G = pickle.load(open('./graphs/extracted/graph_' + name + '.p', 'rb'))





H = nx.create_empty_copy(G)

# import ngtpy

# def add_edges_fast(G, distance, max_conn):

#     for n, node in enumerate(H):
        
#         query = objects[n]
        
#         neighbors = index.search(query, max_conn)
           
#         for neighbor in neighbors:
#             if neighbor[1] < distance:
#                 G.add_edge(node, neighbor[0])

#     return G




H = nx.create_empty_copy(G)

H = add_edges_fast(H, dim=2, distance=10, max_conn=3)

H = label_components(H) 

fig, ax = plt.subplots(1, 1, figsize=(12,4))
plot_components(H, ax)


