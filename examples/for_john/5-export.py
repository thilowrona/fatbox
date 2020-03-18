import networkx as nx
import pandas as pd
import numpy as np

import pickle


# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from metrics import *
from utils import *


filename = "v5w4h60_z95_20MYR"

G = pickle.load(open("./graphs/graph_" + filename + ".p", 'rb'))

#G = simplify(G, 5)

highest_node = max(list(G.nodes))+1






components = [0]*(highest_node)
for node in G:
    components[node] = G.nodes[node]['component']




strain_rate = [0]*(highest_node)
for node in G:
    strain_rate[node] = G.nodes[node]['strain_rate']



plastic_strain = [0]*(highest_node)
for node in G:
    plastic_strain[node] = G.nodes[node]['plastic_strain']







writeObjects(G,
             fileout='./paraview/graph_' + filename,
             scalar=components, 
             name='components',
             scalar2=strain_rate,
             name2='strain_rate',
             scalar3=plastic_strain,
             name3='plastic_strain'
             )



