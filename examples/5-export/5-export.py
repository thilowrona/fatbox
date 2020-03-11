import networkx as nx
import pandas as pd
import numpy as np

import pickle


# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from metrics import *
from utils import *



G = pickle.load(open("graph.p", 'rb'))









components = [0]*(highest_node)
for node in G:
    components[node] = G.nodes[node]['component']




strain_rate = [0]*(highest_node)
for node in G:
    strain_rate[node] = G.nodes[node]['strain_rate']











writeObjects(G,
             fileout='graph',
             scalar=components, 
             name='components',
             scalar2=strain_rate,
             name2='strain_rate'
             )



