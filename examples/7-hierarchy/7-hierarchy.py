import networkx as nx
import pandas as pd
import numpy as np

import pickle


# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from metrics import *
from plots import *
from utils import *


import matplotlib.pyplot as plt
plt.close("all")

G = pickle.load(open("graph.p", 'rb'))


                  
plot_components(G)

noc = number_of_components(G)

components = range(noc)


lengths = fault_lengths(G)



n_comp = 1000        
palette = sns.color_palette(None, 2*n_comp)
color = np.zeros((noc,3))

plt.figure()
for n in range(noc):
    plt.plot([components[n],components[n]],[0,lengths[n]], color=palette[n])
    plt.text(components[n]-0.5,lengths[n]+10, components[n], fontsize = 15, color = palette[n])
    
    
plt.xlabel('Component')
plt.ylabel('Length')


