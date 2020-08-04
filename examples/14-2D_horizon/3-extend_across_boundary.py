import pickle
import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt
plt.close("all")


# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from image_processing import *
from edits import *
from metrics import *
from plots import *




# %% LOAD DATA

for time in range(11200000, 0, -50000):
        
    # time = 11200000
    
    print(time)
    
    name = str(time).zfill(8)
    
    G = pickle.load(open('./graphs/extracted/' + name + '.p', 'rb'))
        
    
    
    
# %% CALCULATE COORDINATES
    
    x_pix = 5000
    y_pix = 1600
    
    x_max = 500 #km
    y_max = 160 #km
    
    for node in G:
        G.nodes[node]['x'] = G.nodes[node]['pos'][0] * 0.1
        G.nodes[node]['y'] = G.nodes[node]['pos'][1] * 0.1
        
        
# %% CORRELATE ACROSS BOUNDARY
    
    G_exp = expand_network(G, relabel=False)
        
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    plot_components(G_exp, label=True, ax = ax)
    plt.savefig('./images/extended/' + str(name) + '.png', dpi=300)
    plt.close("all")
    
    
    # Pickle graph
    pickle.dump(G, open('./graphs/extended/' + name + '.p', "wb" ))