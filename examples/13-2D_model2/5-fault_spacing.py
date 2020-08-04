import pickle
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

from edits import *
from metrics import *
from plots import *

times = range(0, 10400000, 200000)

# faults             = np.zeros(len(times))
# faults_connected   = np.zeros(len(times))


count = 0

for n, time in enumerate(times):
    
    print(time)
    
    name = str(times[n]).zfill(7)

    
    G = pickle.load(open('./graphs/extracted/graph_' + name + '.p', 'rb'))
        
    n_comp = number_of_components(G)
  
    
  
    matrix_min = np.zeros((n_comp, n_comp))
    matrix_max = np.zeros((n_comp, n_comp))
    for n0, cc0 in enumerate(sorted(nx.connected_components(G))):
        for n1, cc1 in enumerate(sorted(nx.connected_components(G))):
            matrix_min[n0, n1] = min_dist_comp(G, cc0, cc1) * 0.1
            matrix_max[n0, n1] = max_dist_comp(G, cc0, cc1) * 0.1
            

    dist_min = np.zeros(n_comp)
    dist_max = np.zeros(n_comp)
    
    for x in range(n_comp):
        m0 = 1e6
        for y in range(n_comp):
            
            if 0 < matrix_min[x,y] < m0:
                m0 = matrix_min[x,y]
                y0 = y
                
        dist_min[x] = matrix_min[x,y0]
        dist_max[x] = matrix_max[x,y0]
    
    dist_mean = (dist_min + dist_max)/2
    
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,15))
    axes[0].hist(dist_min, density = False, bins=30)
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlabel('Min distance [km]')
    
    axes[1].hist(dist_max, density = False, bins=30)
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlabel('Max distance [km]')
    
    axes[2].hist(dist_mean, density = False, bins=30)
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlabel('Mean distance [km]')   
    plt.savefig('./images/spacing/' + name + '.png', dpi=300)
    plt.close("all") 
 