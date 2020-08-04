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
max_strain           = np.zeros(len(times))


count = 0

for n, time in enumerate(times):
    
    
    
    print(time)
    
    name = str(times[n]).zfill(7)

    
    data = genfromtxt('./csv/' + name + '.csv', delimiter=',')
    data = np.flip(data, axis=0)
    
    # vel_der             = data[:-1,2].reshape(324, 2500)
    plastic_strain      = data[:-1,0].reshape(324, 2500)
    # strain_rate         = data[:-1,1].reshape(324, 2500)    
    
    
    G = pickle.load(open('./graphs/extracted/graph_' + name + '.p', 'rb'))
        
    G = extract_attribute(G, plastic_strain, 'plastic_strain')
    
    n_comp = number_of_components(G)
    
    heights    = np.zeros(n_comp)
    strain_max = np.zeros(n_comp)
    
    for n in range(n_comp):
        heights[n] = total_length2(select_component(G, component=n))
        strain_max[n] = max_value_nodes(select_component(G, component=n), 'plastic_strain')


    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    plt.scatter(heights, strain_max)
    plt.xlim([0, 50])
    plt.ylim([0, 5])
    plt.xlabel('Fault heights [km]')
    plt.ylabel('Maximum strain')
    plt.savefig('./images/DL/' + name + '.png', dpi=100)
    plt.close("all") 

    max_strain[n] = np.max(plastic_strain)


#%%

    
fig, ax = plt.subplots(1, 1, figsize=(8,8))
plt.scatter(times, max_strain)
plt.xlabel('Times [Myrs]')
plt.ylabel('Maximum strain')






