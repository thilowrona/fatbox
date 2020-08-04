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

time_0 = 0
time_1 = 600000




x0 = []
y0 = []
sources0 = []
targets0 = []
values0  = [] 
labels0  = []


  


name_0 = str(time_0).zfill(7)
name_1 = str(time_1).zfill(7)


# MAKE SURE TO COPY FIRST ONE TO NEW FOLDER
G_0 = pickle.load(open('./graphs/correlated/graph_' + name_0 + '.p', 'rb'))        
G_1 = pickle.load(open('./graphs/correlated/graph_' + name_1 + '.p', 'rb'))


components_0 = return_components(G_0)
components_1 = return_components(G_1)


noc_0 = len(components_0)
noc_1 = len(components_1)


table = np.zeros((noc_0+noc_1, 5))


x = 0

for n, comp in enumerate(components_0):
    
    table[n,0] = time_0
    table[n,1] = comp
    table[n,2] = (time_0, comp)
    table[n,3] = 0
    table[n,4] = x
    x += 0.1





# 
#     labels0.append((time_0, comp))
#     x0.append(0)
#     y0.append(m) 
    
# for m, comp in enumerate(components_1):
#     labels0.append((time_1, comp)) 
#     x0.append(0.5)
#     y0.append(m)      
    


# connections = list(set(components_0).intersection(components_1))


# for connection in connections:
#     for m, label in enumerate(labels0):
#         if label[1] == connection:
#             sources0.append(m)
#             targets0.append(m + len(components_0))
#             break
    
    


# # Weights
# weights = np.zeros((len(components_0)))

# for m, component in enumerate(components_0):
#     weights[m] = total_length(select_component(G_0, component=component))   
    
    
    
# for connection in connections:    
#     index = components_0.index(connection)    
#     values0.append(weights[index])
    
    






# pickle.dump(x, open('./sankey/x_'            + str(name_0) + '-' + str(name_1) + '.p', "wb" ))
# pickle.dump(y, open('./sankey/y_'            + str(name_0) + '-' + str(name_1) + '.p', "wb" ))
# pickle.dump(labels, open('./sankey/label_'   + str(name_0) + '-' + str(name_1) + '.p', "wb" ))

# pickle.dump(sources, open('./sankey/source_' + str(name_0) + '-' + str(name_1) + '.p', "wb" ))
# pickle.dump(targets, open('./sankey/target_' + str(name_0) + '-' + str(name_1) + '.p', "wb" ))
# pickle.dump(values, open('./sankey/value_'   + str(name_0) + '-' + str(name_1) + '.p', "wb" ))














    
    
    # count = 0
    
    
    # data = genfromtxt('./csv/' + name + '.csv', delimiter=',')
    
    # data = np.flip(data, axis=0)
    
    # vel_der        = data[:-1,2].reshape(324, 2500)
    # plastic_strain = data[:-1,0].reshape(324, 2500)
    # strain_rate    = data[:-1,1].reshape(324, 2500)
    
    
    
    
    # fig, ax = plt.subplots(1, 1, figsize=(12,4))
    # ax.imshow(plastic_strain)
    # plot_components(G, ax)     
    
    
    
    
    # G = extract_attribute(G, plastic_strain, 'plastic_strain')
    
    
    
    
    # fig, ax = plt.subplots(1, 1, figsize=(8,10))
    # plot_attribute(G, 'plastic_strain', ax=ax)
    
    
    
    

    
    
    
    
    
    
    
    # strain_max  = np.zeros(n_comp)
    # strain_mean = np.zeros(n_comp)
    
    
    # for n in range(n_comp):
    #     strain_max[n]  = max_value_nodes(select_component(G, component=n), 'plastic_strain')
    #     strain_mean[n] = mean_value_nodes(select_component(G, component=n), 'plastic_strain')
    
    
    
    # fig = plt.figure()
    # plt.hist(strain_max, density=False, bins=30)
    # plt.xlabel('Maximum strain')
    # plt.ylabel('Frequency')
    # plt.savefig('./images/histogram/max/' + name + '.png', dpi=300)
    
    
    
    # fig = plt.figure()
    # plt.hist(strain_mean, density=False, bins=30)
    # plt.xlabel('Mean strain')
    # plt.ylabel('Frequency')
    # plt.savefig('./images/histogram/mean/' + name + '.png', dpi=300)
    
    
    # plt.close("all")




#%%
# faults_new   = np.zeros(len(times))
# faults_dying = np.zeros(len(times))

# for n in range(len(times)):
#     if n == 0:
#         faults_new[n]   = faults[n]
#         faults_dying[n] = 0
#     else:
#         faults_new[n]     = faults[n] - faults_connected[n-1]
#         faults_dying[n-1] = faults[n-1] - faults_connected[n]


# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5,15))
# axes[0].scatter(times, faults)
# axes[0].set_xlabel('Time [Myrs]')
# axes[0].set_ylabel('Number of faults')

# axes[1].scatter(times, faults_connected)
# axes[1].set_xlabel('Time [Myrs]')
# axes[1].set_ylabel('Number of new faults')






