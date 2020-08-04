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

times = range(50000, 11200000, 50000)

# faults             = np.zeros(len(times))
# faults_connected   = np.zeros(len(times))




count = 0

for n, time in enumerate(times):

    # n = 0    
    # time = times[n]
    
    print(time)
    
    name_0 = str(times[n]).zfill(8)
    name_1 = str(times[n+1]).zfill(8)
    
    
    
    # data = genfromtxt('./csv/' + name_1 + '.csv', delimiter=',')
    # data = np.flip(data, axis=0)
    
    # vel_der             = data[:-1,2].reshape(324, 2500)
    # plastic_strain      = data[:-1,0].reshape(324, 2500)
    # strain_rate         = data[:-1,1].reshape(324, 2500)    
    
    
    # MAKE SURE TO COPY FIRST ONE TO NEW FOLDER
    G_0 = pickle.load(open('./graphs/correlated/' + name_0 + '.p', 'rb'))
        
    G_1 = pickle.load(open('./graphs/extracted/' + name_1 + '.p', 'rb'))
    
    
    
    # faults[n] = number_of_components(G_0)
    
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    plot_components(G_0, ax1)
    plot_components(G_1, ax2)
    plt.gca().invert_yaxis()
    plt.show()
    plt.savefig('./images/correlation/before/' + name_1 + '.png', dpi=100)
    plt.close("all")     
    
    
    factor = 1
    sm, rows, columns = similarity_between_graphs(G_0, G_1, factor)
    
    
    
    plot_matrix(sm, rows, columns)
    plt.savefig('./images/correlation/matrix/' + name_0 + '.png', dpi=100)
    plt.close("all")   
    
    
    
    threshold = 100
    connections = similarity_to_connection(sm, rows, columns, threshold)
    
    # print(connections)
    
    # faults_connected[n] = count
        
    G_1, count = relabel(G_1, connections, count)
    
           
    
    
    pickle.dump(G_1, open('./graphs/correlated/' + name_1 + '.p', "wb" )) 
    
    
    
    f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
    plot_components(G_0, ax1)
    plot_components(G_1, ax2)
    plt.gca().invert_yaxis()
    plt.show()
    plt.savefig('./images/correlation/after/' + name_1 + '.png', dpi=100)
    plt.close("all") 
    
    
    
    
    
#     #%% For Sankey diagram
    
#     # Sources and targets
#     for connection in connections:
#         sources.append(connection[0]) 
#         targets.append(connection[1] + len(sources))
    
    
#     # Labels
#     n_comp_0 = len(sorted(nx.connected_components(G_0)))
#     n_comp_1 = len(sorted(nx.connected_components(G_1)))
    
#     for m in range(n_comp_0):
#         labels.append('Fault_' + str(n) + '_' + str(m))
#     for m in range(n_comp_1):
#         labels.append('Fault_' + str(n+1) + '_' + str(m)) 
    
    
#     # Weights
#     weights = np.zeros((n_comp_0))
    
#     components_0 = [G_0.nodes[next(iter(c))]['component'] for c in sorted(nx.connected_components(G_0))]
    
#     for m, component in enumerate(components_0):
#         weights[m] = total_length(select_component(G_0, component=component))   
        
        
        
        
        
#     for connection in connections:    
#         index = components_0.index(connection[0])    
#         values.append(weights[index])
    
    
    
    





# pickle.dump(labels, open('label.p', "wb" ))
# pickle.dump(sources, open('source.p', "wb" ))
# pickle.dump(targets, open('target.p', "wb" ))
# pickle.dump(values, open('value.p', "wb" ))











    
    
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






