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



count = 0

for n, time in enumerate(times):
        
    # n = 4    
    
    time = times[n]
    
    print(time)
    
    name_0 = str(times[n]).zfill(7)
    name_1 = str(times[n+1]).zfill(7)
    
    
    
    data = genfromtxt('./csv/others/' + name_1 + '.csv', delimiter=',')
    
    data = np.flip(data, axis=0)
    
    # plastic_strain = data[:-1,0].reshape(600, 2500)
    strain_rate    = data[:-1,1].reshape(600, 3000)  
    
    
    # MAKE SURE TO COPY FIRST ONE TO NEW FOLDER
    if n == 0:        
        G_0 = pickle.load(open('./graphs/extracted/graph_' + name_0 + '.p', 'rb'))
        pickle.dump(G_0, open('./graphs/correlated/graph_' + name_0 + '.p', "wb" ))
        
        
        
    
    G_0 = pickle.load(open('./graphs/correlated/graph_' + name_0 + '.p', 'rb'))        
    G_1 = pickle.load(open('./graphs/extracted/graph_' + name_1 + '.p', 'rb'))
    
    
    
    
    
    
    matrix = hausdorff_distance(G_0, G_1)
    
    
    
    
       
    threshold = 0.1
    rows      = get_components(G_0)
    columns   = get_components(G_1)
    connections = similarity_to_connection(matrix, rows, columns, threshold)
    
    # plot_matrix(matrix, rows, columns)
    # plt.savefig('./images/correlation/matrix/' + name_1 + '.png', dpi=200)
    
    
    
    
    f, axs = plt.subplots(3, 1, figsize=(16,8), sharey=True)
    
    
    axs[0].set_title('G_1 with original labels')
    axs[0].matshow(strain_rate, cmap='gray_r')
    plot_components(G_1, axs[0])
    
    
    
    
    
    G_1, count = relabel(G_1, connections, count)
    
    
    
    
    
            
    axs[1].matshow(strain_rate, cmap='gray_r')
    axs[1].set_title('G_1 with relabelled labels')
    plot_components(G_1, axs[1])
      
    
    
    
    axs[2].matshow(strain_rate, cmap='gray_r')
    axs[2].set_title('G_0 with original labels')
    plot_components(G_0, axs[2])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    cc = common_components(G_0, G_1)
    
    G_0_common = select_components(G_0, cc)
    G_1_common = select_components(G_1, cc)
    
    
    
    (c_0_unique, c_1_unique) = unique_components(G_0, G_1)
    
    G_0_unique = select_components(G_0, c_0_unique)
    G_1_unique = select_components(G_1, c_1_unique)
    
    
    
    
    
    f, axs = plt.subplots(3, 1, figsize=(16,8), sharey=True)
    
    axs[0].matshow(strain_rate, cmap='gray_r')
    plot(G_0, axs[0], color='red', with_labels=False)
    plot(G_1, axs[0], color='blue', with_labels=False)
    axs[0].set_title('Both')
    
    
    axs[1].matshow(strain_rate, cmap='gray_r')
    plot_components(G_0_common, axs[1])
    plot_components(G_1_common, axs[1])
    axs[1].set_title('Matched')
    
    
    axs[2].matshow(strain_rate, cmap='gray_r')
    plot_components(G_0_unique, axs[2])
    plot_components(G_1_unique, axs[2])
    axs[2].set_title('Unmatched')
    
    plt.savefig('./images/correlation/match/' + name_1 + '.png', dpi=200)
    # plt.close("all") 
    






    f, ax = plt.subplots(1, 1, figsize=(12,4))
    ax.matshow(strain_rate, cmap='gray_r')
    plot_components(G_1, ax)
    
    plt.show()
    plt.savefig('./images/overlays/correlated/' + name_1 + '.png', dpi=200)
    plt.close("all") 





    
    
    pickle.dump(G_1, open('./graphs/correlated/graph_' + name_1 + '.p', "wb" )) 
    




































# f, (ax1, ax2) = plt.subplots(2, 1, figsize=(20,20), sharey=True)
# plot_components(G_0, ax1)
# plot_components(G_1, ax2)
# plt.gca().invert_yaxis()
# plt.show()
# plt.savefig('./images/correlation/after/' + name_1 + '.png', dpi=200)










    



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






