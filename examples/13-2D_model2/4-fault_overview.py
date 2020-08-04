import pickle
import matplotlib.pyplot as plt
plt.close("all")

import pandas as pd
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



times = range(0, 10400000, 200000)


count = 0

for n, time in enumerate(times):
    
    
    print(time)
    
    name = str(times[n]).zfill(7) 
    
    G = pickle.load(open('./graphs/correlated/graph_' + name + '.p', 'rb'))
    
    components = return_components(G)
    
    lengths = fault_lengths(G)
    
    max_strain = max_value_components(G, 'plastic_strain')
    
    max_strain_rate = max_value_components(G, 'strain_rate') 
    
    
    if n == 0:        
        faults    = np.empty((len(components),len(times),7))
        faults[:] = np.NaN
        
        
        
    faults[:len(components),n,0] = np.asarray(components)
    faults[:len(components),n,1] = np.asarray(lengths)
    faults[:len(components),n,2] = np.asarray(max_strain)       
    faults[:len(components),n,3] = np.asarray(max_strain_rate) 
    faults[:len(components),n,4] = np.asarray(sorted(lengths))
    faults[:len(components),n,5] = np.asarray(sorted(max_strain))
    faults[:len(components),n,6] = np.asarray(sorted(max_strain_rate))


#%%

fig, axs = plt.subplots(1, 4, sharey=True, figsize=(15,5)) 




      
im0 = axs[0].matshow(np.flipud(faults[:,:,0]))
axs[0].set_title('Faults')
axs[0].set_xlabel('Time [Myrs]')
axs[0].set_ylabel('Number of faults')

axs[0].set_xticklabels(np.asarray(times)/1e6)
axs[0].set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])

axs[0].xaxis.tick_bottom()


divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im0, cax=cax, orientation='vertical')
cbar.set_label('Fault ID')



    
im1 = axs[1].matshow(np.flipud(faults[:,:,6]))
axs[1].set_title('Max strain rate')
axs[1].set_xlabel('Time [Myrs]')
axs[1].set_ylabel('Number of faults')

axs[1].set_xticklabels(np.asarray(times)/1e6)

axs[1].xaxis.tick_bottom()


divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label('Max strain rate on faults')





im2 = axs[2].matshow(np.flipud(faults[:,:,5]))
axs[2].set_title('Max strain')
axs[2].set_xlabel('Time [Myrs]')
axs[2].set_ylabel('Number of faults')

axs[2].set_xticklabels(np.asarray(times)/1e6)

axs[2].xaxis.tick_bottom()


divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
cbar.set_label('Max strain on faults')




im3 = axs[3].matshow(np.flipud(faults[:,:,4]))
axs[3].set_title('Fault length')
axs[3].set_xlabel('Time [Myrs]')
axs[3].set_ylabel('Number of faults')

axs[3].set_xticklabels(np.asarray(times)/1e6)

axs[3].xaxis.tick_bottom()


divider = make_axes_locatable(axs[3])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
cbar.set_label('Fault length [km]')

plt.subplots_adjust(wspace=0.5, hspace=0)
plt.show()





#%%

plt.close('all')


from time import sleep


n = 0

indeces = np.where(faults[:,:,0] == n)


lengths = faults[indeces[0],indeces[1],1]
strain_rates = faults[indeces[0],indeces[1],3]

plt.figure()

for n in range(40):

    plt.plot(lengths[:n], strain_rates[:n], 'black')
    plt.xlim(np.min(lengths), np.max(lengths))
    plt.ylim(np.min(strain_rates), np.max(strain_rates))
    
    plt.savefig('./images/DL/0/' + str(n) + '.png')
    
    
    


    
    
plt.show()



























#%%
# 

# t_0 = 10
# t_1 = 20
# t_2 = 30
# t_3 = 40

# def sankey(faults, t_0, t_1):
    
    
#     # Define labels
#     labels_start = np.nonzero(faults[:,t_0,0])[0]
#     times_start = np.ones_like(labels_start) * t_0
    
#     labels_end = np.nonzero(faults[:,t_1,0])[0]
#     times_end  = np.ones_like(labels_end) * t_1
        
#     labels = []
    
#     for label in labels_start:    
#         labels.append((t_0, label))
    
#     for label in labels_end:    
#         labels.append((t_1, label))
    
    
   
#     # Define connections (i.e. sources, targets, flows)
#     connections = sorted(list(set(labels_start) & set(labels_end)))    
    
#     sources = []
#     targets = []



    
#     for connection in connections:
        
#         for n, label in enumerate(labels):
#             if label == (t_0, connection):
#                 sources.append(n)
                

    
#         for n, label in enumerate(labels):
#             if label == (t_1, connection):
#                 targets.append(n)   

    
#     flows = list(faults[labels_start,t_0,2])
    
    
    
    
#     X = []
#     Y = []
    
#     # Define coordinates (only for nodes wiht connections)
#     for n, label in enumerate(labels):
#         if label[0] == t_0 and label[1] in sources:
#             X.append(0)
#             Y.append(n)
            
#     for n, label in enumerate(labels):           
#         if label[0] == t_1 and label[1] in targets:
#             X.append(1)
#             Y.append(n)


#     return labels, X, Y, sources, targets, flows







# labels0, X0, Y0, sources0, targets0, flows0 = sankey(faults, t_0, t_1)
# labels1, X1, Y1, sources1, targets1, flows1 = sankey(faults, t_1, t_2)
# # labels2, X2, Y2, sources2, targets2, flows2 = sankey(faults, t_2, t_3)




# def merge_two_sankeys(labels0, X0, Y0, sources0, targets0, flows0, labels1, X1, Y1, sources1, targets1, flows1):

#     # Merge lists of labels
#     labels = sorted(list(set(labels0) | set(labels1)))
    
#     # Update x-coordinate
#     X = []
#     for x in X0:
#         if x == 0:
#             X.append(0)
#         if x == 1:
#             X.append(0.5)
                       
#     for x in X1:
#         if x == 1:
#             X.append(1)
    
#     Y = Y0 + Y1
         
    
    
    
#     # Use connections from first set
#     sources = sources0
#     targets = targets0
#     flows   = flows0
    
#     # Find starting point N
#     for n, label in enumerate(labels0):
#         if label in labels1:
#             N = n
#             break
    
#     # Correct connections of second set
#     for m in range(len(sources1)):        
#         sources.append(sources1[m] + N)
#         targets.append(targets1[m] + N)
    
#     # Join flows    
#     flows = flows + flows1
    
    
    
    
    
    

#     return labels, X, Y, sources, targets, flows






# labels, X, Y, sources, targets, flows = merge_two_sankeys(labels0, X0, Y0, sources0, targets0, flows0, labels1, X1, Y1, sources1, targets1, flows1)
# # labels, sources, targets, flows = merge_two_sankeys(labels, sources, targets, flows, labels2, sources2, targets2, flows2)






# pickle.dump(labels, open('./sankey/labels.p', "wb" ))
# pickle.dump(X, open('./sankey/X.p', "wb" ))
# pickle.dump(Y, open('./sankey/Y.p', "wb" ))
# pickle.dump(sources, open('./sankey/sources.p', "wb" ))
# pickle.dump(targets, open('./sankey/targets.p', "wb" ))
# pickle.dump(flows, open('./sankey/flows.p', "wb" ))



            














