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

    name = str(time).zfill(8)
    
    data = genfromtxt('./csv/' + name + '.csv', delimiter=',')
    
    data = np.flip(data, axis=0)
    
    # strain = data[:-1,0].reshape(1600, 5000)
    strain_rate = data[:-1,1].reshape(1600, 5000)



    # # Plot data
    # fig, ax = plt.subplots(1, 1, figsize=(12,4))
    
    # p = plt.imshow(strain_rate)
    
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.15)    
    # cb0 = fig.colorbar(p, ax=ax, cax=cax)





# %% THRESHOLDING

    value = 2e-14
    threshold = simple_threshold_binary(strain_rate, value = value)
    # threshold = adaptive_threshold(strain_rate)
    
     
    threshold = remove_small_regions(threshold, 10)
       
    # plot_threshold(strain_rate, threshold, value)
    # plt.close("all")
    
    
    ## SKELETONIZE
    skeleton = guo_hall(threshold)
    # plot_comparison([data, threshold, skeleton])
    
    
    # CONVERT TO POINTS
    points = np_to_points(skeleton)
    
    
    N = points.shape[0]
    
    # Set up graph
    G = nx.Graph()
    
    # Add nodes
    for n in range(N):
        G.add_node(n, 
                    pos=(int(points[n,1]), int(points[n,0]))
                    )
    
    # Add edges to graph
    G = add_edges(G, N)
    
    # Label faults
    G = label_components(G) 

    # Add coordinates
    # x_pix = 5000
    # y_pix = 1600
    
    # x_max = 500 #km
    # y_max = 160 #km
    
    for node in G:
        G.nodes[node]['x'] = G.nodes[node]['pos'][0] * 0.1
        G.nodes[node]['y'] = G.nodes[node]['pos'][1] * 0.1


    # Pickle graph
    pickle.dump(G, open('./graphs/extracted/' + name + '.p', "wb" ))



# %% PLOTTING

    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    
    p = plt.imshow(strain_rate)
    
    plot_components(G, ax) 
    
    # Color bar locator
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    cb0 = fig.colorbar(p, ax=ax, cax=cax)
    cb0.ax.plot([-1, 1], [value]*2, 'r')    
    
    
    plt.savefig('./images/overlays/' + str(name) + '.png', dpi=300)
    
    plt.close('all')






