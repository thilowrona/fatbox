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

from image_processing import *
from edits import *
from metrics import *
from plots import *




for time in range(0, 10400000, 200000):
    
    # time = 1600000
    
    print('time ' + str(time))
    
    name = str(time).zfill(7)
    
    
    
    data = genfromtxt('./csv/vel_der/' + name + '.csv', delimiter=',')
    
    data = np.flip(data, axis=0)
    
    vel_der = data[:-1,0].reshape(600, 3000)
    
        
    
    # plt.imshow(vel_der)
    # plt.colorbar()
    
    
    # THRESHOLDING
    threshold = vel_der.clip(min=0)
    
    value = 2e-7
    threshold = simple_threshold_binary(threshold, value = value)
    # threshold = adaptive_threshold(data)
    
     
    threshold = remove_small_regions(threshold, 10)
       
    # plot_threshold(vel_der, threshold, value)
    
    
    
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
        G.nodes[n]['polarity'] = 0
    
    # Add edges to graph
    G = add_edges_fast(G, dim=2, distance=10, max_conn=3)
    
    G = count_edges(G)
    G = label_components(G)
    
    G = remove_small_components(G, minimum_size = 10)
    
    
    
            
    
        
        
        
        
        
        
    # THRESHOLDING
    threshold = -vel_der
    threshold = threshold.clip(min=0)
    
    value = 2e-7
    threshold = simple_threshold_binary(threshold, value = value)
    # threshold = adaptive_threshold(data)
        
    threshold = remove_small_regions(threshold, 10)
    
    # plot_threshold(data, threshold, value, './threshold/threshold_' + name + '.png')
    # plt.close("all")   
        
        
    ## SKELETONIZE
    skeleton = guo_hall(threshold)
    # plot_comparison([data, threshold, skeleton])
    
    
    # CONVERT TO POINTS
    points = np_to_points(skeleton)
    
    
    N = points.shape[0]
    
    # Set up graph
    H = nx.Graph()
    
    # Add nodes
    for n in range(N):
        H.add_node(n, 
                    pos=(int(points[n,1]), int(points[n,0]))
                    )
        H.nodes[n]['polarity'] = 1
    
    
    # Add edges to graph
    H = add_edges_fast(H, dim=2, distance=10, max_conn=3)
    
    
    H = count_edges(H)  
    H = label_components(H)    
       
    # Remove small components
    H = remove_small_components(H, minimum_size = 10)
    
    
                
    # Merge graphs   
    highest_node = max(list(G.nodes)) + 1
    nodes_new = [node + highest_node for node in H.nodes]
    
    mapping = dict(zip(H.nodes, nodes_new))
    H = nx.relabel_nodes(H, mapping)
        
    F = nx.compose(G, H)
    
    
    
    F = count_edges(F)  
    F = label_components(F)     
    
    
    
      
        
        
        
    
    
    
    #%% Extract and calculate fault properties
    
    data = genfromtxt('./csv/others/' + name + '.csv', delimiter=',')
    
    data = np.flip(data, axis=0)
    
    plastic_strain = data[:-1,0].reshape(600, 3000)
    strain_rate    = data[:-1,1].reshape(600, 3000)
    
    # plt.imshow(strain_rate)
    # plt.colorbar()    
    
    
    F = extract_attribute(F, plastic_strain, 'plastic_strain')
    F = extract_attribute(F, strain_rate, 'strain_rate')
    
    
    factor_x = 1
    factor_y = 1
    
    
    for node in F:
        F.nodes[node]['x'] = F.nodes[node]['pos'][0] * factor_x
        F.nodes[node]['y'] = F.nodes[node]['pos'][1] * factor_y
    
    F = compute_edge_length(F)
    
        
        
        
    
    #%%  Plot
    fig, axs = plt.subplots(2, 1, figsize=(16,8))
    
    p = axs[0].matshow(strain_rate, cmap='gray_r') 

    # Color bar locator
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    cb0 = fig.colorbar(p, ax=axs[0], cax=cax)






    
    p = axs[1].matshow(strain_rate, cmap='gray_r')     
    plot_components(F, axs[1])
    
    # Color bar locator
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    cb0 = fig.colorbar(p, ax=axs[1], cax=cax)
          
    
    plt.savefig(fname='./images/overlays/extracted/image_' + name + '.png', dpi=200)
    plt.close("all")
    
    
    
    
    # Pickle graph
    pickle.dump(F, open('./graphs/extracted/graph_' + name + '.p', "wb" ))
    
    
    


