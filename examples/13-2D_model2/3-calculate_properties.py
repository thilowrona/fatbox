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

    # time = 800000
    
    print('time ' + str(time))
    
    name = str(time).zfill(7)
    
    count = 0
    
    
    data = genfromtxt('./csv/' + name + '.csv', delimiter=',')
    
    data = np.flip(data, axis=0)
    
    vel_der             = data[:-1,2].reshape(324, 2500)
    # plastic_strain      = data[:-1,0].reshape(324, 2500)
    # strain_rate         = data[:-1,1].reshape(324, 2500)
    
    
    
    # plt.imshow(vel_der)
    
    
    # THRESHOLDING
    threshold = vel_der.clip(min=0)
    
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
    G = nx.Graph()
    
    # Add nodes
    for n in range(N):
        G.add_node(n, 
                    pos=(int(points[n,1]), int(points[n,0]))
                    )
    
    # Add edges to graph
    G = add_edges(G, N)
    
    G = count_edges(G)
    G = label_components(G)
    
    count = len(list(G.nodes))
    
    
    # Connect componnts   
    G = connect_close_components(G, 10)
            
        
        
        
        
        
        
        
        
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
        H.add_node(n + count, 
                    pos=(int(points[n,1]), int(points[n,0]))
                    )
    
    # Add edges to graph
    H = add_edges(H, N)
    
    
    H = count_edges(H)  
    H = label_components(H)    
       
    # Connect close components   
    H = connect_close_components(H, 10)
                
    # Merge graphs
    F = nx.compose(G, H)
    
    F = count_edges(F)  
    F = label_components(F)     
    
    

    
    
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    p = plt.imshow(vel_der)
    plot_components(F, ax, filename='./images/overlays/overlay_' + name + '.png')     
    
    # # Color bar locator
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="3%", pad=0.15)    
    # cb0 = fig.colorbar(p, ax=ax, cax=cax)
           
    plt.close("all")
       
        
        
        
    # Pickle graph
    pickle.dump(F, open('./graphs/graph_' + name + '.p', "wb" ))








