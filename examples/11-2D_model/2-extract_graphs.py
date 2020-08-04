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




for time in range(0,13000000,500000):
    
    # time = 1500000
    
    print('time ' + str(time))
    
    name = str(time).zfill(7)
    
    count = 0
    
    
    data = genfromtxt('./csv/time_' + name + '.csv', delimiter=',')
    
    data = data[:-1,0].reshape(1000, 3000)
    
    data = np.flip(data, axis=0) 
    
        
    plt.imshow(data)
    
        
    # THRESHOLDING
    threshold = data.clip(min=0)
    
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
    
    
    count = len(list(G.nodes))
    
    
        
        
        
        
        
        
        
        
        
        
        
    # THRESHOLDING
    threshold = -data
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
    
    
    # Merge graphs
    F = nx.compose(G, H)
    
    
    
    
    F = count_edges(F)  
    F = label_components(F)
    
    
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12,4))
    p = plt.imshow(data)
    plot_components(F, ax, filename='./overlays/image_' + name + '.png')     
    
    # Color bar locator
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    cb0 = fig.colorbar(p, ax=ax, cax=cax)
        
        
        
        
        
        
    # Pickle graph
    pickle.dump(F, open('./graphs/g_' + name + '.p', "wb" ))








