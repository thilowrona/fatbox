import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

import seaborn as sns


import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from metrics import *
from utils import *

















def plot_components(G, label=True, box=True, ax = []):
    
    if ax == []:
        fig, ax = plt.subplots()
        
    n_comp = 1000
            
    palette = sns.color_palette(None, 2*n_comp)
    
    node_color = np.zeros((len(G),3))
    
    for n, node in enumerate(G):
        color = palette[G.nodes[node]['component']]
            
        node_color[n,0] = color[0]
        node_color[n,1] = color[1]
        node_color[n,2] = color[2]   
            
    
    nx.draw(G,
            pos = nx.get_node_attributes(G, 'pos'),
            node_color = node_color,
            node_size = 1,
            ax = ax)
    
    plt.axis('equal')
    
    
    
    
    if label == True:
        
        for cc in sorted(nx.connected_components(G)): 
            # Calculate centre
            x_avg = 0
            y_avg = 0
            
            for n in cc:        
                y_avg = y_avg + G.nodes[n]['pos'][0]
                x_avg = x_avg + G.nodes[n]['pos'][1]
            
            N = len(cc)
            y_avg = y_avg/N
            x_avg = x_avg/N
            
            # Scale color map
            label = G.nodes[n]['component']
            
            ax.text(y_avg, x_avg, label, fontsize = 15, color = palette[G.nodes[n]['component']])
      
        
    if box == True:
        left, bottom, width, height = (0, 0, 704, 960)
        
        rect = patches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor='black', facecolor='none')
        
        ax.add_patch(rect)







def plot_attribute(G, attribute, box=True, ax=[]):
    
    if ax == []:
        fig, ax = plt.subplots()
    
    
    ax = plt.gca()
    
    nx.draw(G,
            pos = nx.get_node_attributes(G, 'pos'),
            node_color = get_labels(G, attribute),
            node_size = 2,
            ax=ax)
    
    # Colorbar
    cmap = plt.cm.viridis
    vmax = max_value_nodes(G, attribute)
    vmin = min_value_nodes(G, attribute)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
            
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(attribute, rotation=270)
    plt.axis('equal')



    if box == True:
        left, bottom, width, height = (0, 0, 704, 960)
        
        rect = patches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor='black', facecolor='none')
        
        ax.add_patch(rect)







def plot_rose(G):
    
    strikes = np.zeros(len(G.edges))
    lengths = np.zeros(len(G.edges))
    
    for n, edge in enumerate(G.edges):
        strikes[n] = G.edges[edge]['strike']
        lengths[n] = G.edges[edge]['length']
    
         
    ## ROSE PLOT
    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(strikes, bin_edges, weights = lengths)           
    number_of_strikes[0] += number_of_strikes[-1]
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])
    
     
    
    fig = plt.figure(figsize=(8,8))
        
    ax = fig.add_subplot(111, projection='polar')
    
    ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, 
           width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
#    ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight= 'black')
    ax.set_title('Rose Diagram', y=1.10, fontsize=15)
    
    fig.tight_layout()










def cross_plot(G, var0, var1):
    
    x = np.zeros(len(G.nodes))
    y = np.zeros(len(G.nodes))
    
    
    if var0 == 'x':
        for n, node in enumerate(G):        
            x[n] = G.nodes[node]['pos'][1]
            y[n] = G.nodes[node][var1]

    if var0 == 'y':
        for n, node in enumerate(G):        
            x[n] = G.nodes[node]['pos'][0]
            y[n] = G.nodes[node][var1]        

    if var1 == 'x':
        for n, node in enumerate(G):        
            x[n] = G.nodes[node][var0]
            y[n] = G.nodes[node]['pos'][1]

    if var1 == 'y':
        for n, node in enumerate(G):        
            x[n] = G.nodes[node][var0]
            y[n] = G.nodes[node]['pos'][0] 
    
    plt.plot(x, y, '.')





