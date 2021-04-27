import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

import seaborn as sns


import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from metrics import *
from utils import *


from matplotlib import colors


cmap = colors.ListedColormap(['#ffffffff', '#64b845ff', '#9dcd39ff', '#efe81fff', '#f68c1bff', '#f01b23ff'])




def get_node_colors(G, attribute):

    n_comp = 10000
            
    palette = sns.color_palette(None, 2*n_comp)
    
    node_color = np.zeros((len(G),3))
    
    for n, node in enumerate(G):
        color = palette[G.nodes[node][attribute]]
            
        node_color[n,0] = color[0]
        node_color[n,1] = color[1]
        node_color[n,2] = color[2] 
    
    return node_color






def plot_overlay(label, image):

    label = (label-np.min(label))/(np.max(label)-np.min(label))    
    
    label_rgb = np.zeros((label.shape[0],label.shape[1],4), 'uint8')
    label_rgb[:,:,0] = 255 - 255*label
    label_rgb[:,:,1] = 255 - 255*label
    label_rgb[:,:,2] = 255 - 255*label
    label_rgb[:,:,3] = 255*label
       
    overlay = Image.fromarray(label_rgb, mode='RGBA')
    
    
    
    
    image = (image-np.min(image))/(np.max(image)-np.min(image))    
    
    background = Image.fromarray(np.uint8(cmap(image)*255))   
    
    background.paste(overlay, (0, 0), overlay)
    
    plt.imshow(background)
    plt.xticks([])
    plt.yticks([])












def plot_comparison(data_sets, colorbar=False):
    
    count = len(data_sets)
    
    fig, axs = plt.subplots(count, 1, figsize=(12,12))
    for n, data in enumerate(data_sets):
        axs[n].imshow(data)
        if colorbar:
            axs[n].colorbar()





def plot(G, ax=[], color='red', with_labels=False):
    
    if ax==[]:
        fig, ax = plt.subplots() 
        
    nx.draw(G,
            pos = nx.get_node_attributes(G, 'pos'),
            node_size = 1,
            node_color = color,
            with_labels=with_labels,
            ax = ax)
    
    # plt.axis('equal')    
    ax.axis('on')












def plot_components(G, ax=[], crop=False, node_size=0.75, label=True, filename=False):
        
    
    n_comp = 10000
            
    palette = sns.color_palette(None, 2*n_comp)
    
    if crop:
        
        (x_min, x_max), (z_min, z_max) = calculate_crop(G, edge=edge)        
        
        for node in G:              
            G.nodes[node]['pos'] = (G.nodes[node]['pos'][0]-x_min, G.nodes[node]['pos'][1]-z_min)

    if ax==[]:
        fig, ax = plt.subplots() 
    
    
    
    nx.draw(G,
            pos = nx.get_node_attributes(G, 'pos'),
            node_color = get_node_colors(G, 'component'),
            node_size = node_size,
            ax = ax)
    
    limits=ax.axis('on') # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)    
    
    
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
      

    if filename:
        plt.savefig(filename, dpi=300)









def plot_faults(G, ax=[], crop=False, node_size=0.75, label=True, filename=False):
        
    
    n_comp = 10000
            
    palette = sns.color_palette(None, 2*n_comp)

    
    if ax==[]:
        fig, ax = plt.subplots()

  
    nx.draw(G,
            pos = nx.get_node_attributes(G, 'pos'),
            node_color = get_node_colors(G, 'fault'),
            node_size=0.75,
            ax=ax)
    
    if label==True:
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
            label = G.nodes[n]['fault']
            
            ax.text(y_avg, x_avg, label, fontsize = 15, color = palette[G.nodes[n]['fault']])
    
    
    limits=ax.axis('equal') # turns on axis
    ax.axis('on')
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)    
    
    ax.set_ylim(ax.get_ylim()[::-1])
                






def plot_attribute(G, attribute, ax=[], vmin=[], vmax=[], crop=False, node_size=1, filename=False):

    if ax==[]:
        fig, ax = plt.subplots() 
    
    if vmin==[]:
        vmin = min_value_nodes(G, attribute)

    if vmax==[]:
        vmax = max_value_nodes(G, attribute)

    if crop:        
        (x_min, x_max), (z_min, z_max) = calculate_crop(G, edge=edge)                
        for node in G:              
            G.nodes[node]['pos'] = (G.nodes[node]['pos'][0]-x_min, G.nodes[node]['pos'][1]-z_min)
    
    # Colorbar
    cmap = plt.cm.seismic 
    
    nx.draw(G,
            pos = nx.get_node_attributes(G, 'pos'),
            node_color = np.array([G.nodes[node][attribute] for node in G.nodes]),
            node_size = node_size,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap)
    
    limits=ax.axis('on') # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)      
   
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
            
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(attribute, rotation=270)
    plt.axis('equal')

    if filename:
        plt.savefig(filename, dpi=300)










def plot_edge_attribute(G, attribute, ax=[]):
    
    if ax==[]:
        fig, ax = plt.subplots() 

    nx.draw_networkx_edges(G,
                           pos = nx.get_node_attributes(G, 'pos'),
                           edge_color = np.array([G.edges[edge][attribute] for edge in G.edges]),
                           edge_cmap=plt.cm.twilight_shifted,
                           ax=ax)
    
    # Colorbar
    cmap = plt.cm.twilight_shifted
    vmax = max_value_edges(G, attribute)
    vmin = min_value_edges(G, attribute)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
            
    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(attribute, rotation=270)

    









def plot_rose(G, ax=[]):
    
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
    
    

    cmap = plt.cm.twilight_shifted(np.concatenate((np.linspace(0, 1, 18), np.linspace(0, 1, 18)), axis=0))
     
    if ax==[]:
        fig = plt.figure(figsize=(8,8))
            
        ax = fig.add_subplot(111, projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))

    
    ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, 
           width=np.deg2rad(10), bottom=0.0, color=cmap, edgecolor='k')

#    ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight= 'black')
    ax.set_title('Rose Diagram', y=1.10, fontsize=15)
    
    # fig.tight_layout()










def cross_plot(G, var0, var1):
    
    x = np.zeros(len(G.nodes))
    z = np.zeros(len(G.nodes))
    
    
    if var0 == 'x':
        for n, node in enumerate(G):        
            x[n] = G.nodes[node]['pos'][1]
            z[n] = G.nodes[node][var1]

    if var0 == 'z':
        for n, node in enumerate(G):        
            x[n] = G.nodes[node]['pos'][0]
            z[n] = G.nodes[node][var1]        

    if var1 == 'x':
        for n, node in enumerate(G):        
            x[n] = G.nodes[node][var0]
            z[n] = G.nodes[node]['pos'][1]

    if var1 == 'z':
        for n, node in enumerate(G):        
            x[n] = G.nodes[node][var0]
            z[n] = G.nodes[node]['pos'][0] 
    
    plt.plot(x, z, '.')






def plot_matrix(matrix, rows, columns, threshold):


    fig, ax = plt.subplots(1,1, figsize=(12,12))
    ax.imshow(matrix, 'Blues_r')
    
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticklabels(columns)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(rows)
    
    ax.set_xlim(-0.5, matrix.shape[1]-0.5)
    ax.set_ylim(-0.5, matrix.shape[0]-0.5)
    
    # Loop over data dimensions and create text annotations.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] < threshold:
                text = ax.text(j, i, round(matrix[i, j],3),
                        ha="center", va="center", color="r")
            else:        
                text = ax.text(j, i, round(matrix[i, j],3),
                                ha="center", va="center", color="k")






def plot_compare_graphs(G, H):
    fig, ax = plt.subplots(2,1)
    plot_components(G, ax[0])
    plot_components(H, ax[1])





from mpl_toolkits.axes_grid1 import make_axes_locatable




def plot_threshold(data, threshold, value, filename=False):

    fig, axs = plt.subplots(2, 1, figsize=(15,10))
      
    # First plot
    p0 = axs[0].imshow(data)
        
    # Color bar locator
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    cb0 = fig.colorbar(p0, ax=axs[0], cax=cax)
    cb0.ax.plot([-1, 1], [value]*2, 'r')
    
    # Second plot
    p1 = axs[1].imshow(threshold)
    
    # Color bar locator
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    cb0 = fig.colorbar(p1, ax=axs[1], cax=cax)
    
    if filename:
        plt.savefig(filename)
        
        
        



def plot_connections(matrix, rows, columns):
    for n in range(100):    
        threshold = n/100        
        connections = similarity_to_connection(matrix, rows, columns, threshold)        
        plt.scatter(threshold, len(connections), c='red')
        plt.xlabel('Threshold')
        plt.ylabel('Number of connections')
        
        








def plot_location(log, name, ax=[], title=[]):

    colors = get_colors()
    
    cm = LinearSegmentedColormap.from_list('something', colors, N=colors.shape[0])
    
    
    ax.imshow(log, aspect='auto', alpha=0.75, cmap=cm)
    ax.set_xlabel('Time [10^6 yrs]')
    ax.set_yticks([1000, 2250, 3500])
    ax.set_yticklabels([125, 0, -125])
    ax.set_ylabel(name)
    ax.set_title(title)









def bar_plot(attribute, faults, times, name, steps=[], ax=[], title=[]):    

    colors = get_colors() 
        
    if ax==[]:
        fig, ax = plt.subplots()  

    if steps==[]:
        steps = range(attribute.shape[1])    
    
    
    for n, step in enumerate(steps):
        bottom = 0
        for m, fault in enumerate(faults[:, step]):
            if np.isfinite(fault):
                a=attribute[m,step]
                ax.bar(n, a, 1, bottom=bottom, alpha=0.75, edgecolor='white', color=colors[int(fault),:])
                bottom += a
            else:
                break
    
    if title!=[]:
        ax.set_title(title)
        
    ax.set_ylabel(name)
    

    ax.set_xlabel('Time [10^6 yrs]')
    ax.set_ylabel(name)









def stack_plot(attribute, faults, times, name, steps=[], ax=[], title=[]):
    
    colors = get_colors()
    
    if ax==[]:
        fig, ax = plt.subplots()  

    if steps==[]:
        steps = range(attribute.shape[1])   
    
    max_fault = int(np.nanmax(faults))
   
    
    x = np.arange(len(steps))
    
    
    y = np.zeros((max_fault, len(steps)))
    
    
        
    for N in range(max_fault):
    
        
        for n in steps:
            row = faults[:,n]
            if N in faults[:,n]:
                index = np.where(row==N)[0][0]
                y[N,n] = attribute[index,n]
    
     
            
    ax.stackplot(x, y, fc=colors[:max_fault,:], alpha=0.75, edgecolor='white', linewidth=0.5)
    
    if title!=[]:
        ax.set_title(title)    

    ax.set_xlabel('Time [10^6 yrs]')
    ax.set_ylabel(name)
    
    
    
    
