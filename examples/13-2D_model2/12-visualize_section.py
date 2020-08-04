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




times = range(0, 10400000, 200000)


for n in range(len(times)):
    
        
    # n = 0    
    
    time = times[n]
    
    print('time ' + str(time))
    
    name = str(time).zfill(7)
    
    data = genfromtxt('./csv/others/' + name + '.csv', delimiter=',')
    
    data = np.flip(data, axis=0)
    
    plastic_strain = data[:-1,0].reshape(600, 3000)
    strain_rate    = data[:-1,1].reshape(600, 3000)
    v_x            = data[:-1,2].reshape(600, 3000)
    v_z            = data[:-1,3].reshape(600, 3000)
    
    
    G = pickle.load(open('./graphs/displacement/total/graph_' + name + '.p', 'rb'))
    
    
    
    
    
    
    
    
    
    
    
    
    fig, axs = plt.subplots(3, 1, figsize=(10,8))
    
    
    
    p = axs[0].matshow(strain_rate, cmap='gray_r')
    
    axs[0].set_title('Strain rate')
    
    axs[0].set_xlabel('Distance [km]')
    axs[0].set_xticks(range(0, strain_rate.shape[1]+100, 500))
    axs[0].set_xticklabels(range(0, 310, 50))
    
    axs[0].set_ylabel('Depth [km]')
    axs[0].set_yticks(range(0, strain_rate.shape[0]+100, 100))
    axs[0].set_yticklabels(range(0, 70, 10))
    
    
    # Color bar locator
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    cb0 = fig.colorbar(p, ax=axs[0], cax=cax)
    cb0.ax.set_ylabel('Strain rate', rotation=270)
    
    
    
    
    
    
    
    
    
    
    
    p = axs[1].matshow(strain_rate, cmap='gray_r')
    
    n_comp = 1000
            
    palette = sns.color_palette(None, 2*n_comp)
    
    node_color = np.zeros((len(G),3))
    
    for n, node in enumerate(G):
        color = palette[G.nodes[node]['component']]
            
        node_color[n,0] = color[0]
        node_color[n,1] = color[1]
        node_color[n,2] = color[2]   
            
    
    nx.draw_networkx_nodes(G,
                           pos = nx.get_node_attributes(G, 'pos'),
                           node_color = node_color,
                           node_size = 0.125,
                           ax = axs[1])
    
    
    
    
    limits=plt.axis('on') 
    axs[1].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
    
    axs[1].set_title('Faults')
    
    axs[1].set_xlabel('Distance [km]')
    axs[1].set_xticks(range(0, strain_rate.shape[1]+100, 500))
    axs[1].set_xticklabels(range(0, 310, 50))
    
    axs[1].set_ylabel('Depth [km]')
    axs[1].set_yticks(range(0, strain_rate.shape[0]+100, 100))
    axs[1].set_yticklabels(range(0, 70, 10))
    
    
    # Color bar locator
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    cb0 = fig.colorbar(p, ax=axs[1], cax=cax)
    cb0.ax.set_ylabel('Strain rate', rotation=270)
    
    
    
    
    
    
    
    
    
    
    
    
    
    attribute = 'displacement'
    
    p = axs[2].matshow(strain_rate, cmap='gray_r')
       
    
    nx.draw_networkx_nodes(G,
            pos = nx.get_node_attributes(G, 'pos'),
            node_color = np.array([G.nodes[node][attribute] for node in G.nodes]),
            node_size = 0.125,
            cmap=plt.cm.magma,
            ax=axs[2])
    
    
    # # Colorbar
    # cmap = plt.cm.viridis
    # vmax = max_value_nodes(G, attribute)
    # vmin = min_value_nodes(G, attribute)
    
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm.set_array([])
            
    # cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    # cbar.ax.set_ylabel(attribute, rotation=270)
    # plt.axis('equal')
        
        
    limits=plt.axis('on') 
    axs[2].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        
    
    axs[2].set_title('Displacement')
    
    axs[2].set_xlabel('Distance [km]')
    axs[2].set_xticks(range(0, strain_rate.shape[1]+100, 500))
    axs[2].set_xticklabels(range(0, 310, 50))
    
    axs[2].set_ylabel('Depth [km]')
    axs[2].set_yticks(range(0, strain_rate.shape[0]+100, 100))
    axs[2].set_yticklabels(range(0, 70, 10))
    
    
    # Color bar locator
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes("right", size="3%", pad=0.15)    
    
    cmap = plt.cm.magma
    vmax = max_value_nodes(G, attribute)
    vmin = min_value_nodes(G, attribute)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    
    
    
    
    
    
    cb0 = fig.colorbar(sm, ax=axs[2], cax=cax)
    cb0.ax.set_ylabel(attribute, rotation=270)
    
    
    fig.tight_layout()
    plt.show()
    
    
    plt.savefig('./images/sections/' + name + '.png', dpi=200)
    plt.close('all')
    

