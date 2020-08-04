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


for n, time in enumerate(times):
    
    
    # n = 0
    
    time = times[n]
    
    print('time ' + str(time))
    
    name = str(time).zfill(7)
    
    data = genfromtxt('./csv/others/' + name + '.csv', delimiter=',')
    
    data = np.flip(data, axis=0)
    
    # plastic_strain = data[:-1,0].reshape(600, 2500)
    # strain_rate    = data[:-1,1].reshape(600, 2500)
    v_x            = data[:-1,2].reshape(600, 3000)
    v_z            = data[:-1,3].reshape(600, 3000)
    
    
    G = pickle.load(open('./graphs/correlated/graph_' + name + '.p', 'rb'))
    
    
    # plt.matshow(v_z)
    # plt.colorbar()
    
    
    
    # v_x[v_x == 0] = 'nan'
    # v_z[v_z == 0] = 'nan'
    
    
    # Unfreeze graph
    G = nx.Graph(G)
    
    # G = select_components(G, 0)
    
    
    G = calculate_direction(G, 3)
    
    H = calculate_pickup_points(G, 5)
    
    H = extract_attribute(H, v_x, 'v_x')
    H = extract_attribute(H, v_z, 'v_z')
    
    
    
            
            
    
    
    
    
    # for node in G:
    #     if G.nodes[node]['pos'][1] < 10 and G.nodes[node]['displacement'] > 200:
    #         G.nodes[node]['displacement'] = 0
    
    
    
    H = filter_pickup_points(G, H)
    
    
    
    
    
    G = calculate_displacement(G, H, dt=200000)
    
    
    
    
    # fig, ax = plt.subplots()
    # ax.matshow(v_x)
    # plot_attribute(H, 'v_x', ax)
    
    
    # G = remove_below(G, 'displacement', value = 500)  
    
    
    
    
    # fig, ax = plt.subplots()
    # ax.matshow(v_x)
    # plot_components(G, ax, label=False)
    # plot_components(H, ax, label=False)
    
    
    
    # # fig, ax = plt.subplots()
    # # ax.matshow(v_x)
    # # plot_attribute(G, 'displacement', ax)
    
    
    
    
    fig, ax = plt.subplots()
    ax.matshow(v_x)
    plot_attribute(G, 'displacement', ax)
    plt.savefig('./images/displacement/instant/' + name + '.png', dpi=200)
    plt.close("all")       
    
    
    
    
    # # plt.figure()
    # # cross_plot(G, 'displacement', 'x')
    # # plt.gca().invert_yaxis()
      
    
    
    
    
    pickle.dump(G, open('./graphs/displacement/instant/graph_' + name + '.p', "wb" ))

