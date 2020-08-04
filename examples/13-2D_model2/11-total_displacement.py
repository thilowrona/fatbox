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
    
    name_0 = str(times[n]).zfill(7)
    name_1 = str(times[n+1]).zfill(7)


    # MAKE SURE TO COPY FIRST ONE TO NEW FOLDER
    if n == 0:        
        G_0 = pickle.load(open('./graphs/displacement/instant//graph_' + name_0 + '.p', 'rb'))
        pickle.dump(G_0, open('./graphs/displacement/total//graph_' + name_0 + '.p', "wb" ))
        
        
    
    G_0 = pickle.load(open('./graphs/displacement/total/graph_' + name_0 + '.p', 'rb'))        
    G_1 = pickle.load(open('./graphs/displacement/instant/graph_' + name_1 + '.p', 'rb'))







    cc = common_components(G_0, G_1)
    
    for component in cc:
        
        points_0 = get_nodes(select_components(G_0, component))
        points_1 = get_nodes(select_components(G_1, component))
    
    
        for n in range(points_1.shape[0]):    
            index = closest_node(points_1[n,1:3], points_0[:,1:3])    
            value = points_0[index][3]    
            points_1[n,3] += value
        
        
        G_1 = assign_nodes(G_1, points_1)




    fig, ax = plt.subplots()
    ax.matshow(np.zeros((600,3000)))
    plot_attribute(G_1, 'displacement', ax)
    plt.savefig('./images/displacement/total/' + name_1 + '.png', dpi=200)
    plt.close("all")   




    pickle.dump(G_1, open('./graphs/displacement/total/graph_' + name_1 + '.p', "wb" ))
