import numpy as np

import matplotlib.pyplot as plt
plt.close("all")

import pickle

import networkx as nx


import cv2

import math



from graph_edits import*
from graph_metrics import*
from graph_plots import*
from graph_utils import*


image = cv2.imread("./images/strain_rate_reduced-00098.png", 0)

G = pickle.load(open("graph.p", 'rb'))

N = nx.number_of_nodes(G)






G = count_edges(G)

G_clean = remove_small_components(G, minimum_size = 30)

G_clean = length_edges(G_clean)

G_clean = extract_property(G_clean, image, 'strain_rate')

#plt.figure()
#plot(G_clean, 'strain_rate', 'viridis')

G_clean = label_components(G_clean)

plot(G_clean)
plot_labels(G_clean)












G_exp = expand_network(G_clean)

G_exp = length_edges(G_exp)

G_exp = label_components(G_exp)
   
plt.figure()
plot(G_exp)
plot_labels(G_exp)







    

 

#G = calculate_strikes_in_radius(G, radius = 7)
#plot_rose(G)

#G = calculate_strikes_in_neighborhood(G, neighbors = 3)
#plot_rose(G)

#plot_component(G, label=1)






#plt.figure()
#plot(k, 'strain_rate', 'viridis')







components = set(nx.get_node_attributes(G_exp,'component').values())


plt.figure()

for c in components:
    
    k = select_component(G_exp, c)
    
    max_strain = max_value(k, 'strain_rate')
    
    length = total_length(k)

    plt.scatter(length, max_strain, color='black')
    plt.xlabel('Length')
    plt.ylabel('Strain rate')





#plt.figure()
#cross_plot(k, 'y', 'strain_rate')







    
    



















