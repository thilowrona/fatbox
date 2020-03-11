import networkx as nx
import pandas as pd
import numpy as np

import pickle


# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from metrics import *
from plots import *
from utils import *


import matplotlib.pyplot as plt
plt.close("all")

G = pickle.load(open("graph.p", 'rb'))



# Plot graph
fig = plt.figure(figsize=(6,18))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex = ax1)
                
                  
plot_components(G, ax = ax1)

noc = number_of_components(G)

x_mean = mean_x_components(G)












lengths = fault_lengths(G)









n_comp = 1000        
palette = sns.color_palette(None, 2*n_comp)
color = np.zeros((noc,3))

for n in range(noc):
    ax2.plot([x_mean[n],x_mean[n]],[0,lengths[n]], color=palette[n])





dx = np.diff(sorted(x_mean))

plt.figure()

n, bins, patches = plt.hist(dx, bins=20)

plt.title('Fault spacing')
plt.xlabel('fault distance')
plt.ylabel('Frequency')
