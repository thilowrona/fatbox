import pickle
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

from edits import *
from metrics import *
from plots import *

times = range(10200000, 0, -200000)


upper = np.zeros(len(times))
lower = np.zeros(len(times))


# for n, time in enumerate(times):
    
time = 10200000

print(time)

name = str(time).zfill(7)


data = genfromtxt('./csv2/' + name + '.csv', delimiter=',')
data = np.flip(data, axis=0)

   
crust = data[:-1,1].reshape(800, 1800) + data[:-1,0].reshape(800, 1800)

(t_max, x_max) = crust.shape

surface = topo_line(crust)

base    = bottom_line(crust, 0.2)

fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(15,15))

axes[0].imshow(crust)
axes[0].plot(np.arange(x_max), surface, 'red')
axes[0].plot(np.arange(x_max), base, 'red')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

factor = 0.25
thickness = (base-surface) * factor

max_thickness = np.max(thickness)
min_thickness = np.min(thickness)
beta = 2*max_thickness/(min_thickness+max_thickness)

cutoff = 0.9
for n in range(0, 1800-1, 1):
    if thickness[n] < cutoff * max_thickness:
        start = n
        break

for n in range(1800-1, 0, -1):
    if thickness[n] < cutoff * max_thickness:
        end = n
        break

width = (end-start) * factor
d_est = width*beta

axes[1].plot(np.arange(x_max), thickness, 'red')
axes[1].scatter(start, thickness[start], color='red')
axes[1].scatter(end, thickness[end], color='red')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Crustal thickness [km]')
axes[1].set_title('beta = ' + str(round(beta)) + 
                  ' , width = ' + str(int(width)) + 
                  ' km, d_est = ' + str(int(d_est)) + 
                  ' km, d_true = ' + str(int(time/100000)) + 
                  ' km')




# plt.savefig('./images/thinning/' + name + '.png', dpi=100)
# plt.close("all") 







