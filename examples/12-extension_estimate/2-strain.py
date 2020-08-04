import matplotlib.pyplot as plt
plt.close("all")

import numpy as np
from numpy import genfromtxt
import meshio
import networkx as nx
import pickle
import scipy.ndimage

# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from image_processing import *
from edits import *
from metrics import *
from plots import *



stats   = np.zeros((52, 3))




# for k, time in enumerate(range(0, 10400000, 200000)):
        
time = 10200000

print('time ' + str(time))

name = str(time).zfill(8)

count = 0


data = genfromtxt('./csv/' + name + '.csv', delimiter=',')

data = data[:-1,0].reshape(2000, 4000)

data = np.flip(data, axis=0) 


(t_max, x_max) = data.shape


def topo_line(data):
    col = data.shape[1]
    line =  np.zeros(col)
    for n in range(col):
        m = 0
        while data[m,n] == 0:
            m = m + 1        
        line[n] = m
    return line.astype(int)


line = topo_line(data)



fig, axes = plt.subplots(nrows=3, sharex=True)

axes[0].imshow(data)
axes[0].plot(np.arange(x_max), line, 'red')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')


values = data[line.astype(np.int), np.arange(x_max).astype(np.int)]


axes[1].plot(values)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Plastic strain')      



axes[2].plot(line)
axes[2].set_xlabel('X')
axes[2].set_ylabel('Surface elevation')  



def above_below(values, threshold):

    low  = 0
    high = 0
    for value in values:
        if value < threshold:
            low = low + value
        else:
            high = high + value
    
    total = np.sum(values)

    return low, high






threshold = 1
low, high = above_below(values, threshold)
total = np.sum(values)
   


fig = plt.figure()
plt.hist(values, density=True, bins=50)
plt.plot([threshold, threshold], [0, 1])
plt.ylabel('Frequency')
plt.xlabel('Plastic strain');
plt.title(str(int(round(low/total*100))) + ' % background - ' + str(int(round(high/total*100))) + ' % faults ')
plt.savefig('./images/histograms/line/time_' + str(time) + '.png', dpi = 100)





# values = np.nan_to_num(data, nan=0).flatten()

# threshold = 1
# low, high = above_below(values, threshold)
# total = np.sum(values)


# fig = plt.figure()
# plt.hist(values, density=True, bins=50)
# plt.plot([threshold, threshold], [0, 1])
# plt.ylabel('Frequency')
# plt.xlabel('Plastic strain');
# plt.title(str(int(round(low/total*100))) + ' % background - ' + str(int(round(high/total*100))) + ' % faults ')
# plt.savefig('./images/histograms/array/time_' + str(time) + '.png', dpi = 100)


# plt.close('all')


# stats[k, 0] = low
# stats[k, 1] = high
# stats[k, 2] = total
    
    
    
# np.save('./images/histograms/line/stats.npy', stats)


#%%


stats = np.load('./images/histograms/array/stats.npy')



plt.plot(stats[:,0]/stats[:,2])
plt.plot(stats[:,1]/stats[:,2])
plt.title('strain distribution (array)')
plt.xlabel('time')
plt.ylabel('% of strain')
plt.legend(['background', 'faults'])