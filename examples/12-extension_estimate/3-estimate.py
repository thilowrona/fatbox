import matplotlib.pyplot as plt
plt.close("all")

import math
import numpy as np
from numpy import genfromtxt


# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')




stats   = np.zeros((52, 3))

threshold = 1


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



fig, axes = plt.subplots(nrows=2, sharex=True)

axes[0].imshow(data[:800,:])
axes[0].plot(np.arange(x_max), line, 'red')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')


plastic_strain = data[line.astype(np.int), np.arange(x_max).astype(np.int)]

areas = np.where(plastic_strain > threshold, 1, 0)

x  = np.arange(4000)
y1 = np.zeros(4000)
y2 = areas*plastic_strain


axes[1].plot(plastic_strain)
axes[1].fill_between(x, y1, y2)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Plastic strain')      


pairs = []

for n in range(4000):
    
    if areas[n] == 1:
        if areas[n-1] == 0:
            x0 = n
            
    if areas[n] == 0:
        if areas[n-1] == 1:
            x1 = n            
            pairs.append((x0, x1))    
        

        
thickness = np.zeros(len(pairs))
max_strain= np.zeros(len(pairs))

dip   = math.radians(45)

for n, pair in enumerate(pairs):

    thickness[n]  = ((pair[1] - pair[0]) * 0.1) * math.cos(dip)
    max_strain[n] = np.max(plastic_strain[pair[0]:pair[1]])



# surface_elevation = -line * 100


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

axes[0].scatter(thickness, max_strain)
axes[0].set_xlabel('Thickness [km]')
axes[0].set_ylabel('Max. strain')  


initial_strain = 0.97

displacement = thickness * (max_strain - initial_strain)



axes[1].scatter(thickness, displacement)
axes[1].set_title('Total displacement: ' + str(int(round(np.sum(displacement)))) + ' km')
axes[1].set_xlabel('Thickness [km]')
axes[1].set_ylabel('Displacement [km]') 




heave = math.cos(dip) * displacement


axes[2].scatter(thickness, heave)
axes[2].set_title('Sum of heaves: ' + str(int(round(np.sum(heave)))) + ' km')
axes[2].set_xlabel('Thickness [km]')
axes[2].set_ylabel('Heave [km]') 



# fig.savefig('./images/displacement/time_' + str(time) + '.png', dpi = 100)

# plt.close('all')


