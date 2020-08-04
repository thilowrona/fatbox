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



# for n, time in enumerate(times):
    
time = 10200000

print(time)

name = str(time).zfill(7)


data = genfromtxt('./csv2/' + name + '.csv', delimiter=',')
data = np.flip(data, axis=0)


upper_crust = data[:-1,1].reshape(800, 1800)
lower_crust = data[:-1,0].reshape(800, 1800)



(t_max, x_max) = (800, 1800)


surface     = topo_line(upper_crust)
transition  = bottom_line(upper_crust, 0.1)
base        = bottom_line(lower_crust, 0.1)

import scipy
from scipy import interpolate, spatial

x = np.arange(x_max)
y = surface
f_surface = interpolate.interp1d(x, y)




dist = scipy.spatial.distance.pdist(np.hstack((x, transition)), f_surface)



plt.plot(np.arange(x_max), transition - surface, 'blue')
plt.plot(np.arange(x_max), dist, 'red')



# factor = 0.25    
# thickness_upper = (transition - surface) * factor
# thickness_lower = (base-transition) * factor
# thickness_whole = thickness_upper + thickness_lower

# initial_upper = 26.75
# initial_lower = 15
# initial_whole = initial_upper + initial_lower

# thinning_upper = thickness_upper/initial_upper
# thinning_lower = thickness_lower/initial_lower
# thinning_whole = thickness_whole/initial_whole


# fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(6,12))

# axes[0].set_title('Time ' + str(time))
# axes[0].imshow(upper_crust)
# axes[0].plot(np.arange(x_max), surface, 'red')
# axes[0].plot(np.arange(x_max), transition, 'red')
# axes[0].set_xlabel('X')
# axes[0].set_ylabel('Y')


# axes[1].imshow(lower_crust)
# axes[1].plot(np.arange(x_max), transition, 'blue')
# axes[1].plot(np.arange(x_max), base, 'blue')
# axes[1].set_xlabel('X')
# axes[1].set_ylabel('Y')


# axes[2].plot(np.arange(x_max), thinning_upper, 'red')
# axes[2].plot(np.arange(x_max), thinning_lower, 'blue')
# axes[2].plot(np.arange(x_max), thinning_whole, 'black')
# axes[2].legend(['Upper crust', 'Lower crust', 'Whole crust'])
# axes[2].set_xlabel('X')
# axes[2].set_ylim(0, 1)
# axes[2].set_ylabel('Thinning')

# plt.tight_layout()
# plt.show()
# plt.savefig('./images/thinning/sections/' + name + '.png', dpi=100)
# plt.close("all") 




# plt.figure()
# plt.plot(thinning_lower[:900], thinning_upper[:900])
# plt.title('Time ' + str(time))
# plt.xlabel('Lower crustal thinning')
# plt.ylabel('Upper crustal thinning')
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.savefig('./images/thinning/crossplots/upper_lower/' + name + '.png', dpi=100)
# plt.close("all") 













