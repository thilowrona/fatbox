import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from edits import *
from metrics import *
from plots import *


filename = "v5w2h55_z95_20MYR"

G = pickle.load(open("./graphs/graph_" + filename + ".p", 'rb'))


image = np.load("./npy/" + filename + ".npy")[:,:,4]


#plt.imshow(threshold)

#G = select_component(G, component=2)

length = 10
dl = 1250

node = 110

y,x = G.nodes[node]['pos']


plt.imshow(image)

plt.scatter(y,x)

plt.figure()
plt.plot(np.arange(y-length//2, y+length//2), image[y-length//2:y+length//2,x])

#for node in G:   
#
#    
#    y,x = G.nodes[node]['pos']  
#    
#    if x > length:              
#        
#        xs = np.arange(length)        
#        ys = image[y-length//2:y+length//2,x]
#        
#        plt.plot(xs,ys)
##        plt.scatter(x,image[x,y])
#
#
