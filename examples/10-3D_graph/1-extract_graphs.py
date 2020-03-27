import matplotlib.pyplot as plt
plt.close("all")

import numpy as np
import meshio
import networkx as nx

# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from image_processing import *
from edits import *
from metrics import *
from plots import *


## LOAD DATA
mesh = meshio.read('grid.vtk')

Nx, Ny, Nz = 1000, 1000, 10

x = mesh.points[:,0].reshape(Nz+1, Nx+1, Ny+1)
y = mesh.points[:,1].reshape(Nz+1, Nx+1, Ny+1)
z = mesh.points[:,2].reshape(Nz+1, Nx+1, Ny+1)

strain_rate = mesh.point_data['strain_rate'].reshape(Nz+1, Nx+1, Ny+1)


# Set up graph
G = nx.Graph()
count = 0

for m in range(Nz+1):  
    
    
    data = strain_rate[m,:,:]
    
    
    ## THRESHOLDING
    threshold = simple_threshold(data, value = 0.0015)
    #threshold = adaptive_threshold(data)
        
    
    ## SKELETONIZE
    #from skimage.morphology import skeletonize
    #skeleton = skeletonize(threshold)
       
    skeleton = guo_hall(threshold)
    
    
    ## PLOT
    plot_comparison([data, threshold, skeleton])
    
    ## CONVERT TO POINTS
    N = np.count_nonzero(skeleton)
    points = np.zeros((N, 2))
    (points[:,0], points[:,0]) = np.where(skeleton != 0)
    
    ## OBTAIN COORDINATES
    points_x = np.zeros(N)
    points_y = np.zeros(N)
    points_z = np.zeros(N)
    
    for n, row in enumerate(points):
        points_x[n] = x[m, int(row[0]), int(row[1])]
        points_y[n] = y[m, int(row[0]), int(row[1])]
        points_z[n] = z[m, int(row[0]), int(row[1])]
    
    
    # Add nodes
    for n in range(N):
        G.add_node(n + count, 
                   pos=(points[n,1], points[n,0]),
                   x = points_x[n],
                   y = points_y[n],
                   z = points_z[n]
                   )
    
    count = count + N

writeObjects(G,
             fileout='graph'
             )
