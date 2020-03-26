from pyevtk.hl import gridToVTK 

import numpy as np 

# Dimensions 

nx, ny, nz = 1000, 1000, 10

lx, ly, lz = 0.1, 0.1, 0.0148067

dx, dy, dz = lx/nx, ly/ny, lz/nz 

ncells = nx * ny * nz 

npoints = (nx + 1) * (ny + 1) * (nz + 1) 

# Coordinates 

x = np.arange(0, lx + 0.1*dx, dx, dtype='float64') 

y = np.arange(0, ly + 0.1*dy, dy, dtype='float64') 

z = np.arange(0, lz + 0.1*dz, dz, dtype='float64') 


gridToVTK("./box", x, y, z) 