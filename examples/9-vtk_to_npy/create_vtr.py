from pyevtk.hl import gridToVTK 

import numpy as np 

# Dimensions 
nx, ny, nz = 1500,   100,    500
x0, y0, z0 = 150000, 0, 151820
lx, ly, lz = 300000, 10000, 201820

dx, dy, dz = (lx-x0)/nx, (ly-y0)/ny, (lz-z0)/nz 

ncells = nx * ny * nz 
npoints = (nx + 1) * (ny + 1) * (nz + 1) 

# Coordinates 
x = np.arange(x0, lx + 0.1*dx, dx, dtype='float64') 
y = np.arange(y0, ly + 0.1*dy, dy, dtype='float64') 
z = np.arange(z0, lz + 0.1*dz, dz, dtype='float64') 

# Write vtr
gridToVTK("./box", x, y, z) 




