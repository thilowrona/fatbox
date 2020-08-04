import meshio

import numpy as np

import matplotlib.pyplot as plt
plt.close("all")

mesh = meshio.read("grid.vtk")


nx, ny, nz = 1500, 100, 500

x = mesh.points[:,0].reshape(nz+1, ny+1, nx+1)
y = mesh.points[:,1].reshape(nz+1, ny+1, nx+1)
z = mesh.points[:,2].reshape(nz+1, ny+1, nx+1)

data = mesh.point_data['Result'].reshape(nz+1, ny+1, nx+1)

x = np.flip(x, axis=0)
y = np.flip(y, axis=0)
z = np.flip(z, axis=0)

data = np.flip(data, axis=0)

plt.imshow(data[:,50,:])
