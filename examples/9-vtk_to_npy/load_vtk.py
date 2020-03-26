import meshio


mesh = meshio.read('solution-00050.vtk')

nx, ny, nz = 1000, 1000, 10

x = mesh.points[:,0].reshape(nz+1, nx+1, ny+1)
y = mesh.points[:,1].reshape(nz+1, nx+1, ny+1)
z = mesh.points[:,2].reshape(nz+1, nx+1, ny+1)



strain = mesh.point_data['strain_rate'].reshape(nz+1, nx+1, ny+1)


plt.imshow(strain[5,:,:])