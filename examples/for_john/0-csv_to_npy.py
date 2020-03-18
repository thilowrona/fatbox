import numpy as np
import pandas as pd
from scipy.interpolate import griddata


# Define grid spacing
dx = 1250
dy = 1250


# Input file
file = "./csv/v10w2h55_z95_10MYR.csv"


# Load file
df = pd.read_csv(file)
df = df.rename({'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z'}, axis='columns')  
    
    
# Extract property
x=df.X
y=df.Y
z=df.Z
strain_rate=df.strain_rate
plastic_strain=df.plastic_strain


#import matplotlib.pyplot as plt
#plt.close("all")
#plt.scatter(x,y,c = strain_rate)


# Compute coordinates
xmin, xmax = min(x), max(x)
ymin, ymax = min(y), max(y)

Nx = int((xmax-xmin)/dx)
Ny = int((ymax-ymin)/dy)


# Define grid
xi = np.linspace(xmin, xmax, Nx+1)
yi = np.linspace(ymin, ymax, Ny+1)


## Interpolate property on to grid
x_arr, y_arr = np.meshgrid(xi, yi)
z_arr = df.Z.mean()*np.ones((Nx+1,Ny+1))

strain_rate_arr = griddata((x, y), strain_rate, (xi[None,:], yi[:,None]), method='cubic')
plastic_strain_arr = griddata((x, y), plastic_strain, (xi[None,:], yi[:,None]), method='cubic')


arr = np.stack((x_arr, y_arr, z_arr, strain_rate_arr, plastic_strain_arr), axis = 2)

# Save gridded data
np.save("./npy/v10w2h55_z95_10MYR.npy", arr)