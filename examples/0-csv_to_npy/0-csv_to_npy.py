import numpy as np
import pandas as pd
from scipy.interpolate import griddata


# Define grid spacing
dx = 500
dy = 500


# Input file
file = "NearSurfaceIsotherm_335K.csv"


# Load file
df = pd.read_csv(file)
df = df.rename({'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z'}, axis='columns')  
    
    
# Extract property
x=df.X
y=df.Y
strain_rate=df.strain_rate


# Compute coordinates
xmin, xmax = min(x), max(x)
ymin, ymax = min(y), max(y)

Nx = int((xmax-xmin)/dx)
Ny = int((ymax-ymin)/dy)


# Define grid
xi = np.linspace(xmin, xmax, Nx)
yi = np.linspace(ymin, ymax, Ny)


# Interpolate property on to grid
arr_strain_rate = griddata((x, y), strain_rate, (xi[None,:], yi[:,None]), method='cubic')


# Save gridded data
np.save("NearSurfaceIsotherm_335K_strain_rate.npy", arr_strain_rate)