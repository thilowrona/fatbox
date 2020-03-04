import pickle
import matplotlib.pyplot as plt
plt.close("all")

# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from edits import *
from metrics import *
from plots import *




G = pickle.load(open("graph.p", 'rb'))

strain_rate = np.load("NearSurfaceIsotherm_335K_strain_rate.npy")

G = extract_attribute(G, strain_rate, 'strain_rate')




fig, ax = plt.subplots(1, 1, figsize=(8,10))
plt.imshow(strain_rate)
cb = plt.colorbar()
plot_components(G, ax=ax)




G = select_component(G, component=4)

y, strain_rate = strain_profile(G, 'strain_rate')




fig, ax = plt.subplots(1, 1, figsize=(8,10))
ax.fill_between(y,strain_rate, color='gray')
ax.scatter(y,strain_rate, color='black')
ax.set_xlabel('y-coordinate')
ax.set_ylabel('strain rate')









