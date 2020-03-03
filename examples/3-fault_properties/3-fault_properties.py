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
plot_attribute(G, 'strain_rate', ax=ax)



G = compute_edge_length(G)


n_comp = number_of_components(G)

lengths = np.zeros(n_comp)
strain_rates = np.zeros(n_comp)

for n in range(n_comp):
    lengths[n] = total_length(select_component(G, component=n))
    strain_rates[n] = max_value_nodes(select_component(G, component=n), 'strain_rate')




fig, ax = plt.subplots(1, 1, figsize=(8,8))
for n in range(n_comp):
    plt.scatter(lengths[n], 1e14*strain_rates[n])


plt.xlabel('Fault lengths')
plt.ylabel('Strain_rates * 1e+14')












