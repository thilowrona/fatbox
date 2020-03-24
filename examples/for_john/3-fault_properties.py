import pickle
import matplotlib.pyplot as plt
plt.close("all")

# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from edits import *
from metrics import *
from plots import *


filename = "v5w4h60_z95_20MYR"

G = pickle.load(open("./graphs/graph_" + filename + ".p", 'rb'))


x = np.load("./npy/" + filename + ".npy")[:,:,0]
G = extract_attribute(G, x, 'x')

y = np.load("./npy/" + filename + ".npy")[:,:,1]
G = extract_attribute(G, y, 'y')

z = np.load("./npy/" + filename + ".npy")[:,:,2]
G = extract_attribute(G, z, 'z')

strain_rate = np.load("./npy/" + filename + ".npy")[:,:,3]
G = extract_attribute(G, strain_rate, 'strain_rate')

plastic_strain = np.load("./npy/" + filename + ".npy")[:,:,4]
G = extract_attribute(G, plastic_strain, 'plastic_strain')


#fig, ax = plt.subplots(1, 1, figsize=(8,10))
#ax.imshow(plastic_strain)
#plot_components(G, 'plastic_strain', ax=ax)


#for node in G:
#    G.nodes[node]['pos'] = (G.nodes[node]['x'], G.nodes[node]['y'])




G = compute_edge_length(G)


n_comp = number_of_components(G)

lengths = np.zeros(n_comp)
plastic_strain = np.zeros(n_comp)


for n in range(n_comp):
    lengths[n] = total_length(select_component(G, component=n))/1000
    plastic_strain[n] = max_value_nodes(select_component(G, component=n), 'plastic_strain')



fig, ax = plt.subplots(1, 1, figsize=(8,8))
for n in range(n_comp):
    plt.scatter(lengths[n], plastic_strain[n])

plt.xlabel('Fault lengths [km]')
plt.ylabel('Strain_rates * 1e+14')
plt.savefig("./images/Fault_length_vs_plastic_strain" + filename +  ".png", dpi=300)



fig, ax = plt.subplots(1, 1, figsize=(8,8))
for n in range(n_comp):
    plt.scatter(lengths[n], plastic_strain[n])

plt.xlabel('Fault lengths')
plt.ylabel('Plastic strain')
plt.savefig('./images/Plastic_strain-fault_length_v5w2h55_z95_20MYR.png', dpi=300)


pickle.dump(G, open("./graphs/graph_" + filename + ".p", "wb" ))


import pandas as pd
df = pd.DataFrame(data = np.column_stack((lengths, plastic_strain)), index=range(n_comp), columns = ['Length', 'Strain rate'])
df.index.name ='Fault'

df.to_csv(filename + ".csv")











