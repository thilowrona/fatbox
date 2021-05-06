#!/usr/bin/env python3
""" Does the same as the Jupyter notebook but as a normal python script """
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from faultanalysistoolbox.edits import array_to_points, label_components
from faultanalysistoolbox.image_processing import guo_hall
from faultanalysistoolbox.plots import plot_components
from scipy.spatial import distance_matrix
from tqdm import tqdm

# %%
# ## Fault extraction
# First, we load our data - a strain rate map extracted just below the surface
# of the model:
path = '/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/1-fault_extraction/NearSurfaceIsotherm_335K_strain_rate.npy'
strain_rate = np.load(
    path
)

# %%
# Now we can plot it to look the faults in the model

plt.figure()
plt.title('Strain rate')
plt.imshow(strain_rate, vmin=0)
plt.colorbar()
plt.show(block=False)

# %%
# Next we want to separate the faults from the background using a threshold:

threshold = np.where(strain_rate > 1.5e-14, 1, 0).astype(np.uint8)

plt.figure()
plt.title('Threshold')
plt.imshow(threshold)
plt.axis('off')
plt.show(block=False)

# %%
# Now we can reduce the areas above the threshold to lines using a skeletonize
# algorithm:

skeleton = guo_hall(threshold)

plt.figure()
plt.title('Skeleton')
plt.imshow(skeleton)
plt.axis('off')
plt.show(block=False)

# %%
# Now we can convert these lines to points:

points = array_to_points(skeleton)

# %%
# These points become the nodes of our graph G:

G = nx.Graph()

for node, point in enumerate(points):
    G.add_node(node)
    G.nodes[node]['pos'] = point

# %%
# Remember a graph is an object consisting only of nodes and edges. Our graph
# for example looks like this:

fig, axs = plt.subplots(1, 2)

axs[0].set_title('Network')
nx.draw(G,
        pos=nx.get_node_attributes(G, 'pos'),
        node_size=1,
        ax=axs[0])
axs[0].axis('equal')


axs[1].set_title('Zoom in')
nx.draw(G,
        pos=nx.get_node_attributes(G, 'pos'),
        node_size=1,
        ax=axs[1])
axs[1].axis('equal')
axs[1].set_ylim([500, 600])

plt.show(block=False)

# %%
# You can see that the graph only consists of closely spaced points, which are
# not yet connected. So let's change that!
#
#
# We calculate the distance between all nodes in a distance matrix and connect
# the ones close to each other (<1.5 pixels away):

dm = distance_matrix(points, points)

# print(str(points.shape[0]) + ' Points')
for n in tqdm(range(points.shape[0]), desc='Connecting Points'):
    # stdout.write("\r%d" % n)
    # stdout.flush()
    for m in range(points.shape[0]):
        if dm[n, m] < 1.5 and n != m:
            G.add_edge(n, m)


fig, axs = plt.subplots(1, 2)

axs[0].set_title('Graph')
nx.draw(G,
        pos=nx.get_node_attributes(G, 'pos'),
        node_size=1,
        ax=axs[0])
axs[0].axis('equal')


axs[1].set_title('Zoom in')
nx.draw(G,
        pos=nx.get_node_attributes(G, 'pos'),
        node_size=1,
        ax=axs[1])
axs[1].axis('equal')
axs[1].set_ylim([500, 600])

plt.show(block=False)

# %%
# Now we can see that neighboring nodes are connected by edges (black lines).
# This allows us to label the nodes connected to one another as components:

G = label_components(G)

fig, axs = plt.subplots(1, 1)
axs.imshow(strain_rate, 'gray_r', vmin=0)
plot_components(G, axs, label=True)
plt.title('Strain rate with fault network')
plt.show(block=False)

# %%
# When we zoom in, we can see the nodes colored by their component and the
# edges connecting them:

fig, axs = plt.subplots(1, 1)
axs.imshow(strain_rate, 'gray_r', vmin=0)
plot_components(G, axs, label=False)
axs.set_xlim([250, 450])
axs.set_ylim([400, 600])
plt.title('Strain rate with fault network')
plt.show(block=False)

# %%
# ## Structure of the network
# Let's have a look at the structure of the fault network (or graph). Remember
# it only consists of nodes and edges. So let's have a look at the nodes:

print(G.nodes)

# %%
# Okay, nothing special here, just a list of the nodes. Let's pick out one:

print(G.nodes[0])

# %%
# Alright, we can see the position of the node and the component it belongs to.
# Let's say we want to give it an extra property, e.g. the strain rate at its
# location:

G.nodes[0]['strain_rate'] = strain_rate[
    int(G.nodes[0]['pos'][1]),
    int(G.nodes[0]['pos'][0])
]
print(G.nodes[0])

# %%
# Nice! Let's do that for all nodes:

for node in G.nodes:
    G.nodes[node]['strain_rate'] = strain_rate[
        int(G.nodes[node]['pos'][1]),
        int(G.nodes[node]['pos'][0])
    ]

# %%
# and plot it:

fig, ax = plt.subplots()

ax.set_title('Fault network with strain rate')
nx.draw(G,
        pos=nx.get_node_attributes(G, 'pos'),
        node_color=np.array([G.nodes[node]['strain_rate']
                            for node in G.nodes]),
        node_size=1,
        ax=ax)
ax.axis('equal')
plt.show(block=False)

# %%
# Like this we can compute and visualize all kinds of properties on the fault
# network.
#
# But what about the edges?

print(G.edges)

# %%
# Alright, just tuples of nodes. Let's pick one:

print(G.edges[(0, 5)])

# %%
# Okay, they have no property yet. Let's calculate its length:

edge = (0, 5)
G.edges[edge]['length'] = np.linalg.norm(
    G.nodes[edge[0]]['pos']-G.nodes[edge[1]]['pos'])


print(G.edges[(0, 5)])

# %%
# Again, we can do this for all edges:

for edge in G.edges:
    G.edges[edge]['length'] = np.linalg.norm(
        G.nodes[edge[0]]['pos']-G.nodes[edge[1]]['pos'])

# %%
# and plot it:

fig, ax = plt.subplots()

ax.set_title('Fault network with edge lenghts')
nx.draw(G,
        pos=nx.get_node_attributes(G, 'pos'),
        edge_color=np.array([G.edges[edge]['length'] for edge in G.edges]),
        node_size=0.001,
        ax=ax)
ax.axis('equal')
plt.show()

# %%
# Awesome! That's it. You've extracted your first fault network. In the next
# tutorial, we will learn how to compute and visualize fault strikes:
# https://github.com/thilowrona/fault_analysis_toolbox/blob/master/examples/2-fault_properties/2-fault_properties.ipynb
#
