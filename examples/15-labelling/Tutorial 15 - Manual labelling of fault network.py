import json
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.image import imread

# read file
with open('/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/15-labelling/polygons.json', 'r') as myfile:
    data = myfile.read()

# parse file
obj = json.loads(data)

# %%
# Next, we obtain all group IDs in the .json file

group_id = set()

shapes = obj['shapes']
for shape in shapes:
    group_id.add(shape['group_id'])

print(group_id)

# %%
# Now, we can define an empty network and fill it with nodes and edges from the
# polygons of each group ID:
G = nx.Graph()

node = 0
for shape in shapes:
    for group in group_id:
        points = shape['points']
        for m in range(len(points)):
            position = (int(points[m][0]), int(points[m][1]))

            G.add_node(node)
            G.nodes[node]['pos'] = position

            if m < len(points)-1:
                G.add_edge(node, node+1)
            node += 1

# %%
# We can also plot the network like this:
img = imread('/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/15-labelling/outcrop.jpg')
fig, axs = plt.subplots(1, 1, figsize=(20, 12))
axs.imshow(img)
nx.draw(G,
        pos=nx.get_node_attributes(G, 'pos'),
        node_size=5,
        with_labels=True,
        font_size=15,
        ax=axs)
plt.show()

# %%
# Although the network already looks fine, looking at the edges of the network
# more closely:

print(G.edges)

# %%
# we find that the nodes 15 and 25 are not connected, because they were part of
# different polygons. We can correct this by merging nodes within a certain
# distance threshold while preserving their edges:


H = G.copy()

threshold = 25

for node in G:
    for another in G:

        if H.has_node(node) and H.has_node(another) and node != another:

            if math.sqrt(
                (G.nodes[node]['pos'][0] - G.nodes[another]['pos'][0])**2 +
                (G.nodes[node]['pos'][1] - G.nodes[another]['pos'][1])**2
            ) < threshold:
                neighbors = H.neighbors(another)
                for neighbor in neighbors:
                    H.add_edge(node, neighbor)
                H.remove_node(another)

fig, axs = plt.subplots(1, 1, figsize=(20, 12))
axs.imshow(img)

nx.draw(
    H,
    pos=nx.get_node_attributes(G, 'pos'),
    node_size=5,
    with_labels=True,
    font_size=15,
    ax=axs
)
plt.show()
# %%
# Looking at the edges of the network:

print(H.edges)

# %%
# shows an edge between node 3 and node 24.
# %%
# # Application
# %%
# Now, we can use this method to map all faults in the image. First, we label
# all polygons in labelme:
# %%
# ![title](labelme/s6.png)
# %%
# Then, we save the .json file and load it in python:


# read file
with open('/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/15-labelling/polygons_all.json', 'r') as myfile:
    data = myfile.read()

# parse file
obj = json.loads(data)

# %%
# We obtain the group IDs:

group_id = set()

shapes = obj['shapes']
for shape in shapes:
    group_id.add(shape['group_id'])

# %%
# and build the network:


G = nx.Graph()

node = 0
for shape in shapes:
    for group in group_id:
        points = shape['points']
        for m in range(len(points)):
            position = (int(points[m][0]), int(points[m][1]))

            G.add_node(node)
            G.nodes[node]['pos'] = position

            if m < len(points)-1:
                G.add_edge(node, node+1)
            node += 1


H = G.copy()

threshold = 10

for node in G:
    for another in G:

        if H.has_node(node) and H.has_node(another) and node != another:

            if math.sqrt(
                (G.nodes[node]['pos'][0] - G.nodes[another]['pos'][0])**2 +
                (G.nodes[node]['pos'][1] - G.nodes[another]['pos'][1])**2
            ) < threshold:

                neighbors = H.neighbors(another)

                for neighbor in neighbors:
                    H.add_edge(node, neighbor)

                H.remove_node(another)

# %%
# Now, we label the components (faults) in the network:

for label, cc in enumerate(sorted(nx.connected_components(H))):
    for n in cc:
        H.nodes[n]['component'] = label

# %%
# and generate palette of colors for the labels:


n_comp = 100

palette = sns.color_palette(None, 2*n_comp)

node_color = np.zeros((len(H), 3))

for n, node in enumerate(H):
    color = palette[H.nodes[node]['component']]

    node_color[n, 0] = color[0]
    node_color[n, 1] = color[1]
    node_color[n, 2] = color[2]

# %%
# Finally, we can plot our fault network:

fig, axs = plt.subplots(1, 1, figsize=(20, 12))

axs.imshow(img)

nx.draw(H,
        pos=nx.get_node_attributes(G, 'pos'),
        node_size=20,
        node_color=node_color,
        with_labels=False,
        font_size=15,
        ax=axs)
plt.show()
