import math
import pickle

import faultanalysistoolbox.metrics as fatb_metrics
import faultanalysistoolbox.plots as fatb_plots
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# %%
# ## Load and plot faults
# %%
# First, we load our fault networks extracted from a 2-D model over several
# timesteps:

G = pickle.load(
    open(
        '/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/4-fault_slip/graphs/g_3.p',
        'rb'
    )
)

# %%
# Now we can visualize these faults:

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_faults(G, ax, label=True)
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
# Load velocities in x- and z-direction:
data = np.load('/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/4-fault_slip/velocity.npy')
v_x = data[:, :, 0]
v_z = data[:, :, 1]

# %%
# Visualize the velocity in z-direction:

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_faults(G, ax, label=True)
ax.matshow(v_z)
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
# ## Extract velocities
# Now we want to pick up the velocity left and right of each fault, but to do
# this we first need to calculate the direction of the fault:


def calculate_direction(G, cutoff, normalize=True):

    for node in G.nodes:
        length = nx.single_source_shortest_path_length(G, node, cutoff=cutoff)
        keys = [keys for keys, values in length.items() if values ==
                max(length.values())]
        if len(keys) > 2:
            (node_0, node_1) = keys[:2]
        if len(keys) == 2:
            (node_0, node_1) = keys
        if len(keys) == 1:
            node_0 = keys[0]

            length = nx.single_source_shortest_path_length(
                G, node, cutoff=cutoff-1)
            keys = [keys for keys, values in length.items() if values ==
                    max(length.values())]

            node_1 = keys[0]

        # extrac position
        pt_0 = G.nodes[node_0]['pos']
        pt_1 = G.nodes[node_1]['pos']
        # calculate vector
        dx = pt_0[0] - pt_1[0]
        dy = pt_0[1] - pt_1[1]
        # write to graph
        G.nodes[node]['dx'] = dx
        G.nodes[node]['dy'] = dy

    return G


G = calculate_direction(G, 3)

# %%
# Now we use the direction to calculate the pick-up points:


def calculate_pickup_points(G, factor):

    H = nx.Graph()

    for node in G.nodes:

        (x, y) = G.nodes[node]['pos']

        dx = G.nodes[node]['dx']
        dy = G.nodes[node]['dy']

        dx = factor * dx
        dy = factor * dy

        x_p = int(x - dy)
        y_p = int(y + dx)

        x_n = int(x + dy)
        y_n = int(y - dx)

        node_mid = (node, 0)
        H.add_node(node_mid)
        H.nodes[node_mid]['pos'] = (x, y)
        H.nodes[node_mid]['component'] = -1

        node_p = (node, 1)
        H.add_node(node_p)
        H.nodes[node_p]['pos'] = (x_p, y_p)
        H.nodes[node_p]['component'] = -2

        node_n = (node, 2)
        H.add_node(node_n)
        H.nodes[node_n]['pos'] = (x_n, y_n)
        H.nodes[node_n]['component'] = -3

        H.add_edge(node_n, node_p)

    return H


H = calculate_pickup_points(G, 1)

# %%
#

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_components(H, ax, label=False)
ax.matshow(v_z)
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
#

fig, ax = plt.subplots(figsize=(16, 8))
fatb_plots.plot_components(H, ax, label=False)
ax.matshow(v_z)
plt.xlim([2200, 2350])
plt.ylim([150, 75])
plt.show()


def extract_attribute(G, image, name):
    (x_max, y_max) = image.shape
    for node in G:
        y, x = G.nodes[node]['pos']
        if x >= x_max or y >= y_max:
            G.nodes[node][name] = float('nan')
        else:
            G.nodes[node][name] = image[int(x), int(y)]
    return G


H = extract_attribute(H, v_x, 'v_x')
H = extract_attribute(H, v_z, 'v_z')


def filter_pickup_points(G, H):
    for node in G:
        if (
            (H.nodes[(node, 1)]['pos'][1] < 0) or
            (H.nodes[(node, 2)]['pos'][1] < 0)
        ):

            H.nodes[(node, 0)]['v_x'] = 0
            H.nodes[(node, 0)]['v_z'] = 0

            H.nodes[(node, 1)]['v_x'] = 0
            H.nodes[(node, 1)]['v_z'] = 0

            H.nodes[(node, 2)]['v_x'] = 0
            H.nodes[(node, 2)]['v_z'] = 0

    return H


H = filter_pickup_points(G, H)

# %%
# ## Compute slip and slip rate


def calculate_slip_rate(G, H, dim):
    if dim == 2:
        for node in H.nodes:
            if node[1] == 0:    # centre point

                if (
                    (H.nodes[(node[0], 1)]['v_x'] == 0) or
                    (H.nodes[(node[0], 2)]['v_x'] == 0)
                ):
                    G.nodes[node[0]]['slip_rate_x'] = 0
                    G.nodes[node[0]]['slip_rate_z'] = 0
                    G.nodes[node[0]]['slip_rate'] = 0
                else:
                    G.nodes[node[0]]['slip_rate_x'] = abs(
                        H.nodes[(node[0], 1)]['v_x'] -
                        H.nodes[(node[0], 2)]['v_x']
                    )
                    G.nodes[node[0]]['slip_rate_z'] = abs(
                        H.nodes[(node[0], 1)]['v_z'] -
                        H.nodes[(node[0], 2)]['v_z']
                    )
                    G.nodes[node[0]]['slip_rate'] = math.sqrt(
                        G.nodes[node[0]]['slip_rate_x']**2 +
                        G.nodes[node[0]]['slip_rate_z']**2
                    )
    if dim == 3:
        for node in H.nodes:
            if node[1] == 0:    # centre point
                # Outside of the box
                if (
                    (H.nodes[(node[0], 1)]['v_x'] == 0) or
                    (H.nodes[(node[0], 2)]['v_x'] == 0)
                ):
                    G.nodes[node[0]]['slip_rate_x'] = 0
                    G.nodes[node[0]]['slip_rate_y'] = 0
                    G.nodes[node[0]]['slip_rate_z'] = 0
                    G.nodes[node[0]]['slip_rate'] = 0
                # Inside the box
                else:
                    G.nodes[node[0]]['slip_rate_x'] = abs(
                        H.nodes[(node[0], 1)]['v_x'] -
                        H.nodes[(node[0], 2)]['v_x']
                    )
                    G.nodes[node[0]]['slip_rate_y'] = abs(
                        H.nodes[(node[0], 1)]['v_y'] -
                        H.nodes[(node[0], 2)]['v_y']
                    )
                    G.nodes[node[0]]['slip_rate_z'] = abs(
                        H.nodes[(node[0], 1)]['v_z'] -
                        H.nodes[(node[0], 2)]['v_z']
                    )
                    G.nodes[node[0]]['slip_rate'] = math.sqrt(
                        G.nodes[node[0]]['slip_rate_x']**2 +
                        G.nodes[node[0]]['slip_rate_y']**2 +
                        G.nodes[node[0]]['slip_rate_z']**2
                    )

    return G


G = calculate_slip_rate(G, H, dim=2)


def calculate_slip(G, H, dt, dim):
    if dim == 2:
        for node in H.nodes:
            if node[1] == 0:

                if (
                    (H.nodes[(node[0], 1)]['v_x'] == 0) or
                    (H.nodes[(node[0], 2)]['v_x'] == 0)
                ):
                    G.nodes[node[0]]['slip_x'] = 0
                    G.nodes[node[0]]['slip_z'] = 0
                    G.nodes[node[0]]['slip'] = 0
                else:
                    G.nodes[node[0]]['slip_x'] = abs(
                        H.nodes[(node[0], 1)]['v_x'] -
                        H.nodes[(node[0], 2)]['v_x']
                    )*dt
                    G.nodes[node[0]]['slip_z'] = abs(
                        H.nodes[(node[0], 1)]['v_z'] -
                        H.nodes[(node[0], 2)]['v_z']
                    )*dt
                    G.nodes[node[0]]['slip'] = math.sqrt(
                        G.nodes[node[0]]['slip_x']**2 +
                        G.nodes[node[0]]['slip_z']**2
                    )

    if dim == 3:
        for node in H.nodes:
            if node[1] == 0:
                if (
                    (H.nodes[(node[0], 1)]['v_x'] == 0) or
                    (H.nodes[(node[0], 2)]['v_x'] == 0)
                ):
                    G.nodes[node[0]]['slip_x'] = 0
                    G.nodes[node[0]]['slip_y'] = 0
                    G.nodes[node[0]]['slip_z'] = 0
                    G.nodes[node[0]]['slip'] = 0
                else:
                    G.nodes[node[0]]['slip_x'] = abs(
                        H.nodes[(node[0], 1)]['v_x'] -
                        H.nodes[(node[0], 2)]['v_x']
                    )*dt
                    G.nodes[node[0]]['slip_y'] = abs(
                        H.nodes[(node[0], 1)]['v_y'] -
                        H.nodes[(node[0], 2)]['v_y']
                    )*dt
                    G.nodes[node[0]]['slip_z'] = abs(
                        H.nodes[(node[0], 1)]['v_z'] -
                        H.nodes[(node[0], 2)]['v_z']
                    )*dt
                    G.nodes[node[0]]['slip'] = math.sqrt(
                        G.nodes[node[0]]['slip_x']**2 +
                        G.nodes[node[0]]['slip_y']**2 +
                        G.nodes[node[0]]['slip_z']**2
                    )
    return G


G = calculate_slip(G, H, dim=2, dt=94804)

# %%
# ## Visualization
#

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_attribute(G, 'slip', ax)
ax.matshow(v_x, cmap='gray')
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()


def get_fault_labels(G):
    labels = set()
    for node in G:
        labels.add(G.nodes[node]['fault'])
    return sorted(list(labels))


def get_fault(G, n):
    nodes = [node for node in G if G.nodes[node]['fault'] == n]
    return G.subgraph(nodes)


labels = get_fault_labels(G)

lengths = []
slips = []
for label in labels:
    fault = get_fault(G, label)
    lengths.append(fatb_metrics.total_length(fault))
    slips.append(np.max([fault.nodes[node]['slip'] for node in fault]))


G.nodes[10]
plt.figure(figsize=(12, 8))
plt.scatter(lengths, slips, c=slips, s=100, cmap='seismic', vmin=0)
plt.xlabel('Slip')
plt.ylabel('Length')
plt.axis('equal')
cbar = plt.colorbar()
cbar.set_label('Slip')
plt.show()
