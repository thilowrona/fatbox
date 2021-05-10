import pickle

import faultanalysistoolbox.edits as fatb_edits
import faultanalysistoolbox.plots as fatb_plots
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Layout, interactive, widgets
from tqdm import tqdm

# %%
# ## Load and plot faults
# %%
# First, we load our fault networks extracted from a 2-D model over several
# timesteps:

Gs = []
for n in range(3, 50):
    Gs.append(pickle.load(open(
        '/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/5-displacement/graphs/g_' + str(n) + '.p', 'rb')))


# %%
# Now we can visualize these faults:
def f(time):
    fig, ax = plt.subplots(figsize=(16, 4))
    fatb_plots.plot_attribute(Gs[time], 'slip', ax)
    plt.xlim([1000, 3500])
    plt.ylim([600, 0])
    plt.show()


interactive_plot = interactive(f, time=widgets.IntSlider(
    min=3, max=49, step=1, layout=Layout(width='900px')))
output = interactive_plot.children[-1]
output.layout.width = '1000px'
interactive_plot

# %%
# ## Sum up two time steps

G_0 = Gs[0]
G_1 = Gs[1]

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_attribute(G_0, 'slip', ax)
plt.title('time 0')
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

fig, ax = plt.subplots(figsize=(16, 4))
plt.title('time 1')
fatb_plots.plot_attribute(G_1, 'slip', ax)
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
# First we just copy over slip to displacement:


def write_slip_to_displacement(G, dim):
    if dim == 2:
        for node in G:
            G.nodes[node]['heave'] = G.nodes[node]['slip_x']
            G.nodes[node]['throw'] = G.nodes[node]['slip_z']
            G.nodes[node]['displacement'] = G.nodes[node]['slip']

    if dim == 3:
        for node in G:
            G.nodes[node]['heave'] = G.nodes[node]['slip_x']
            G.nodes[node]['lateral'] = G.nodes[node]['slip_y']
            G.nodes[node]['throw'] = G.nodes[node]['slip_z']
            G.nodes[node]['displacement'] = G.nodes[node]['slip']
    return G


G_0 = write_slip_to_displacement(G_0, dim=2)
G_1 = write_slip_to_displacement(G_1, dim=2)

print(G_0.nodes[10])

# %%
# Now we find the faults common to both time steps:


def get_fault_labels(G):
    labels = set()
    for node in G:
        labels.add(G.nodes[node]['fault'])
    return sorted(list(labels))


def common_faults(G, H):
    C_G = get_fault_labels(G)
    C_H = get_fault_labels(H)
    return list(set(C_G) & set(C_H))


cf = common_faults(G_0, G_1)
print(cf)

# %%
# Let's check:

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_faults(G_0, ax, label=True)
plt.title('time 0')
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

fig, ax = plt.subplots(figsize=(16, 4))
plt.title('time 1')
fatb_plots.plot_faults(G_1, ax, label=True)
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
# Looks good!
#
# Now we need two more functions: one to get a fault and one to get the
# displacement with coordinates of the fault.


def get_fault(G, n):
    nodes = [node for node in G if G.nodes[node]['fault'] == n]
    return G.subgraph(nodes)


def get_displacement(G, dim):
    if dim == 2:
        points = np.zeros((len(list(G)), 6))
        for n, node in enumerate(G):
            points[n, 0] = node
            points[n, 1] = G.nodes[node]['pos'][0]
            points[n, 2] = G.nodes[node]['pos'][1]
            points[n, 3] = G.nodes[node]['heave']
            points[n, 4] = G.nodes[node]['throw']
            points[n, 5] = G.nodes[node]['displacement']
    if dim == 3:
        points = np.zeros((len(list(G)), 7))
        for n, node in enumerate(G):
            points[n, 0] = node
            points[n, 1] = G.nodes[node]['pos'][0]
            points[n, 2] = G.nodes[node]['pos'][1]
            points[n, 3] = G.nodes[node]['heave']
            points[n, 4] = G.nodes[node]['lateral']
            points[n, 5] = G.nodes[node]['throw']
            points[n, 6] = G.nodes[node]['displacement']
    return points

# %%
# Now let's:
#
# 1.   Go through the common faults
# 2.   Get their displacement with coordinates
# 3.   Find the closest points
# 4.   Add the displacement from time 0 to time 1


for fault in cf:

    points_0 = get_displacement(get_fault(G_0, fault), dim=2)
    points_1 = get_displacement(get_fault(G_1, fault), dim=2)

    for n in range(points_1.shape[0]):
        index = fatb_edits.closest_node(points_1[n, 1:3], points_0[:, 1:3])

        points_1[n, 3] += points_0[index][3]
        points_1[n, 4] += points_0[index][4]
        points_1[n, 5] += points_0[index][5]

    G_1 = fatb_edits.assign_displacement(G_1, points_1, dim=2)


fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_attribute(G_0, 'displacement', ax)
plt.title('time 0')
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

fig, ax = plt.subplots(figsize=(16, 4))
plt.title('time 1')
fatb_plots.plot_attribute(G_1, 'displacement', ax)
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
# Okay, they still look similar... but the colorbar stretches much further, so
# the displacement at time 1 really includes the slip from time 0!
#
# Now let's do this for all time steps:

max_comp = 0

for time in tqdm(range(len(Gs)-1)):

    G_0 = Gs[time]
    G_1 = Gs[time+1]

    if time == 0:
        G_0 = write_slip_to_displacement(G_0, dim=2)
        Gs[time] = G_0

    G_1 = write_slip_to_displacement(G_1, dim=2)

    cf = common_faults(G_0, G_1)

    for fault in cf:

        points_0 = get_displacement(get_fault(G_0, fault), dim=2)
        points_1 = get_displacement(get_fault(G_1, fault), dim=2)

        for n in range(points_1.shape[0]):
            index = fatb_edits.closest_node(points_1[n, 1:3], points_0[:, 1:3])

            points_1[n, 3] += points_0[index][3]
            points_1[n, 4] += points_0[index][4]
            points_1[n, 5] += points_0[index][5]

        G_1 = fatb_edits.assign_displacement(G_1, points_1, dim=2)

    Gs[time+1] = G_1


def f(time):
    fig, ax = plt.subplots(figsize=(16, 4))
    fatb_plots.plot_attribute(Gs[time], 'displacement', ax)
    plt.xlim([1000, 3500])
    plt.ylim([600, 0])
    plt.show()


interactive_plot = interactive(f, time=widgets.IntSlider(
    min=3, max=45, step=1, layout=Layout(width='900px')))
output = interactive_plot.children[-1]
output.layout.width = '1000px'
interactive_plot
