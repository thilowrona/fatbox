# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% '
import math
import pickle

import faultanalysistoolbox.metrics as fatb_metrics
import faultanalysistoolbox.plots as fatb_plots
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# %%
# ## Load fault network
# %%
# First, we load a fault network extracted from an oblique rift model:

G = pickle.load(
    open(
        '/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/' +
        'examples/2-fault_properties/g_27.p',
        'rb'
    )
)

# %%
# Now we can visualize the faults and look at the orientations:

fig, ax = plt.subplots()
fatb_plots.plot_components(G, ax, label=True)
ax.axis('equal')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
plt.show()

# %%
# We can see that most faults are striking N-S and a few E-W. Let's calculate
# the strike of the edges of the network:

for edge in G.edges:
    x1 = G.nodes[edge[0]]['pos'][0]
    x2 = G.nodes[edge[1]]['pos'][0]
    y1 = G.nodes[edge[0]]['pos'][1]
    y2 = G.nodes[edge[1]]['pos'][1]

    if (x2-x1) < 0:
        G.edges[edge]['strike'] = math.degrees(
            math.atan2((x2-x1), (y2-y1))) + 360
    else:
        G.edges[edge]['strike'] = math.degrees(math.atan2((x2-x1), (y2-y1)))

# %%
# Let's plot these strikes as edge attributes using this function:


def plot_edge_attribute(G, attribute, ax=[]):

    if ax == []:
        fig, ax = plt.subplots()

    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_size=0.001,
            ax=ax)

    nx.draw_networkx_edges(G,
                           pos=nx.get_node_attributes(G, 'pos'),
                           edge_color=np.array(
                               [G.edges[edge][attribute] for edge in G.edges]),
                           edge_cmap=plt.cm.twilight_shifted,
                           ax=ax)
    ax.axis('equal')

    # Colorbar
    cmap = plt.cm.twilight_shifted
    vmax = fatb_metrics.max_value_edges(G, attribute)
    vmin = fatb_metrics.min_value_edges(G, attribute)

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(attribute, rotation=270)


fig, ax = plt.subplots()
plot_edge_attribute(G, 'strike', ax)

# %%
# Looks a bit odd, let's zoom in:

fig, ax = plt.subplots()
plot_edge_attribute(G, 'strike', ax)
ax.set_ylim([50, 150])
plt.show()

# %%
# Okay, the edges have very discrete strikes (0, 45, 90 degrees), because they
# were extracted from a regular array. This becomes even more evident when we
# plot the Rose diagram (a polar histogram of the strikes):


def plot_rose(strikes, lengths=[], ax=[]):

    if lengths == []:
        lengths = np.ones_like(np.array(strikes))

    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(
        strikes, bin_edges, weights=lengths)
    number_of_strikes[0] += number_of_strikes[-1]
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])

    cmap = plt.cm.twilight_shifted(np.concatenate(
        (np.linspace(0, 1, 18), np.linspace(0, 1, 18)), axis=0))

    if ax == []:
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='polar')

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))

    ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves,
           width=np.deg2rad(10), bottom=0.0, color=cmap, edgecolor='k')

    #    ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight=
    #    'black')
    ax.set_title('Rose Diagram', y=1.10, fontsize=15)

    # fig.tight_layout()


strikes = [G.edges[edge]['strike'] for edge in G.edges]
plot_rose(strikes)

# %%
# Okay, that's not really what we want. We want the strikes of the faults, not
# the edges. So let's calculate those:

strikes = []
for cc in nx.connected_components(G):
    edges = G.edges(cc)
    edge_strikes = []
    for edge in edges:
        edge_strikes.append(G.edges[edge]['strike'])
    strikes.append(np.mean(edge_strikes))

# %%
# and the fault lenghts as weights:

for edge in G.edges:
    G.edges[edge]['length'] = np.linalg.norm(
        np.array(G.nodes[edge[0]]['pos'])-np.array(G.nodes[edge[1]]['pos']))


lengths = []
for cc in nx.connected_components(G):
    edges = G.edges(cc)
    edge_lengths = []
    for edge in edges:
        edge_lengths.append(G.edges[edge]['length'])
    lengths.append(np.mean(edge_lengths))

# %%
# Now we can plot the Rose diagram again:

plot_rose(strikes, lengths)

# %%
# That's a much better representation of the fault strikes in our network!

fig, ax = plt.subplots()
fatb_plots.plot_components(G, ax, label=True)
ax.axis('equal')
ax.set_xlabel('X-coordinate')
ax.set_ylabel('Y-coordinate')
plt.show()
