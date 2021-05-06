import math
import pickle

import faultanalysistoolbox.metrics as fatb_metrics
import faultanalysistoolbox.plots as fatb_plots
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ipywidgets import Layout, interactive, widgets
from tqdm import tqdm

# %%
# ## Load and visualize networks

# %%
# First, we load our fault networks extracted from a 2-D model over several
# timesteps:
Gs = []

for n in range(3, 50):
    Gs.append(
        pickle.load(
            open(
                '/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/3-time_stepping/graphs/g_' + str(n) + '.p',
                'rb'
            )
        )
    )

# %%
# Now we can visualize and compare the faults extracted through time:


def f(time):
    fig, ax = plt.subplots(figsize=(16, 4))
    fatb_plots.plot_components(Gs[time], ax, label=True)
    plt.xlim([1000, 3500])
    plt.ylim([600, 0])
    plt.show()


interactive_plot = interactive(f, time=widgets.IntSlider(
    min=3, max=49, step=1, layout=Layout(width='900px')))
output = interactive_plot.children[-1]
output.layout.width = '1000px'
interactive_plot

# %%
# The faults have different labels in every time steps. That's a result of the
# extraction, where arbitrary labels are assigned to the faults. To track
# faults through time, we want these labels to be consistent between time
# steps. So let's look at two consecutive time steps:

G_0 = Gs[0]
G_1 = Gs[1]

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_components(G_0, ax, label=True)
plt.title('time 0')
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

fig, ax = plt.subplots(figsize=(16, 4))
plt.title('time 1')
fatb_plots.plot_components(G_1, ax, label=True)
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
# ## Correlate two time steps
#
# We can already see that the green fault 12 at time 0 is probably the same as
# the red fault 13 at time 1. So in order to match them, we will compare all
# faults between two time steps and re-label them.
#
# But first we will introduce a new label. We typically extract faults as
# components of the graph (i.e. nodes connected to one another). But geological
# faults can consist of multiple of such components, such as faults 22 and 23
# at time 0 - they probably belong to one fault. So let's introduce this new
# fault label:

for node in G_0:
    G_0.nodes[node]['fault'] = G_0.nodes[node]['component']

for node in G_1:
    G_1.nodes[node]['fault'] = G_1.nodes[node]['component']

print(G_0.nodes[0])

# %%
# Now each node has an additional property 'fault' that we will re-label later
# on. But first we want to get the fault labels for the first two time steps:


def get_fault_labels(G):
    labels = set()
    for node in G:
        labels.add(G.nodes[node]['fault'])
    return sorted(list(labels))


fault_labels_0 = get_fault_labels(G_0)
fault_labels_1 = get_fault_labels(G_1)

print(fault_labels_0)

# %%
# You can check that these are all the faults labelled in the image above.
#
# Next we want to get the polarities of the faults, i.e. dipping to the left or
# the right. We computed the polarity during the extraction to help us
# differentiate cross-cutting faults.


def get_fault(G, n):
    nodes = [node for node in G if G.nodes[node]['fault'] == n]
    return G.subgraph(nodes)


def get_polarity(G):
    for node in G:
        polarity = G.nodes[node]['polarity']
        break
    return polarity


def get_fault_polarities(G):
    labels = get_fault_labels(G)
    polarities = []
    for label in labels:
        G_fault = get_fault(G, label)
        polarities.append(get_polarity(G_fault))
    return polarities


fault_polarities_0 = get_fault_polarities(G_0)
fault_polarities_1 = get_fault_polarities(G_1)

print('faults')
print(fault_labels_0)
print('polarities')
print(fault_polarities_0)

# %%
# Let's check if these polarities are correct:

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_components(G_0, ax, label=True)
plt.title('faults')
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

fig, ax = plt.subplots(figsize=(17.5, 4))
fatb_plots.plot_attribute(G_0, 'polarity', ax)
plt.title('polarities')
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
# Looks alright! Now let's convert our graphs to point clouds, which we can
# compare easily:


def G_to_pts(G):
    labels = get_fault_labels(G)
    point_set = []
    for label in labels:
        G_fault = get_fault(G, label)
        points = []
        for node in G_fault:
            points.append(G_fault.nodes[node]['pos'])
        point_set.append(points)
    return point_set


pt_set_0 = G_to_pts(G_0)
pt_set_1 = G_to_pts(G_1)

# %%
# Now it's getting interesting. To correlate faults across time steps, we want
# to check if a fault from the time step 0 is within a fault from time step 1
# and vice versa. This allows us to correlate faults even if they merge or
# split up. We do this by:
#
# 1.   Calculating the distance matrix between each node of fault A and B, i.e.
#      the point clouds
# 2.   Calculating the minimum distance of the closest nodes
# 3.   Computing the mean of these distances
# 4.   Using a threshold radius R to decide whether a fault A is within fault B
#
#


def is_A_in_B(set_A, set_B, R):
    distances = np.zeros((len(set_A), len(set_B)))
    for n, pt_0 in enumerate(set_A):
        for m, pt_1 in enumerate(set_B):
            distances[n, m] = math.sqrt(
                (pt_0[0]-pt_1[0])**2 + (pt_0[1]-pt_1[1])**2)
    if np.mean(np.min(distances, axis=1)) > R:
        return False
    else:
        return True

# %%
# Now we use this function for all faults with the same polarities and in both
# directions (i.e. A is in B, B is in A) between the two time steps:


R = 10

correlations = set()
for n in range(len(fault_labels_0)):
    for m in range(len(fault_labels_1)):
        if fault_polarities_0[n] == fault_polarities_1[m]:
            if is_A_in_B(pt_set_0[n], pt_set_1[m], R):
                correlations.add((fault_labels_0[n], fault_labels_1[m]))
            if is_A_in_B(pt_set_1[m], pt_set_0[n], R):
                correlations.add((fault_labels_0[n], fault_labels_1[m]))

print(correlations)

# %%
# We can check if the correlations, we derived are correct. The third
# correlation for example says that fault 3 at time 0 is the same as fault 0 at
# time 1. Is that correct?

fig, ax = plt.subplots(figsize=(16, 4))
fatb_plots.plot_components(G_0, ax, label=True)
plt.title('time 0')
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

fig, ax = plt.subplots(figsize=(16, 4))
plt.title('time 1')
fatb_plots.plot_components(G_1, ax, label=True)
plt.xlim([1000, 3500])
plt.ylim([600, 0])
plt.show()

# %%
# Yeah, looks right! So let's relabel these faults consistently using these
# correlations. When doing this we need to keep track of the faults, we've
# correlated. To do this, we first set them all to uncorrelated:

for node in G_1:
    G_1.nodes[node]['correlated'] = False

# %%
# Next, we sort the correlations by fault length, so that when two faults
# merge, we preserve the larger one.

lengths = [fatb_metrics.total_length(get_fault(G_0, correlation[0]))
           for correlation in correlations]
lengths, correlations = zip(*sorted(zip(lengths, correlations)))

# %%
# Remember you can find all functions such as `total_length` in the folder:
#
# https://github.com/thilowrona/fault_analysis_toolbox/tree/master/code
# %%
# Now we relabel the faults using correlations:

for node in G_1:
    for correlation in correlations:
        if G_1.nodes[node]['component'] == correlation[1]:
            G_1.nodes[node]['fault'] = correlation[0]
            G_1.nodes[node]['correlated'] = True

# %%
# Finally, we relabel the uncorrelated faults to avoid confusion:

max_comp = max(get_fault_labels(G_1))

G_1_sub = nx.subgraph(
    G_1, [node for node in G_1 if G_1.nodes[node]['correlated'] is False])
for label, cc in enumerate(sorted(nx.connected_components(G_1_sub))):
    for n in cc:
        G_1.nodes[n]['fault'] = label+max_comp+1

# %%
# Now let's check if it worked:

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
# Awesome! It worked! You can see that faults in the same location have the
# same label.
# %%
# ## Correlate multiple time steps
# Alright, let's do this for all 50 time steps:

max_comp = 0

for time in tqdm(range(len(Gs)-1)):
    G_0 = Gs[time]
    G_1 = Gs[time+1]

    if n == 0:
        for node in G_0:
            G_0.nodes[node]['fault'] = G_0.nodes[node]['component']
        Gs[time] = G_0

    for node in G_1:
        G_1.nodes[node]['fault'] = G_1.nodes[node]['component']

    # Correlation
    fault_labels_0 = get_fault_labels(G_0)
    fault_labels_1 = get_fault_labels(G_1)

    fault_polarities_0 = get_fault_polarities(G_0)
    fault_polarities_1 = get_fault_polarities(G_1)

    pt_set_0 = G_to_pts(G_0)
    pt_set_1 = G_to_pts(G_1)

    R = 10
    correlations = set()
    for n in range(len(fault_labels_0)):
        for m in range(len(fault_labels_1)):
            if fault_polarities_0[n] == fault_polarities_1[m]:
                if is_A_in_B(pt_set_0[n], pt_set_1[m], R):
                    correlations.add((fault_labels_0[n], fault_labels_1[m]))
                if is_A_in_B(pt_set_1[m], pt_set_0[n], R):
                    correlations.add((fault_labels_0[n], fault_labels_1[m]))

    # Relabel
    for node in G_1:
        G_1.nodes[node]['correlated'] = False

    lengths = [fatb_metrics.total_length(get_fault(G_0, correlation[0]))
               for correlation in correlations]
    lengths, correlations = zip(*sorted(zip(lengths, correlations)))

    for node in G_1:
        for correlation in correlations:
            if G_1.nodes[node]['component'] == correlation[1]:
                G_1.nodes[node]['fault'] = correlation[0]
                G_1.nodes[node]['correlated'] = True

    max_comp = max(get_fault_labels(G_1))

    G_1_sub = nx.subgraph(
        G_1, [node for node in G_1 if G_1.nodes[node]['correlated'] is False])
    for label, cc in enumerate(sorted(nx.connected_components(G_1_sub))):
        for n in cc:
            G_1.nodes[n]['fault'] = label+max_comp+1

    Gs[time+1] = G_1


# %%
# and let's look at the results:
def f(time):
    fig, ax = plt.subplots(figsize=(16, 4))
    fatb_plots.plot_faults(Gs[time], ax, label=True)
    plt.xlim([1000, 3500])
    plt.ylim([600, 0])
    plt.show()


interactive_plot = interactive(f, time=widgets.IntSlider(
    min=3, max=49, step=1, layout=Layout(width='900px')))
output = interactive_plot.children[-1]
output.layout.width = '1000px'
interactive_plot

# %%
# Fantastic! Now that we have correlated the faults, we can compute slip and
# displacement through time to investigate the growth of these faults. The next
# tutorial shows how to calculate fault slip from velocities.
