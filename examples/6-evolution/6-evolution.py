import pickle

import faultanalysistoolbox.metrics as fatb_metrics
import faultanalysistoolbox.plots as fatb_plots
import faultanalysistoolbox.utils as fatb_utils
import matplotlib.pyplot as plt
import networkx as nx
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
        '/home/mrudolf/Nextcloud/GitRepos/fault_analysis_toolbox/examples/6-evolution/graphs/g_' + str(n) + '.p', 'rb')))


# %%
# Now we can visualize these faults:
def f(time):
    fig, ax = plt.subplots(figsize=(16, 4))
    fatb_plots.plot_faults(Gs[time], ax, label=False)
    plt.xlim([1000, 3500])
    plt.ylim([600, 0])
    plt.show()


interactive_plot = interactive(f, time=widgets.IntSlider(
    min=3, max=49, step=1, layout=Layout(width='900px')))
output = interactive_plot.children[-1]
output.layout.width = '1000px'
interactive_plot

# %%
# Note that you use the arrow keys to play the time forward and backward.
# %%
# ## Loop through time
# One of the easiest ways to follow the evolution of our faults is to track
# them through time. All we need to do is:
#
# 1.   Loop through time
# 2.   Get fault labels
# 3.   Visualize them
#
#

max_comp = 45

faults = np.zeros((max_comp, len(Gs)))
faults[:, :] = np.nan

for time in range(len(Gs)):
    G = Gs[time]
    labels = fatb_metrics.get_fault_labels(G)
    number_of_faults = len(labels)
    for n, label in enumerate(labels):
        fault = fatb_metrics.get_fault(G, label)
        faults[n, time] = label


# %%
# To visualize them, we can use these nice bar plots:
def bar_plot(attribute, faults, times, steps=[], ax=[]):

    colors = fatb_utils.get_colors()

    if ax == []:
        fig, ax = plt.subplots()

    if steps == []:
        steps = range(attribute.shape[1])

    for n, step in enumerate(steps):
        bottom = 0
        for m, fault in enumerate(faults[:, step]):
            if np.isfinite(fault):
                a = attribute[m, step]
                ax.bar(n, a, 1, bottom=bottom, alpha=0.75,
                       edgecolor='white', color=colors[int(fault), :])
                bottom += a
            else:
                break


steps = range(len(Gs))

fig, ax = plt.subplots(figsize=(16, 5))
bar_plot(np.ones_like(faults), faults, len(Gs), steps, ax)
ax.set_xlim([-0.5, 45.5])
ax.set_xlabel('Time')
ax.set_ylabel('Faults')
plt.show()

# %%
# Here we can see the fault labels as colors as well as the number of faults as
# histogram, but the plot doesn't capture the fault size.
#
# So let's add the fault length and displacement to the mix. First, we need to
# track these attributes through time:

max_comp = 45

faults = np.zeros((max_comp, len(Gs)))
faults[:, :] = np.nan

lengths = np.zeros((max_comp, len(Gs)))
lengths[:, :] = np.nan

displacements = np.zeros((max_comp, len(Gs)))
displacements[:, :] = np.nan

for time in range(len(Gs)):

    G = Gs[time]

    labels = fatb_metrics.get_fault_labels(G)
    number_of_faults = len(labels)

    for n, label in enumerate(labels):

        fault = fatb_metrics.get_fault(G, label)

        faults[n, time] = label
        lengths[n, time] = fatb_metrics.total_length(fault)
        displacements[n, time] = fatb_metrics.max_value_nodes(
            fault, 'displacement'
        )

    lengths[:, time] = np.sort(lengths[:, time])
    displacements[:, time] = np.sort(displacements[:, time])

# %%
# Now we can plot


def stack_plot(attribute, faults, times, steps=[], ax=[]):

    colors = fatb_utils.get_colors()

    if ax == []:
        fig, ax = plt.subplots()

    if steps == []:
        steps = range(attribute.shape[1])

    max_fault = int(np.nanmax(faults))

    x = np.arange(len(steps))
    y = np.zeros((max_fault, len(steps)))

    for N in range(max_fault):
        for n in steps:
            row = faults[:, n]
            if N in faults[:, n]:
                index = np.where(row == N)[0][0]
                y[N, n] = attribute[index, n]

    ax.stackplot(
        x, y,
        colors=colors[:max_fault, :],
        alpha=0.75,
        edgecolor='white',
        linewidth=0.5
    )


fig, ax = plt.subplots(figsize=(16, 5))
stack_plot(lengths, faults, len(Gs), steps, ax)
ax.set_xlim([0, 45])
ax.set_xlabel('Time')
ax.set_ylabel('Faults weighted by length')
plt.show()


fig, ax = plt.subplots(figsize=(16, 5))
stack_plot(displacements/1000, faults, len(Gs), steps, ax)
ax.set_xlim([0, 45])
ax.set_xlabel('Time')
ax.set_ylabel('Faults weighted by displacement')
plt.show()

# %%
# This is great! We can summarize the evolution of the entire fault system in
# one plot, but what we don't see is how individual faults grow. We know they
# consist of multiple segments (components), which merge and split up with
# time. How could we keep track of these processes?
#
# The basic idea is that we define another graph, which captures the evolution
# of the fault network through time. Let's start from here again:

time = 0

G_0 = Gs[time]
G_1 = Gs[time+1]

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
# Let's set up a graph (H), where each node corresponds to a fault segment
# (component) at a particular time. Moreover, we populate these nodes with some
# basic properties:

H = nx.Graph()


def H_add_nodes(H, G, time):

    for cc in sorted(nx.connected_components(G)):
        for node in cc:
            label = G.nodes[node]['component']
            H.add_node((time, label))

            H.nodes[(time, label)]['pos'] = (time, label)
            H.nodes[(time, label)]['time'] = time
            H.nodes[(time, label)]['fault'] = G.nodes[node]['fault']
            H.nodes[(time, label)]['component'] = G.nodes[node]['component']

            break

        G_cc = nx.subgraph(G, cc)

        H.nodes[(time, label)]['length'] = fatb_metrics.total_length(G_cc)
        H.nodes[(time, label)]['displacement'] = fatb_metrics.max_value_nodes(
            G_cc, 'displacement')

    return H


H = H_add_nodes(H, G_0, 0)
H = H_add_nodes(H, G_1, 1)

# %%
# Let's look at H:

plt.figure(figsize=(3, 6))
nx.draw(H,
        pos=nx.get_node_attributes(H, 'pos'),
        labels=nx.get_node_attributes(H, 'fault'),
        with_labels=True,
        node_color=fatb_plots.get_node_colors(H, 'fault'))

# %%
# Alright, just some nodes corresponding to the components at time 0 and 1
# labelled by the fault that they belong to.
#
# Now let's connect the components which belong to the same fault, i.e. have
# the same label in the above plot. For this purpose, we create a dictionary of
# the components belonging to each fault:


def get_dictionary(G):
    faults = fatb_metrics.get_fault_labels(G)
    dic = {}
    for fault in faults:
        G_fault = fatb_metrics.get_fault(G, fault)
        components = set()
        for node in G_fault:
            components.add(G_fault.nodes[node]['component'])
        dic[fault] = sorted(list(components))
    return dic


dic_0 = get_dictionary(G_0)
dic_1 = get_dictionary(G_1)

print(dic_0)

# %%
# Now we compute the faults common between consecutive time steps:

faults_0 = fatb_metrics.get_fault_labels(G_0)
faults_1 = fatb_metrics.get_fault_labels(G_1)

common_faults = list(set(faults_0).intersection(set(faults_1)))

# %%
# Next we loop through the common faults, get the components for time 0
# (starts) and time 1 (ends) and connect the components. This is a bit
# complicated because we can have different numbers of components, e.g.
# sometimes the fault has only one component at time 0 (len(starts)==1) and one
# at time 1 (len(ends)==1), then we can simply add an edge, but if there are
# multiple components to each fault, we need add multiple edges:

for fault in common_faults:
    starts = dic_0[fault]
    ends = dic_1[fault]
    if len(starts) == 1 and len(ends) == 1:
        H.add_edge((time, starts[0]), (time+1, ends[0]))
    elif len(starts) == 1 and len(ends) > 1:
        for end in ends:
            H.add_edge((time, starts[0]), (time+1, end))
    elif len(starts) > 1 and len(ends) == 1:
        for start in starts:
            H.add_edge((time, start), (time+1, ends[0]))
    elif len(starts) > 1 and len(ends) > 1:
        if len(starts) == len(ends):
            for start, end in zip(starts, ends):
                H.add_edge((time, start), (time+1, end))
        elif len(starts) < len(ends):
            minimum = min(len(starts), len(ends))
            difference = len(ends)-len(starts)
            for n in range(minimum):
                H.add_edge((time, starts[n]), (time+1, ends[n]))
            for n in range(difference):
                H.add_edge((time, starts[minimum-1]),
                           (time+1, ends[minimum-1+n+1]))
        elif len(starts) > len(ends):
            minimum = min(len(starts), len(ends))
            difference = len(starts)-len(ends)
            for n in range(minimum):
                H.add_edge((time, starts[n]), (time+1, ends[n]))
            for n in range(difference):
                H.add_edge((time, starts[minimum-1+n+1]),
                           (time+1, ends[minimum-1]))

# %%
# Now let's look at the result:

plt.figure(figsize=(3, 6))
nx.draw(H,
        pos=nx.get_node_attributes(H, 'pos'),
        labels=nx.get_node_attributes(H, 'fault'),
        with_labels=True,
        node_color=fatb_plots.get_node_colors(H, 'fault'))

# %%
# Excellent! Let's do this for all time steps:

H = nx.Graph()

for time in tqdm(range(0, len(Gs)-1)):

    G_0 = Gs[time]
    G_1 = Gs[time+1]

    if time == 0:
        H = H_add_nodes(H, G_0, time)
    H = H_add_nodes(H, G_1, time+1)

    faults_0 = fatb_metrics.get_fault_labels(G_0)
    faults_1 = fatb_metrics.get_fault_labels(G_1)

    dic_0 = get_dictionary(G_0)
    dic_1 = get_dictionary(G_1)

    faults = list(set(faults_0).intersection(set(faults_1)))

    for fault in faults:
        starts = dic_0[fault]
        ends = dic_1[fault]
        if len(starts) == 1 and len(ends) == 1:
            H.add_edge((time, starts[0]), (time+1, ends[0]))
        elif len(starts) == 1 and len(ends) > 1:
            for end in ends:
                H.add_edge((time, starts[0]), (time+1, end))
        elif len(starts) > 1 and len(ends) == 1:
            for start in starts:
                H.add_edge((time, start), (time+1, ends[0]))
        elif len(starts) > 1 and len(ends) > 1:
            if len(starts) == len(ends):
                for start, end in zip(starts, ends):
                    H.add_edge((time, start), (time+1, end))
            elif len(starts) < len(ends):
                minimum = min(len(starts), len(ends))
                difference = len(ends)-len(starts)
                for n in range(minimum):
                    H.add_edge((time, starts[n]), (time+1, ends[n]))
                for n in range(difference):
                    H.add_edge((time, starts[minimum-1]),
                               (time+1, ends[minimum-1+n+1]))
            elif len(starts) > len(ends):
                minimum = min(len(starts), len(ends))
                difference = len(starts)-len(ends)
                for n in range(minimum):
                    H.add_edge((time, starts[n]), (time+1, ends[n]))
                for n in range(difference):
                    H.add_edge(
                        (time, starts[minimum-1+n+1]), (time+1, ends[minimum-1]))


plt.figure(figsize=(18, 8))
nx.draw(H,
        pos=nx.get_node_attributes(H, 'pos'),
        labels=nx.get_node_attributes(H, 'fault'),
        with_labels=True,
        node_color=fatb_plots.get_node_colors(H, 'fault'))

# %%
# Now we start to get a grasp of the complexity involved in the evolution of
# the fault system. Let's just check out one fault:

fault = 10

H_sub = nx.subgraph(H, [node for node in H if H.nodes[node]['fault'] == fault])

fig, ax = plt.subplots(figsize=(18, 8))
nx.draw(H_sub,
        pos=nx.get_node_attributes(H_sub, 'pos'),
        labels=nx.get_node_attributes(H_sub, 'time'),
        with_labels=True,
        node_color=fatb_plots.get_node_colors(H_sub, 'fault'),
        ax=ax)
plt.show()

# %%
# Now we can see how the fault splits in two segments at time 12 and then
# merges again into one at time 13 and much more...
#
# For example, we can use the H-graph to plot the evolution of fault
# properties, such as length and displacement through time:

fault = 10
attribute_0 = 'length'
attribute_1 = 'displacement'

H_sub = nx.subgraph(H, [node for node in H if H.nodes[node]['fault'] == fault])

fig, ax = plt.subplots(figsize=(12, 12))
nx.draw(H_sub,
        pos=dict(
            zip(
                [node for node in H_sub],
                [
                    (
                        H_sub.nodes[node][attribute_0],
                        H_sub.nodes[node][attribute_1]
                    )
                    for node in H_sub
                ]
            )
        ),
        labels=nx.get_node_attributes(H_sub, 'time'),
        with_labels=True,
        node_color=fatb_plots.get_node_colors(H_sub, 'fault'),
        ax=ax)
ax.set_xlabel('Length')
ax.set_ylabel('Displacement')
ax.axis('on')
plt.show()

# %%
# We can see how the fault grows sometimes in length and sometimes in
# displacement, how it captures smaller segements (time 12) and how it dies out
# (time >30) becoming shorter and shorter.
#
# And we can plot these properties for all faults in the system:

attribute_0 = 'length'
attribute_1 = 'displacement'

fig, ax = plt.subplots(figsize=(12, 12))
nx.draw(H,
        pos=dict(zip([node for node in H], [
                 (H.nodes[node][attribute_0], H.nodes[node][attribute_1]) for node in H])),
        labels=nx.get_node_attributes(H, 'fault'),
        with_labels=True,
        node_color=fatb_plots.get_node_colors(H, 'fault'),
        ax=ax)
ax.set_xlabel('Length')
ax.set_ylabel('Displacement')
ax.axis('on')
plt.show()

# %%
# Last, we want to capture these processes in Sankey-type diagram, where the
# edge width of our graph H shows a fault property, e.g. displacement:

attribute = 'displacement'
factor = 0.00001
width = dict(zip([node for node in H], [
             H.nodes[node][attribute]*factor for node in H]))

fig, ax = plt.subplots(figsize=(16, 8))

fatb_plots.plot_width(H, ax, width, tips=False)

# %%
# Check this out! The long-lived blue line which thickens with time is fault
# 10, which we can track throughout our model run:


def f(time):
    fig, ax = plt.subplots(figsize=(16, 4))
    fatb_plots.plot_faults(Gs[time], ax, label=True)
    plt.xlim([1000, 3500])
    plt.ylim([600, 0])
    plt.show()


interactive_plot = interactive(
    f,
    time=widgets.IntSlider(
        min=3, max=49, step=1, layout=Layout(width='900px')
    )
)
output = interactive_plot.children[-1]
output.layout.width = '1000px'
interactive_plot
