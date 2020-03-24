import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import cv_algorithms
import networkx as nx
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


#==============================================================================
# Pre-processing
#==============================================================================

# Define grid spacing
dx = 1250
dy = 1250


# Input file
filename = "v5w2h55_z95_20MYR.csv"

# Load file
df = pd.read_csv(filename)
df = df.rename({'Points:0': 'X', 'Points:1': 'Y', 'Points:2': 'Z'}, axis='columns')  
    
    
# Extract property
x=df.X
y=df.Y
z=df.Z
strain_rate=df.strain_rate
plastic_strain=df.plastic_strain


# Compute coordinates
xmin, xmax = min(x), max(x)
ymin, ymax = min(y), max(y)

Nx = int((xmax-xmin)/dx)
Ny = int((ymax-ymin)/dy)

# Define grid
xi = np.linspace(xmin, xmax, Nx+1)
yi = np.linspace(ymin, ymax, Ny+1)


## Interpolate property on to grid
x_arr, y_arr = np.meshgrid(xi, yi)
z_arr = df.Z.mean()*np.ones((Nx+1,Ny+1))

strain_rate_arr = griddata((x, y), strain_rate, (xi[None,:], yi[:,None]), method='cubic')
plastic_strain_arr = griddata((x, y), plastic_strain, (xi[None,:], yi[:,None]), method='cubic')





#==============================================================================
# Fault network extraction 
#==============================================================================


## THRESHOLDING
value = 3e-15
threshold = np.where(strain_rate_arr > value, 1, 0)
threshold = np.uint8(threshold)


## SKELETONIZE
skeleton = cv_algorithms.guo_hall(threshold)
skeleton[0, :] = skeleton[1, :]
skeleton[-1,:] = skeleton[-2,:]
skeleton[:, 0] = skeleton[:, 1]
skeleton[:,-1] = skeleton[:,-2]


## EXTRACT POINTS
points = np.where(skeleton != 0)


## EXTRACT GRAPH
N = len(points[0])

# Set up graph
G = nx.Graph()

# Add nodes
for n in range(N):
    G.add_node(n, pos=(points[1][n], points[0][n]))

# Add edges to graph
def add_edges(G, N):   
 
    def distance_between_nodes(G, node0, node1):
        (x0, y0) = G.nodes[node0]['pos']
        (x1, y1) = G.nodes[node1]['pos']
        return math.sqrt((x0-x1)**2+(y0-y1)**2)


    def find_closest(G, node):
        threshold = 1000000
        for other in G:
            d = distance_between_nodes(G, node, other)
            if 0 < d < threshold:
                threshold = d
                index = other
        return index
  

    for node in G:
        print(str(node) + ' of ' + str(N))
        closest = find_closest(G, node)
        if (closest, node) not in G.edges:
            G.add_edge(node, closest)


    def clostest_except(G, node, cn):
        threshold = 1000000
        for other in G:
            if other not in cn:
                d = distance_between_nodes(G, node, other)
                if 0 < d < threshold:
                    threshold = d
                    index = other
        return index, threshold


    for node in G:
        print(str(node) + ' of ' + str(N))
        if len(G.edges(node)) == 1:
            cn = nx.node_connected_component(G, node)
            index, threshold  = clostest_except(G, node, cn)
            if threshold < 2:
                G.add_edge(node, index)
    return G


G = add_edges(G, N)









#==============================================================================
# Fault network analysis
#==============================================================================

# Label components (i.e. faults)
for label, cc in enumerate(sorted(nx.connected_components(G))): 
    for n in cc:
        G.nodes[n]['component'] = label

# Number of faults
def number_of_components(G):
    return len(sorted(nx.connected_components(G)))        
        
nof = number_of_components(G)


# Plot network
def plot_components(G, label=True, box=True, ax = []):
    
    if ax == []:
        fig, ax = plt.subplots()
        
    n_comp = 1000
            
    palette = sns.color_palette(None, 2*n_comp)
    
    node_color = np.zeros((len(G),3))
    
    for n, node in enumerate(G):
        color = palette[G.nodes[node]['component']]
            
        node_color[n,0] = color[0]
        node_color[n,1] = color[1]
        node_color[n,2] = color[2]   
            
    
    nx.draw(G,
            pos = nx.get_node_attributes(G, 'pos'),
            node_color = node_color,
            node_size = 1,
            ax = ax)
    
    plt.axis('equal') 
    
    
    if label == True:
        
        for cc in sorted(nx.connected_components(G)): 
            # Calculate centre
            x_avg = 0
            y_avg = 0
            
            for n in cc:        
                y_avg = y_avg + G.nodes[n]['pos'][0]
                x_avg = x_avg + G.nodes[n]['pos'][1]
            
            N = len(cc)
            y_avg = y_avg/N
            x_avg = x_avg/N
            
            # Scale color map
            label = G.nodes[n]['component']
            
            ax.text(y_avg, x_avg, label, fontsize = 15, color = palette[G.nodes[n]['component']])
      
        
    if box == True:
        left, bottom, width, height = (0, 0, 704, 960)
        
        rect = patches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor='black', facecolor='none')
        
        ax.add_patch(rect)


# Plot network
fig, ax = plt.subplots(1, 1, figsize=(8,10))
plt.imshow(strain_rate_arr)
cb = plt.colorbar()
cb.ax.plot([0, 1], [value]*2, 'r')
plot_components(G, box=False, ax = ax)


# Extract properties
def extract_attribute(G, image, name):    
    for node in G:
        y,x = G.nodes[node]['pos']    
        G.nodes[node][name] = image[x,y]
    return G


G = extract_attribute(G, x_arr, 'x')
G = extract_attribute(G, y_arr, 'y')
G = extract_attribute(G, plastic_strain_arr, 'plastic_strain')


# Compoute fault lengths
def distance_between_nodes2(G, n0, n1):
    x0 = G.nodes[n0]['x']
    y0 = G.nodes[n0]['y']
    x1 = G.nodes[n1]['x']
    y1 = G.nodes[n1]['y']     
    return math.sqrt((x0 -x1)**2+(y0-y1)**2)


def compute_edge_length(G):
    for edge in G.edges:
        G.edges[edge]['length'] = distance_between_nodes2(G, edge[0], edge[1])
    return G


def total_length(G):    
    G = compute_edge_length(G)    
    length = 0
    for edge in G.edges:
        length = length + G.edges[edge]['length']
    return length


def select_component(G, component=0):    
    selected_nodes = [n[0] for n in G.nodes(data=True) if n[1]['component'] == component]  
    c = G.subgraph(selected_nodes)
    return c


fault_lengths = np.zeros(nof)
for n in range(nof):
    fault_lengths [n] = total_length(select_component(G, component=n))


# Compute max strain
def max_value_components(G, attribute):    
    max_values = np.zeros((number_of_components(G)))    
    for m, cc in enumerate(sorted(nx.connected_components(G))):        
        values = np.zeros((len(cc)))        
        for n, node in enumerate(cc):            
            values[n] = G.nodes[node][attribute]            
        max_values[m] = np.max(values)        
    return max_values

max_strain = max_value_components(G, 'plastic_strain')


# Plot fault length versus maximum strain
plt.figure()
plt.scatter(fault_lengths, max_strain)
plt.xlabel('Fault length (m)')
plt.ylabel('Max strain')













