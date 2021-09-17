# Packages
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from PIL import Image
from fatbox import metrics
#==============================================================================
# This file contains a series of function to plot fault networks (graphs). 
# This includes functions for plotting: 
# (1) arrays
# (2) networks
#==============================================================================



#******************************************************************************
# (1) Useful functions
# A couple of helper functions
#******************************************************************************

cmap = colors.ListedColormap(
    [
        '#ffffffff', '#64b845ff', '#9dcd39ff',
        '#efe81fff', '#f68c1bff', '#f01b23ff'
    ]
)


def get_node_colors(G, attribute, return_palette=False):
    """ Get node colors for plotting
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute name
    
    Returns
    -------  
    array : array
        Node colors
    """

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    n_comp  = 10000
    palette = sns.color_palette('husl', n_comp)
    palette = np.array(palette)
    
    # Shuffle
    perm = np.arange(palette.shape[0])
    np.random.seed(42)
    np.random.shuffle(perm)
    palette = palette[perm]
    
    node_color = np.zeros((len(G), 3))

    for n, node in enumerate(G):
        color = palette[G.nodes[node][attribute]]
        node_color[n, 0] = color[0]
        node_color[n, 1] = color[1]
        node_color[n, 2] = color[2]

    if return_palette:
        return node_color, palette
    else:
        return node_color




def get_edge_colors(G, attribute, return_palette=False):
    """ Get edge colors for plotting
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute name
    
    Returns
    -------  
    array : array
        Node colors
    """

    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"

    # Calculation
    n_comp = 10000
    palette = sns.color_palette('husl', n_comp)
    palette = np.array(palette)
    
    # Shuffle
    perm = np.arange(palette.shape[0])
    np.random.seed(42)
    np.random.shuffle(perm)
    palette = palette[perm]

    for n, edge in enumerate(G.edges):
        color = palette[G.edges[edge][attribute]]
        edge_color[n, 0] = color[0]
        edge_color[n, 1] = color[1]
        edge_color[n, 2] = color[2]

    if return_palette:
        return edge_color, palette
    else:
        return edge_color


#******************************************************************************
# (2) Array plotting
# A couple of functions to visulize arrays
#******************************************************************************

def plot_overlay(label, image, **kwargs):
    """ Plot a label onto of image
    
    Parameters
    ----------
    label : np.array
        Label
    image : np.array
        Image
    
    Returns
    -------  

    """

    label = (label-np.min(label))/(np.max(label)-np.min(label))

    label_rgb = np.zeros((label.shape[0], label.shape[1], 4), 'uint8')
    label_rgb[:, :, 0] = 255 - 255*label
    label_rgb[:, :, 1] = 255 - 255*label
    label_rgb[:, :, 2] = 255 - 255*label
    label_rgb[:, :, 3] = 255*label

    overlay = Image.fromarray(label_rgb, mode='RGBA')

    image = (image-np.min(image))/(np.max(image)-np.min(image))

    background = Image.fromarray(np.uint8(cmap(image)*255))

    background.paste(overlay, (0, 0), overlay)

    plt.figure()
    plt.imshow(background, **kwargs)




def plot_comparison(data_sets, **kwargs):
    """ Plot a couple of images for comparison
    
    Parameters
    ----------
    data_sets : list of np.array
        List of data sets       
    
    Returns
    -------  

    """
    count = len(data_sets)

    fig, axs = plt.subplots(count, 1, figsize=(12, 12))
    
    for n, data in enumerate(data_sets):
        axs[n].imshow(data, **kwargs)
        





def plot_threshold(data, threshold, value, **kwargs):

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    # First plot
    p0 = axs[0].imshow(data)

    # Color bar locator
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cb0 = fig.colorbar(p0, ax=axs[0], cax=cax)
    cb0.ax.plot([-1, 1], [value]*2, 'r')

    # Second plot
    p1 = axs[1].imshow(threshold, **kwargs)

    # Color bar locator
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="3%", pad=0.15)
    cb0 = fig.colorbar(p1, ax=axs[1], cax=cax)







#******************************************************************************
# (3) Network plotting
# A couple of functions to visulize networks
#******************************************************************************

def plot(G, **kwargs):
    """ Plot network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    color : str
        Color of network
    with_labels : bolean
        Whether to plot labels
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            **kwargs)






def plot_components(G, label=True, **kwargs):
    """ Plot network components
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    node_size : float
        Size of network nodes
    label : bolean
        Whether to plot labels
    filename : str
        Save figure with this name
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting    
    node_color, palette = get_node_colors(G, 'component', return_palette=True)
    
    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_color=node_color,
            **kwargs)

    ax=kwargs['ax']
    ax.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if label is True:

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

            ax.text(y_avg, x_avg, label, fontsize=15,
                    color=palette[G.nodes[n]['component']])

    




def plot_faults(G, label=True, **kwargs):
    """ Plot network faults
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    node_size : float
        Size of network nodes
    label : bolean
        Whether to plot labels
    filename : str
        Save figure with this name
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    node_color, palette = get_node_colors(G, 'fault', return_palette=True)

    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_color=node_color,
            **kwargs)

    ax=kwargs['ax']
    if label is True:
        
        labels = metrics.get_fault_labels(G)
        
        for l in labels:
            fault = metrics.get_fault(G, l)
            
            # Calculate centre
            x_avg = 0
            y_avg = 0

            for n in fault:
                y_avg = y_avg + G.nodes[n]['pos'][0]
                x_avg = x_avg + G.nodes[n]['pos'][1]

            N = len(fault.nodes)
            y_avg = y_avg/N
            x_avg = x_avg/N

            # Scale color map
            label = G.nodes[n]['fault']

            ax.text(y_avg, x_avg, label, fontsize=15,
                    color=palette[G.nodes[n]['fault']])

    ax.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True, top=False, labelleft=True, labelbottom=True, labeltop=False)    




def plot_attribute(G, attribute, **kwargs):
    """ Plot network node attribute
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute used for plotting
    ax : plt axis
        Axis
    vmin : float
        Minium value
    vmax : float
        Maximum value
    node_size : float
        Size of network nodes        
    filename : str
        Save figure with this name
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_color=np.array([G.nodes[node][attribute] for node in G]),
            node_size=0.75,
            **kwargs)

    ax = kwargs['ax']
    ax.axis('on')  # turns on axis
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    if 'cmap' in kwargs:
      cmap = kwargs['cmap']
    else:
      cmap = plt.get_cmap('viridis')

    if 'vmin' in kwargs:
      vmin = kwargs['vmin']
    else:
        vmin = metrics.compute_node_values(G, attribute, 'min')

    if 'vmax' in kwargs:
      vmax = kwargs['vmax']
    else:
        vmax = metrics.compute_node_values(G, attribute, 'max')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(attribute, rotation=270)








def plot_edge_attribute(G, attribute, **kwargs):
    """ Plot network edge attribute
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Attribute used for plotting
    ax : plt axis
        Axis      
    filename : str
        Save figure with this name
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    nx.draw(G,
            pos=nx.get_node_attributes(G, 'pos'),
            node_size=0.001,
            **kwargs)

    nx.draw_networkx_edges(G,
                           pos=nx.get_node_attributes(G, 'pos'),
                           edge_color=np.array([G.edges[edge][attribute] for edge in G.edges]),
                           **kwargs)
    ax = kwargs['ax']
    ax.axis('on')

    # Colorbar
    cmap = plt.cm.twilight_shifted
    vmax = metrics.compute_edge_values(G, attribute, 'max')
    vmin = metrics.compute_edge_values(G, attribute, 'min')

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])

    cbar = plt.colorbar(sm, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(attribute, rotation=270)







def cross_plot(G, var0, var1, **kwargs):
    """ Cross-plot two network (node) properties
    
    Parameters
    ----------
    G : nx.graph
        Graph
    var0 : str
        Attribute to plot as x-axis
    var0 : str
        Attribute to plot as y-axis  
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    x = np.zeros(len(G.nodes))
    z = np.zeros(len(G.nodes))

    if var0 == 'x':
        for n, node in enumerate(G):
            x[n] = G.nodes[node]['pos'][1]
            z[n] = G.nodes[node][var1]

    if var0 == 'z':
        for n, node in enumerate(G):
            x[n] = G.nodes[node]['pos'][0]
            z[n] = G.nodes[node][var1]

    if var1 == 'x':
        for n, node in enumerate(G):
            x[n] = G.nodes[node][var0]
            z[n] = G.nodes[node]['pos'][1]

    if var1 == 'z':
        for n, node in enumerate(G):
            x[n] = G.nodes[node][var0]
            z[n] = G.nodes[node]['pos'][0]

    plt.plot(x, z, '.', **kwargs)





def plot_compare_graphs(G, H):
    """ Plot two graphs for comparison
    
    Parameters
    ----------
    G : nx.graph
        Graph
    H : nx.graph
        Graph 
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    fig, ax = plt.subplots(2, 1)
    plot_components(G, ax=ax[0], **kwargs)
    plot_components(H, ax=ax[1], **kwargs)









#******************************************************************************
# (3) Fault evolution plots
# A couple of functions to visualize fault network properties
#******************************************************************************

def plot_rose(G, ax=[]):
    """ Plot rose diagram of fault network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis      
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    strikes = np.zeros(len(G.edges))
    lengths = np.zeros(len(G.edges))

    for n, edge in enumerate(G.edges):
        strikes[n] = G.edges[edge]['strike']
        lengths[n] = G.edges[edge]['length']

    # ROSE PLOT
    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(
        strikes, bin_edges, weights=lengths)
    number_of_strikes[0] += number_of_strikes[-1]
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])

    cmap = plt.cm.twilight_shifted(np.concatenate(
        (np.linspace(0, 1, 18), np.linspace(0, 1, 18)), axis=0))

    if ax == []:
        fig = plt.figure(figsize=(8, 8))

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





def bar_plot(attribute, faults, times, steps=[], ax=[]):
    """ Bar plot of fault network attribute
    
    Parameters
    ----------
    attribute : np.array
        Attribute to plot
    faults : np.array
        Fault labels
    times : np.array
        Times used for plotting
    
    Returns
    -------  
    """
    
    # Plotting
    colors = utils.get_colors()

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





def stack_plot(attribute, faults, times, steps=[], ax=[]):
    """ Stack plot of fault network attribute
    
    Parameters
    ----------
    attribute : np.array
        Attribute to plot
    faults : np.array
        Fault labels
    times : np.array
        Times used for plotting
    
    Returns
    -------  
    """
    
    # Plotting
    colors = utils.get_colors()

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

    ax.stackplot(x, y, fc=colors[:max_fault, :],
                 alpha=0.75, edgecolor='white', linewidth=0.5)






def plot_width(G, ax, width, tips=True, plot=False):
    """ Plot edge width of fault network
    
    Parameters
    ----------
    G : nx.graph
        Graph
    ax : plt axis
        Axis
    width : np.array
        Width of network edges
    tips : bolean
        Plot tips
    plot : False
        Plot helper functions
    
    Returns
    -------  
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph" 
    
    # Plotting
    pos = nx.get_node_attributes(G, 'pos')

    n_comp = 10000

    sns.color_palette(None, 2*n_comp)

    colors = get_node_colors(G, 'fault')

    def get_points(u):

        u0 = np.array(pos[u[0]])
        u1 = np.array(pos[u[1]])

        u_vec = u0-u1

        u_perp = np.array([-u_vec[1], u_vec[0]])
        u_perp = u_perp/np.linalg.norm(u_perp)

        u0a = u0 - u_perp*width[u[0]]
        u0b = u0 + u_perp*width[u[0]]

        u1a = u1 - u_perp*width[u[1]]
        u1b = u1 + u_perp*width[u[1]]

        return u0a, u0b, u1a, u1b

    def get_intersect(a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines passing through a2,a1
        and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])        # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])           # get first line
        l2 = np.cross(h[2], h[3])           # get second line
        x, y, z = np.cross(l1, l2)          # point of intersection
        if z == 0:                          # lines are parallel
            return (float('inf'), float('inf'))
        return np.array([x/z, y/z])

    def clockwiseangle_and_distance(origin, point):
        refvec = [0, 1]
        # Vector between point and the origin: v = p - o
        vector = [point[0]-origin[0], point[1]-origin[1]]
        # Length of vector: ||v||
        lenvector = math.hypot(vector[0], vector[1])
        # If length is zero there is no angle
        if lenvector == 0:
            return -math.pi, 0
        # Normalize vector: v/||v||
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod = normalized[0]*refvec[0] + \
            normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - \
            refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        # Negative angles represent counter-clockwise angles so we need to
        # subtract them
        # from 2*pi (360 degrees)
        if angle < 0:
            return 2*math.pi+angle, lenvector
        # I return first the angle because that's the primary sorting criterium
        # but if two vectors have the same angle then the shorter distance
        # should come first.
        return angle, lenvector

    def get_edges(G, node):
        neighbors = list(G.neighbors(node))
        pts = [G.nodes[neighbor]['pos'] for neighbor in neighbors]
        pts, neighbors = zip(
            *sorted(
                zip(pts, neighbors),
                key=lambda x: clockwiseangle_and_distance(
                    G.nodes[node]['pos'], x[0])
                )
            )
        edges = [(node, neighbor) for neighbor in neighbors]
        return edges

    for node, color in zip(G, colors):
        if tips is True and G.degree(node) == 1:

            edge = get_edges(G, node)[0]

            node0 = np.array(pos[edge[0]])
            node1 = np.array(pos[edge[1]])

            vec = node0-node1
            vec_perp = np.array([-vec[1], vec[0]])
            vec_perp = vec_perp/np.linalg.norm(vec_perp)

            vec_pos = node0 + vec_perp*width[edge[0]]
            vec_neg = node0 - vec_perp*width[edge[0]]

            stack = np.vstack((vec_pos,
                               node0+vec,
                               vec_neg,
                               vec_pos))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 2:

            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[1][1], points[1][3]))
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))

            stack = np.vstack((points[0][3], intersects[1], points[1][2],
                               points[1][3], intersects[0], points[0][2]))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 3:

            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[2][1], points[2][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[0][2]))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 4:
            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[2][1], points[2][3], points[3][0], points[3][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[3][1], points[3][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[3][2],
                               points[3][3], intersects[3], points[0][2]))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

        if G.degree(node) == 5:
            edges = get_edges(G, node)

            points = []
            for edge in edges:
                points.append(get_points(edge))

            intersects = []
            intersects.append(get_intersect(
                points[0][1], points[0][3], points[1][0], points[1][2]))
            intersects.append(get_intersect(
                points[1][1], points[1][3], points[2][0], points[2][2]))
            intersects.append(get_intersect(
                points[2][1], points[2][3], points[3][0], points[3][2]))
            intersects.append(get_intersect(
                points[3][1], points[3][3], points[4][0], points[4][2]))
            intersects.append(get_intersect(
                points[0][0], points[0][2], points[4][1], points[4][3]))

            stack = np.vstack((points[0][3], intersects[0], points[1][2],
                               points[1][3], intersects[1], points[2][2],
                               points[2][3], intersects[2], points[3][2],
                               points[3][3], intersects[3], points[4][2],
                               points[4][3], intersects[4], points[0][2]))

            polygon = Polygon(stack, True, facecolor=color, alpha=1)
            p = PatchCollection([polygon], match_original=True)
            ax.add_collection(p)

    ax.axis('equal')
    plt.show()







#******************************************************************************
# (4) Helper plots
# A couple of functions to visualize a few other properties
#******************************************************************************



def plot_matrix(matrix, rows, columns, threshold):
    """ Plot similarity matrix
    
    Parameters
    ----------
    matrix : np.array
        Matrix to plot
    rows : np.array
        Rows
    columns : np.array
        Columns
    threshold : float
        Threshold
        
    Returns
    -------  
    """
    
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(matrix, 'Blues_r')

    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticklabels(columns)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(rows)

    ax.set_xlim(-0.5, matrix.shape[1]-0.5)
    ax.set_ylim(-0.5, matrix.shape[0]-0.5)

    # Loop over data dimensions and create text annotations.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] < threshold:
                ax.text(
                    j, i, round(matrix[i, j], 3),
                    ha="center", va="center", color="r"
                )
            else:
                ax.text(
                    j, i, round(matrix[i, j], 3),
                    ha="center", va="center", color="k"
                )





def plot_connections(matrix, rows, columns):
    """ Plot connections
    
    Parameters
    ----------
    matrix : np.array
        Matrix to plot
    rows : np.array
        Rows
    columns : np.array
        Columns
        
    Returns
    -------  
    """
    
    # Plotting
    for n in range(100):
        threshold = n/100
        connections = edits.similarity_to_connection(
            matrix, rows, columns, threshold)
        plt.scatter(threshold, len(connections), c='red')
        plt.xlabel('Threshold')
        plt.ylabel('Number of connections')
        
        
        


