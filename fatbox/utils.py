# Packages
import vtk
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from itertools import count


#==============================================================================
# This file contains a series of utility functions
#==============================================================================

def get_times(filename):
    """ Get times from statistics file
    
    Parameters
    ----------
    filename : str
        Path to statistics file (from ASPECT)
    fx : float
    fy : float
    
    Returns
    -------  
    array : array
        Times
    """
    
    # Extract variables
    # read entire file and store each line
    with open(filename) as f:
        header = f.readlines()
    # remove all lines that do not start with "#" (starting from the back)
    # nonheaderID = [x[0] != '#' for x in header]
    for index, linecontent in reversed(list(enumerate(header))):
        if linecontent[0] != '#':
            del header[index]
    # remove whitespace characters like `\n` at the end of each line
    header = [x.strip() for x in header]

    # EXTRACT DATA
    df = pd.read_csv(filename, comment='#', header=None, delim_whitespace=True)

    # Find index of column containing output files
    for col in range(df.shape[1]):
        if df[col][0] == 'output/solution/solution-00000':
            index = col

    # EXTRACT TIMES
    times = []    
    for n in range(df.shape[0]):           
        if pd.notnull(df.iloc[n,index]):
            times.append(df.iloc[n,1])

    return np.array(times)





def get_colors():
    """ Get times from statistics file
    
    Parameters
    ----------
    filename : str
        Path to statistics file (from ASPECT)
    fx : float
    fy : float
    
    Returns
    -------  
    array : array
        Times
    """
    
    n_comp = 1000
    palette = sns.color_palette(None, 2*n_comp)
    node_color = np.ones((2*n_comp, 4))
    node_color[:, :3] = np.matrix(palette)
    return node_color





def get_labels(G, attribute):
    """ Get labels for plotting node attributes
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attribute : str
        Node attribute
    
    Returns
    -------  
    array : array
        Colors
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"   

    # Get unique groups
    groups = set(nx.get_node_attributes(G, attribute).values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = G.nodes()
    colors = [mapping[G.nodes[node][attribute]] for node in nodes]

    return np.array(colors)





def get_edge_labels(G, attribute):
    """ Get labels for plotting edge attributes
    
    Parameters
    ----------
    G : nx.graph
        Graph
        
    attribute : str
        Edge attribute    
        
    Returns
    -------  
    array : np.array
        Colors
    """
    
    # Assertions
    assert isinstance(G, nx.Graph), "G is not a NetworkX graph"   

    # get unique groups
    groups = set(nx.get_edge_attributes(G, attribute).values())
    mapping = dict(zip(sorted(groups), count()))
    colors = [mapping[G.edges[e][attribute]] for e in G.edges()]

    return np.array(colors)





def topo_line(data):
    """ Get topography from 2-D ASPECT model
    
    Parameters
    ----------
    array : np.array
        Model outpout
                
    Returns
    -------  
    line : np.array
        Model outpout
    """
    
    col = data.shape[1]
    line = np.zeros(col)
    for n in range(col):
        m = 0
        while data[m, n] == 0:
            m = m + 1
        line[n] = m
    return line.astype(int)





def bottom_line(data, threshold):
    """ Get bottom line from 2-D ASPECT model
    
    Parameters
    ----------
    array : np.array
        Model outpout
                
    Returns
    -------  
    line : np.array
        Model outpout
    """

    col = data.shape[1]
    line = np.zeros(col)
    for n in range(col):
        m = data.shape[0]-1
        while data[m, n] < threshold:
            m = m - 1
        line[n] = m
    return line.astype(int)







def save_network_for_paraview(G, attributes, nodeLabel=[], fileout='test'):
    """ Save a network in a format readable by ParaView (i.e. vtp - vtkPolyData)
    
    Parameters
    ----------
    G : nx.graph
        Graph
    attributes : list
        List attributes to write in vtp file
    nodeLabel : list
        List of node labels
    fileout : str
        Name of output file
                
    Returns
    -------  

    """    
    

    highest_node = max(list(G.nodes))+1

    coordinates = np.zeros((highest_node, 3))

    for n, node in enumerate(G):
        coordinates[node, 0] = G.nodes[node]['x']
        coordinates[node, 1] = G.nodes[node]['y']
        coordinates[node, 2] = G.nodes[node]['z']

    def generate_points():
        for row in coordinates:
            if row[0] == 0 and row[1] == 0 and row[2] == 0:
                yield list([float("nan"), float("nan"), float("nan")])
            else:
                yield list(row)

    nodeCoords = generate_points()

    points = vtk.vtkPoints()
    for node in nodeCoords:
        points.InsertNextPoint(node)

    edges = G.edges

    if edges:
        line = vtk.vtkCellArray()
        line.Allocate(len(edges))
        for edge in edges:
            line.InsertNextCell(2)
            line.InsertCellPoint(edge[0])
            # line from point edge[0] to point edge[1]
            line.InsertCellPoint(edge[1])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    if edges:
        polydata.SetLines(line)



    def get_attribute(G, name):
        highest_node = max(list(G.nodes))+1
        attribute = [0]*(highest_node)
        for node in G:
            if name in G.nodes[node]:
                attribute[node] = G.nodes[node][name]
            else:
                attribute[node] = 1e6
        return attribute




    for a in attributes:
        scalar = get_attribute(G, a)

        attribute = vtk.vtkFloatArray()
        attribute.SetNumberOfComponents(1)
        attribute.SetName(a)
        attribute.SetNumberOfTuples(len(scalar))
        # i becomes 0, 1, 2,..., and j runs through scalars
        for i, j in enumerate(scalar):
            attribute.SetValue(i, j)

        polydata.GetPointData().AddArray(attribute)

    # if scalar2:
    #     attribute2 = vtk.vtkFloatArray()
    #     attribute2.SetNumberOfComponents(1)
    #     attribute2.SetName(name2)
    #     attribute2.SetNumberOfTuples(len(scalar2))
    #     for i, j in enumerate(scalar2):   # i becomes 0, 1, 2,..., and j
    #     runs through scalar2
    #         attribute2.SetValue(i,j**power2)

    # if scalar3:
    #     attribute3 = vtk.vtkFloatArray()
    #     attribute3.SetNumberOfComponents(1)
    #     attribute3.SetName(name3)
    #     attribute3.SetNumberOfTuples(len(scalar3))
    #     for i, j in enumerate(scalar3):   # i becomes 0, 1, 2,..., and j runs
    #     through scalar2
    #         attribute3.SetValue(i,j**power3)

    # if scalar:
    #     polydata.GetPointData().AddArray(attribute)
    # if scalar2:
    #     polydata.GetPointData().AddArray(attribute2)
    # if scalar3:
    #     polydata.GetPointData().AddArray(attribute3)
    if nodeLabel:
        polydata.GetPointData().AddArray(label)  # label is undefined
        
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fileout+'.vtp')
    writer.SetInputData(polydata)
    writer.Write()







