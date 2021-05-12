from itertools import count

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import vtk


def get_attribute(G, name):
    highest_node = max(list(G.nodes))+1
    attribute = [0]*(highest_node)
    for node in G:
        if name in G.nodes[node]:
            attribute[node] = G.nodes[node][name]
        else:
            attribute[node] = 1e6
    return attribute


def get_times(filename):
    """ Get times from statistics file """
    # EXTRACT VARIABLE NAMES #
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

    # EXTRACT TIMES
    times = []
    for i in range(df.shape[0]):
        if pd.notnull(df.iloc[i, 27]):
            times.append(df.iloc[i, 1])

    return np.array(times)


def get_colors():
    n_comp = 1000
    palette = sns.color_palette(None, 2*n_comp)
    node_color = np.ones((2*n_comp, 4))
    node_color[:, :3] = np.matrix(palette)
    return node_color


def get_labels(G, attribute):
    """ Get labels for plotting node attributes """
    # get unique groups
    groups = set(nx.get_node_attributes(G, attribute).values())
    mapping = dict(zip(sorted(groups), count()))
    nodes = G.nodes()
    colors = [mapping[G.nodes[node][attribute]] for node in nodes]

    return np.array(colors)


def get_edge_labels(G, attribute):
    """ Get labels for plotting edge attributes """
    # get unique groups
    groups = set(nx.get_edge_attributes(G, attribute).values())
    mapping = dict(zip(sorted(groups), count()))
    colors = [mapping[G.edges[e][attribute]] for e in G.edges()]

    return np.array(colors)


def merge(G, H):
    N = len(G.nodes)

    for node in H.nodes:
        G.add_node(node+N)
        G.nodes[node+N]['pos'] = H.nodes[node]['pos']
        G.nodes[node+N]['x'] = H.nodes[node]['x']
        G.nodes[node+N]['y'] = H.nodes[node]['y']
        G.nodes[node+N]['z'] = H.nodes[node]['z']
        G.nodes[node+N]['edges'] = H.nodes[node]['edges']
        G.nodes[node+N]['component'] = H.nodes[node]['component']

    for edge in H.edges:
        G.add_edge(edge[0]+N, edge[1]+N)

    return G


def topo_line(data):
    col = data.shape[1]
    line = np.zeros(col)
    for n in range(col):
        m = 0
        while data[m, n] == 0:
            m = m + 1
        line[n] = m
    return line.astype(int)


def bottom_line(data, threshold):
    col = data.shape[1]
    line = np.zeros(col)
    for n in range(col):
        m = data.shape[0]-1
        while data[m, n] < threshold:
            m = m - 1
        line[n] = m
    return line.astype(int)


def writeObjects(G, attributes, power=1, nodeLabel=[], fileout='test'):
    """
    Store points and/or graphs as vtkPolyData or vtkUnstructuredGrid.
    Required argument:
    - nodeCoords is a list of node coordinates in the format [x,y,z]
    Optional arguments:
    - G is Networkx graph
    - name/name2 is the scalar's name
    - power/power2 = 1 for r~scalars, 0.333 for V~scalars
    - nodeLabel is a list of node labels
    - method = 'vtkPolyData' or 'vtkUnstructuredGrid'
    - fileout is the output file name (will be given .vtp or .vtu extension)
    """

    highest_node = max(list(G.nodes))+1

    coordinates = np.zeros((highest_node, 3))

    for n, node in enumerate(G):
        coordinates[node, 0] = G.nodes[node]['x']
        coordinates[node, 1] = G.nodes[node]['y']
        coordinates[node, 2] = G.nodes[node]['z']

    def generate_points2():
        for row in coordinates:
            if row[0] == 0 and row[1] == 0 and row[2] == 0:
                yield list([float("nan"), float("nan"), float("nan")])
            else:
                yield list(row)

    nodeCoords = generate_points2()

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

    for a in attributes:
        scalar = get_attribute(G, a)

        attribute = vtk.vtkFloatArray()
        attribute.SetNumberOfComponents(1)
        attribute.SetName(a)
        attribute.SetNumberOfTuples(len(scalar))
        # i becomes 0, 1, 2,..., and j runs through scalars
        for i, j in enumerate(scalar):
            attribute.SetValue(i, j**power)

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


def func_linear(x, a, b):
    return a*x + b


def func_powerlaw(x, a, b):
    return a*x**(-b)


def func_exponential(x, a, b):
    return a*np.exp(-b*x)


def metrics(x, y, y_pred):
    # Residuals
    residuals = y - y_pred
    # Residual sum of squares
    ss_res = np.sum(residuals**2)
    # Residual total sum of squares
    ss_tot = np.sum((y-np.mean(y))**2)
    # R^2
    R2 = 1 - (ss_res / ss_tot)

    return R2
