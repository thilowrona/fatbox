import math
import numpy as np
import networkx as nx
from itertools import count
import vtk









# Get labels for plotting node attributes
def get_labels(G, attribute):

    # get unique groups
    groups = set(nx.get_node_attributes(G,attribute).values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = G.nodes()
    colors = [mapping[G.nodes[node][attribute]] for node in nodes]
    
    return np.array(colors)



# Get labels for plotting edge attributes
def get_edge_labels(G, attribute):

    # get unique groups
    groups = set(nx.get_edge_attributes(G,attribute).values())
    mapping = dict(zip(sorted(groups),count()))
    colors = [mapping[G.edges[e][attribute]] for e in G.edges()]
    
    return np.array(colors)




def merge(G,H):
    N = len(G.nodes) 
    
    for node in H.nodes:
        G.add_node(node+N)
        G.nodes[node+N]['pos'] = H.nodes[node]['pos']
        G.nodes[node+N]['x']   = H.nodes[node]['x']
        G.nodes[node+N]['y']   = H.nodes[node]['y']
        G.nodes[node+N]['z']   = H.nodes[node]['z']
        G.nodes[node+N]['edges']   = H.nodes[node]['edges']
        G.nodes[node+N]['component']   = H.nodes[node]['component']
        
    for edge in H.edges:
        G.add_edge(edge[0]+N, edge[1]+N)
        
    return G



def topo_line(data):
    col = data.shape[1]
    line =  np.zeros(col)
    for n in range(col):
        m = 0
        while data[m,n] == 0:
            m = m + 1        
        line[n] = m
    return line.astype(int)







def bottom_line(data, threshold):
    col = data.shape[1]
    line =  np.zeros(col)
    for n in range(col):
        m = data.shape[0]-1
        while data[m,n] < threshold:  
            m = m - 1 
        line[n] = m
    return line.astype(int)





def writeObjects(G,
                 scalar = [], name = '', power = 1,
                 scalar2 = [], name2 = '', power2 = 1,
                 scalar3 = [], name3 = '', power3 = 1,
                 nodeLabel = [],
                 fileout = 'test'):
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

    coordinates = np.zeros((highest_node,3))
    
    for n, node in enumerate(G):
        coordinates[node,0]  = G.nodes[node]['x']
        coordinates[node,1]  = G.nodes[node]['y']
        coordinates[node,2]  = G.nodes[node]['z']
                       
            
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
            line.InsertCellPoint(edge[1])   # line from point edge[0] to point edge[1]

    if scalar:
        attribute = vtk.vtkFloatArray()
        attribute.SetNumberOfComponents(1)
        attribute.SetName(name)
        attribute.SetNumberOfTuples(len(scalar))
        for i, j in enumerate(scalar):   # i becomes 0,1,2,..., and j runs through scalars
            attribute.SetValue(i,j**power)

    if scalar2:
        attribute2 = vtk.vtkFloatArray()
        attribute2.SetNumberOfComponents(1)
        attribute2.SetName(name2)
        attribute2.SetNumberOfTuples(len(scalar2))
        for i, j in enumerate(scalar2):   # i becomes 0,1,2,..., and j runs through scalar2
            attribute2.SetValue(i,j**power2)

    if scalar3:
        attribute3 = vtk.vtkFloatArray()
        attribute3.SetNumberOfComponents(1)
        attribute3.SetName(name3)
        attribute3.SetNumberOfTuples(len(scalar3))
        for i, j in enumerate(scalar3):   # i becomes 0,1,2,..., and j runs through scalar2
            attribute3.SetValue(i,j**power3)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    if edges:
        polydata.SetLines(line)
    if scalar:
        polydata.GetPointData().AddArray(attribute)
    if scalar2:
        polydata.GetPointData().AddArray(attribute2)
    if scalar3:
        polydata.GetPointData().AddArray(attribute3)        
    if nodeLabel:
        polydata.GetPointData().AddArray(label)
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