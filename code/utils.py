import math
import numpy as np
import networkx as nx
from itertools import count









# Get labels for plotting node attributes
def get_labels(G, attribute):

    # get unique groups
    groups = set(nx.get_node_attributes(G,attribute).values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = G.nodes()
    colors = [mapping[G.nodes[n][attribute]] for n in nodes]
    
    return np.array(colors)






