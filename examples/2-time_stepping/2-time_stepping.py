import pickle
import matplotlib.pyplot as plt
plt.close("all")

# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from edits import *
from metrics import *
from plots import *
from utils import *



G = pickle.load(open("graph.p", 'rb'))

G = simplify(G, 5)

G = label_components(G)

G = compute_strikes(G)

G = compute_edge_length(G)

plot_components(G)

plot_rose(G)
















