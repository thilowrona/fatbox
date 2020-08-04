import pickle
import numpy as np
from numpy import genfromtxt

import matplotlib.pyplot as plt
plt.close("all")

from scipy.optimize import curve_fit


# Import module
import sys
sys.path.append('/home/wrona/fault_analysis/code/')

from image_processing import *
from edits import *
from metrics import *
from plots import *
from utils import *



# %% LOAD DATA

a_pl  = np.zeros(224)
b_pl  = np.zeros(224)
R2_pl = np.zeros(224)

a_e  = np.zeros(224)
b_e  = np.zeros(224)
R2_e = np.zeros(224)

for n, time in enumerate(range(11200000, 0, -50000)):
        
    # time = 11200000
    
    print(time)
    
    name = str(time).zfill(8)
    
    G = pickle.load(open('./graphs/extended/' + name + '.p', 'rb'))
        
    
    

# %% ANALYZE FAULT LENGTHS

    n_comp = number_of_components(G)
    
    lengths = fault_lengths(G)
    components = np.arange(n_comp)
    
    
    # Calculate histogram
    hist, bin_edges = np.histogram(lengths, density=False)
    
    width = bin_edges[1]-bin_edges[0]
    
    
    
    
    
    popt, pcov = curve_fit(func_powerlaw, bin_edges[:-1], hist, maxfev=2000)
    
    a_pl[n] = popt[0]
    b_pl[n] = popt[1]
    
    R2_pl[n]  = metrics(bin_edges[:-1], hist, func_powerlaw(bin_edges[:-1], a_pl[n] , b_pl[n] ))
    
    
    
    popt, pcov = curve_fit(func_exponential, bin_edges[:-1], hist, maxfev=2000)
    
    a_e[n] = popt[0]
    b_e[n] = popt[1]
    
    R2_e[n]  = metrics(bin_edges[:-1], hist, func_exponential(bin_edges[:-1], a_e[n] , b_e[n] ))
    
    
    
    # plt.figure()
    # plt.scatter(bin_edges_log[:-1], hist_log)
    # plt.plot(bin_edges_log[:-1], func_linear(bin_edges_log[:-1], a=popt[0], b=popt[1]))
    # plt.xlabel('log(L)')
    # plt.ylabel('log(N(L))')
    # plt.show()
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,15))
    
    
    plot_components(G, axes[0,0])
    axes[0,0].set_xlabel('X-coordinate [km]')
    axes[0,0].set_ylabel('Y-coordinate [km]')
    
    axes[1,0].scatter(components, lengths, color='grey')
    axes[1,0].set_xlabel('Fault')
    axes[1,0].set_ylabel('Fault length [km]')
    
    
    axes[0,1].scatter(bin_edges[:-1], hist, color='grey')
    axes[0,1].plot(bin_edges[:-1], func_powerlaw(bin_edges[:-1], a_pl[n], b_pl[n]), color='red')
    axes[0,1].plot(bin_edges[:-1], func_exponential(bin_edges[:-1], a_e[n], b_e[n]), color='blue')
    axes[0,1].legend(['y=' + str(round(a_pl[n] ,2)) + ' * x^(' + str(-round(b_pl[n] ,2)) + '), R^2= ' + str(round(R2_pl[n] ,2)),
                'y=' + str(round(a_e[n] ,2)) + ' * exp(' + str(-round(b_e[n] ,2)) + '*x), R^2= ' + str(round(R2_e[n] ,2)),
                'Data'])
    axes[0,1].set_xlabel('Fault length [km]')
    axes[0,1].set_ylabel('Number of faults')
    
    
    axes[1,1].scatter(bin_edges[:-1], hist, color='grey')
    axes[1,1].plot(bin_edges[:-1], func_powerlaw(bin_edges[:-1], a_pl[n], b_pl[n]), color='red')
    axes[1,1].plot(bin_edges[:-1], func_exponential(bin_edges[:-1], a_e[n], b_e[n]), color='blue')
    axes[1,1].set_yscale('log')
    axes[1,1].set_xscale('log')
    
    
    axes[1,1].legend(['y=' + str(round(a_pl[n],2)) + ' * x^(' + str(-round(b_pl[n],2)) + '), R^2= ' + str(round(R2_pl[n],2)),
                'y=' + str(round(a_e[n],2)) + ' * exp(' + str(-round(b_e[n],2)) + '*x), R^2= ' + str(round(R2_e[n],2)),
                'Data'])
    axes[1,1].set_xlabel('log(Fault length)')
    axes[1,1].set_ylabel('log(Number of fautls)')
    
    
    plt.savefig('./images/lengths/' + str(name) + '.png', dpi=300)
    
    plt.close('all')




fig, ax = plt.subplots(1, 1, figsize=(12,4))
ax.plot(time, b_pl)
