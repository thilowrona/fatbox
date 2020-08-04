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




times = range(50000, 11200000, 50000)

for n, time in enumerate(times):
        
    # time = 11200000
    
    print(time)
    
    name = str(time).zfill(8)
    
    G = pickle.load(open('./graphs/correlated/' + name + '.p', 'rb'))
            
    
    
    # data = genfromtxt('./csv/' + name + '.csv', delimiter=',')    
    # data = np.flip(data, axis=0)
    
    # plastic_strain = data[:-1,0].reshape(1600, 5000)
    # strain_rate = data[:-1,1].reshape(1600, 5000)
    
    # G = extract_attribute(G, plastic_strain, 'plastic_strain')
    # G = extract_attribute(G, strain_rate, 'strain_rate')
        
    

    
    components = return_components(G)
    
    G = compute_edge_length(G)
    lengths = fault_lengths(G) 
    
    max_strain = max_value_components(G, 'plastic_strain')
    max_strain_rate = max_value_components(G, 'strain_rate') 
    
    
    if n == 0:        
        faults    = np.empty((100,len(times),7))
        faults[:] = np.NaN
        
        
        
    faults[:len(components),n,0] = np.asarray(components)
    faults[:len(components),n,1] = np.asarray(lengths)
    faults[:len(components),n,2] = np.asarray(max_strain)       
    faults[:len(components),n,3] = np.asarray(max_strain_rate) 
    faults[:len(components),n,4] = np.asarray(sorted(lengths))
    faults[:len(components),n,5] = np.asarray(sorted(max_strain))
    faults[:len(components),n,6] = np.asarray(sorted(max_strain_rate))




    

#%%
plt.close('all')

fig, axs = plt.subplots(4, 1, sharey=True, figsize=(8,16)) 




      
im0 = axs[0].matshow(np.flipud(faults[:,:,0]))
axs[0].set_title('Faults')
axs[0].set_xlabel('Time [Myrs]')
axs[0].set_ylabel('Number of faults')

axs[0].set_xticklabels([0, 0.224, 0.448, 0.672, 0.896, 1.112])
axs[0].set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])

axs[0].xaxis.tick_bottom()


divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im0, cax=cax, orientation='vertical')
cbar.set_label('Fault ID')



    
im1 = axs[1].matshow(np.flipud(faults[:,:,6]))
# axs[1].set_title('Max strain rate')
axs[1].set_xlabel('Time [Myrs]')
axs[1].set_ylabel('Number of faults')

axs[1].set_xticklabels([0, 0.224, 0.448, 0.672, 0.896, 1.112])
axs[1].set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])

axs[1].xaxis.tick_bottom()


divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label('Max strain rate on faults')





im2 = axs[2].matshow(np.flipud(faults[:,:,5]))
# axs[2].set_title('Max strain')
axs[2].set_xlabel('Time [Myrs]')
axs[2].set_ylabel('Number of faults')

axs[2].set_xticklabels([0, 0.224, 0.448, 0.672, 0.896, 1.112])
axs[2].set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])

axs[2].xaxis.tick_bottom()


divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
cbar.set_label('Max strain on faults')




im3 = axs[3].matshow(np.flipud(faults[:,:,4]))
# axs[3].set_title('Fault length')
axs[3].set_xlabel('Time [Myrs]')
axs[3].set_ylabel('Number of faults')

axs[3].set_xticklabels([0, 0.224, 0.448, 0.672, 0.896, 1.112])
axs[3].set_yticklabels([80, 70, 60, 50, 40, 30, 20, 10, 0])

axs[3].xaxis.tick_bottom()


divider = make_axes_locatable(axs[3])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im3, cax=cax, orientation='vertical')
cbar.set_label('Fault length [km]')

plt.subplots_adjust(wspace=0, hspace=0.5)
plt.show()
plt.savefig('./images/fault_system_evolution.png', dpi=300)




#%%
plt.close('all')
