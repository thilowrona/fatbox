#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:44:29 2020

@author: wrona
"""

import numpy as np
import matplotlib.pyplot as plt
plt.close('all')

# for n in range(23):

# n = 5   

# name = str(n).zfill(2)     



times = [0,
         200000,
         400000,
         600000,
         800000,
         1000000,
         1200000,
         1403870,
         1604490,
         1803930,
         2002420,
         2204740,
         2404690,
         2604690,
         2804690,
         3004690,
         3204690,
         3403430,
         3600940,
         3801860,
         4002760,
         4200040,
         4401070,
         4601360,
         4802030,
         5002980,
         5202070,
         5402320,
         5602120,
         5800180,
         6001400,
         6200680]



# for time in times:

time = times[10]

name = str(time).zfill(7)





data0 = np.genfromtxt('/home/wrona/fault_analysis/examples/15-2D_particles/csv/particles/' + name + '.csv',
                     delimiter=',',
                     names=True,
                     dtype=None)


x0 = np.zeros((49997))
y0 = np.zeros((49997))
v_x0 = np.zeros((49997))
v_y0 = np.zeros((49997))

for n in range(49997):
    
    x0[n] = data0[n][0]
    y0[n] = data0[n][1]
    
    v_x0[n] = data0[n][6]
    v_y0[n] = data0[n][7]






data1 = np.genfromtxt('/home/wrona/fault_analysis/examples/15-2D_particles/csv/particles_from_field/' + name + '.csv',
                     delimiter=',',
                     names=True,
                     dtype=None)


x1 = np.zeros((49997))
y1 = np.zeros((49997))
v_x1 = np.zeros((49997))
v_y1 = np.zeros((49997))

for n in range(49997):
    
    x1[n] = data1[n][4]
    y1[n] = data1[n][5]
    
    v_x1[n] = data1[n][0]
    v_y1[n] = data1[n][1]







from mpl_toolkits.axes_grid1 import make_axes_locatable

s = 1
fig, axs = plt.subplots(3, 1, sharey=True, figsize=(10,12))

axs[0].set_title('Particle v_y')
im0 = axs[0].scatter(x0, y0, c=v_y0, s=s)

divider = make_axes_locatable(axs[0])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im0, cax=cax, orientation='vertical')
cbar.set_label('v_y')



axs[1].set_title('Field v_y')
im1 = axs[1].scatter(x1, y1, c=v_y1, s=1)

divider = make_axes_locatable(axs[1])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im1, cax=cax, orientation='vertical')
cbar.set_label('v_x')




axs[2].set_title('Difference')
im2 = axs[2].scatter(x0, y0, c=v_y0-v_y1, s=1)

divider = make_axes_locatable(axs[2])
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = fig.colorbar(im2, cax=cax, orientation='vertical')
cbar.set_label('dv_y')


plt.savefig('./images/v_y_comparison.png', dpi=300)