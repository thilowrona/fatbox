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








data = np.genfromtxt('/home/wrona/fault_analysis/examples/15-2D_particles/csv/particles/' + name + '.csv',
                     delimiter=',',
                     names=True,
                     dtype=None)


x = np.zeros((49997))
y = np.zeros((49997))
v_x = np.zeros((49997))
v_y = np.zeros((49997))

for n in range(49997):
    
    x[n] = data[n][0]
    y[n] = data[n][1]
    
    v_x[n] = data[n][5]
    v_y[n] = data[n][6]



plt.scatter(x,y,c=v_x)
