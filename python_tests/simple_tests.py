#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 07:43:04 2023

@author: matthias
"""
import numpy as np
import random
import matplotlib
import json
from matplotlib import pyplot as plt
import sys
import time

time1 = time.time()
delta_t = time.time() - time1
nb = 100000000
large_array = np.random.uniform(low=0, high=1, size=(nb, 2))
while delta_t < 60:
    large_array += 0.000001
    delta_t = time.time() - time1
sys.exit()
def find_nearest_smaller_idx(array, value):
    array = np.asarray(array)
    nearest_smaller_value = array[array < value].max()
    idx = (np.abs(array - nearest_smaller_value)).argmin()
    return idx

nb = 10000
F = 100

iat = 1.0/F
# Initialize arrival time (at) vector
at = np.linspace(0,(nb-1)*iat,nb)
mean_bubble_velocities = np.zeros((nb,3))*np.nan

CV_w = 2
# Get the random probe displacement at control volume edge
np.random.seed(42)
random.seed(42)
X_rand = np.transpose(np.array([np.random.uniform(-CV_w/2.0,CV_w/2.0,nb),
                                np.random.uniform(-CV_w/2.0,CV_w/2.0,nb)]))
f1 = plt.figure(figsize=(3,3))        
a1 = f1.gca()
a1.scatter(X_rand[:,0],X_rand[:,1])
a1.set_xlabel('$x$ [m]')
a1.set_ylabel('$y$ [m]')
f1.tight_layout()
f1.savefig(f"random_distribution.pdf")
plt.close()
a = at[10::]-at[10]
x = at[10::][(at[10::]-at[10])<0.1]

string = f'{10:08d}'
rand = np.zeros(nb)
for ii in range(0,nb):
    rand[ii] = random.uniform(-CV_w/2.0,CV_w/2.0)
    
# mean_ln_diameter = -7.6718092214289895
# sd_ln_diameter = 1.8253076065331872
mean_ln_diameter = -8.889784195117713
sd_ln_diameter = 2.066280927671439
D = np.random.lognormal(mean_ln_diameter,
                        sd_ln_diameter,
                        size=nb)

# Histogram of chord times
fig, ax = plt.subplots(figsize=(3,3))
(n_c, bins_c, patches_c) = ax.hist(D, \
    bins=50, density=True, color='r', alpha=0.7, label='sim.',\
    range=(0,0.002))
# chord_times_meas = chord_times_meas_all[z][:,1]
# (n_c, bins_c, patches_c) = ax.hist(chord_times_meas, \
#     bins=50, density=True, color='g', alpha=0.7, label='meas.',\
#     range=(0,0.002))
ax.set_ylabel('Frequency [-]')
ax.legend(fontsize=10)
ax.set_xlabel(f'Diameter [m]')
ax.set_xlim([0,0.002])
# if jj == 1:
#     axs[jj].set_yscale('log')
# ax.set_xscale('log')
fig.tight_layout()
fig.savefig(f'histrograms_D.png',dpi=300)

import pandas as pd
def SBG_get_signal_traj(
    t_p,
    x_p,
    t_probe):

    t_probe_unique = np.unique(np.hstack((t_p, t_probe[t_probe<=max(t_p)])))
    t_p_del = t_p[np.invert(np.isin(t_p, t_probe))]
    x_interp = pd.DataFrame(x_p, index=t_p, columns=['x','y','z'])
    x_interp = x_interp.reindex(t_probe_unique).interpolate(method='index')
    x_interp = x_interp.drop(index=t_p_del)
    return x_interp.index, x_interp.values

x_p = np.array([np.linspace(0,20,9),np.linspace(0,20,9),np.linspace(0,20,9)]).transpose()
t_p = np.linspace(0,10,9)
t_probe = np.linspace(0,10,101)
t_p_resampled, x_p_resampled = SBG_get_signal_traj(
    t_p,
    x_p,
    t_probe)
