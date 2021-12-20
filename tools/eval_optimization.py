#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 11:48:54 2021

@author: matthias
"""

import json
import pathlib
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

LARGE_NUMBER=10.0**12
LARGE_NEG_NUMBER=-10.0**12
SMALL_NUMBER=10.0**-12

def inverse_den(x):
    """
    Calculate inverse of a number.
    """
    if abs(x) < SMALL_NUMBER:
        return 0.0
    else:
        return 1.0 / x

path = pathlib.Path().cwd() / 'optimization/probe'
path_out = pathlib.Path().cwd() / 'optimization/probe/results'

geometries = ['geom1','geom2','geom3','geom4','geom5','geom6','geom7','geom8','geom9','geom10','geom11','geom12','geom13','geom1_roc','geom3_roc','geom9_roc','geom10_roc','geom13_2','geom13_4','geom13_8','geom13_12','geom13_16','geom13_20']
labels = {'geom1':'Probe1','geom2':'Probe2','geom3':'Probe3','geom4':'Probe4',\
          'geom5':'Probe5','geom6':'Probe6','geom7':'Probe7','geom8':'Probe8',\
          'geom9':'Probe9','geom10':'Probe10','geom11':'Probe11','geom12':'Probe12','geom13':'Probe13','geom1_roc':'Probe1 (roc)','geom3_roc':'Probe3 (roc)','geom9_roc':'Probe9 (roc)','geom10_roc':'Probe10 (roc)','geom13_2':'Probe13 (n=2)','geom13_4':'Probe13 (n=4)','geom13_8':'Probe13 (n=8)','geom13_12':'Probe13 (n=12)','geom13_16':'Probe13 (n=16)','geom13_20':'Probe13 (n=20)'}
runs = ['001','002','003','004','005','006','007','009','010','011']

errors = {}
ux = {}
overview = {}
for geom in geometries:
    for run in runs:
        error_tmp = {}
        p = path / geom / run / 'run'
        if p.is_dir():
            u = pd.read_csv(p / 'error_summary_velocity.csv', index_col=0)
            d = pd.read_csv(p / 'error_summary_bubble_size.csv', index_col=0)
            # error compared to disp. phase times series
            um = pd.read_csv(p / 'error_mean_velocity.csv', index_col=0)
            rs = pd.read_csv(p / 'error_reynolds_stresses.csv', index_col=0)
            ti = pd.read_csv(p / 'error_turbulent_intensity.csv', index_col=0)
            # error compared to bubble times series
            um_b = pd.read_csv(p / 'error_mean_velocity_bubble.csv', index_col=0)
            rs_b = pd.read_csv(p / 'error_reynolds_stresses_bubble.csv', index_col=0)
            ti_b = pd.read_csv(p / 'error_turbulent_intensity_bubble.csv', index_col=0)
            # relative error compared to  disp. phase times series
            um_rel = pd.read_csv(p / 'rel_error_mean_velocity.csv', index_col=0)
            rs_rel = pd.read_csv(p / 'rel_error_reynolds_stresses.csv', index_col=0)
            ti_rel = pd.read_csv(p / 'rel_error_turbulent_intensity.csv', index_col=0)
            # relative error compared to bubble times series
            um_rel_b = pd.read_csv(p / 'rel_error_mean_velocity_bubble.csv', index_col=0)
            rs_rel_b = pd.read_csv(p / 'rel_error_reynolds_stresses_bubble.csv', index_col=0)
            ti_rel_b = pd.read_csv(p / 'rel_error_turbulent_intensity_bubble.csv', index_col=0)

            error_tmp['u'] = u
            error_tmp['d'] = d
            error_tmp['um'] = um
            error_tmp['rs'] = rs
            error_tmp['ti'] = ti
            error_tmp['um_b'] = um_b
            error_tmp['rs_b'] = rs_b
            error_tmp['ti_b'] = ti_b
            error_tmp['rs_rel'] = rs_rel
            error_tmp['um_rel'] = um_rel
            error_tmp['ti_rel'] = ti_rel
            error_tmp['rs_rel_b'] = rs_rel_b
            error_tmp['um_rel_b'] = um_rel_b
            error_tmp['ti_rel_b'] = ti_rel_b

            errors[f'{geom}_{run}'] = error_tmp
            overview[f'{geom}_{run}'] = pd.read_csv(p / 'overview.csv', index_col=0)

            config_file = p / 'config.json'
            config = json.loads(config_file.read_bytes())
            # Velocity scale
            U = config['FLOW_PROPERTIES']['mean_velocity']
            ux[f'{geom}_{run}'] = U[0]


# Plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='6'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["figure.figsize"] = (6,6)
cmap = matplotlib.cm.get_cmap('RdYlBu')
markers = ['^','o','s','*','<','>']
colors = ['r','g','b']
N = 3
tau = r'$\tau$'


# Plot errors and rel. errors compared to disp. phase time series

#geometries = ['geom1','geom2','geom3','geom4','geom5','geom6','geom7','geom8','geom9','geom10','geom11','geom12']
#geometries = ['geom1','geom2','geom3','geom4','geom9','geom10','geom13']
geometries = ['geom1']
#runs = ['001','002','003','004','005','006','007']
runs = ['006','010','011']

# v_labels = {'Ux':'u_x','Uy':'u_y','Uz':'u_z'}
# y_lims = [[0,5],[0,10],[0,10]]
# precision = [1,1,1]
# var = 'u'
# variables = ['Ux','Uy','Uz']
# N = len(geometries)
# w = 0.9/N  # the width of the bars
# x = np.arange(len(runs))  # the label locations
# shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

# fig, axs = plt.subplots(3,1,figsize=(6,5))
# for kk,v in enumerate(variables):
#     rects = {}
#     for ii,geom in enumerate(geometries):
#         y=[]
#         for run in runs:
#             y.append(errors[f'{geom}_{run}'][var][v]['MAE_rec']*100)
#         rects[ii] = axs[kk].bar(x + shifts[ii], y, w, color=cmap(ii/N),label=labels[geom])
#     # Add some text for labels, title and custom x-axis tick labels, etc.
#     axs[kk].set_ylabel(f'MAE ${v_labels[v]}$ [cm/s]')
#     axs[kk].set_ylim(y_lims[kk])
#     axs[kk].set_xticks(x)
#     axs[kk].set_xticklabels(runs, rotation=60)
#     for key in rects:
#         axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f')
#     axs[kk].grid()
# axs[0].legend(loc=1,bbox_to_anchor=(1.31,1))
# plt.subplots_adjust(left=0.15, bottom=0.09, right=0.80, top=0.98, wspace=0.2, hspace=0.4)
# fig.savefig(path_out / 'MAE_u_bars_new.svg',dpi=300)

v_labels = {'Ux':'u_x','Uy':'u_y','Uz':'u_z'}
y_lims = [[0,5],[0,25],[0,25]]
precision = [1,1,1]
var = 'u'
variables = ['Ux','Uy','Uz']
N = len(geometries)
w = 0.9/N  # the width of the bars
x = np.arange(len(runs))  # the label locations
shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v]['MAE_rec']*100)
        rects[ii] = axs[kk].bar(x + shifts[ii], y, w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'MAE ${v_labels[v]}$ [cm/s]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'MAE_u_bars.svg',dpi=300)

v_labels = {'Ux':'u_x','Uy':'u_y','Uz':'u_z'}
y_lims = [[0.000001,0.1],[0.000001,1],[0.0000001,1],[0.000001,0.1],[0.0000001,1],[0.0000001,0.01]]
precision = [5,5,5,5,5,5]
var = 'rs'
variables = [('x','x'),('y','y'),('z','z'),('x','y'),('x','z'),('y','z')]
N = len(geometries)
w = 0.9/N  # the width of the bars
x = np.arange(len(runs))  # the label locations
shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

fig, axs = plt.subplots(6,1,figsize=(14,18))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v[0]][v[1]])
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    vv = r'$_{%s}$'%(v[0]+v[1])
    axs[kk].set_ylabel(f'AE {tau}{vv} [m$^2$/s$^2$]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'AE_tau_bars.svg',dpi=300)

v_labels = {'Ux':'u_x','Uy':'u_y','Uz':'u_z'}
y_lims = [[0.1,100],[0.1,10000],[0.01,10000],[0.1,1000],[1,100000],[1,10000]]
precision = [1,2,2,1,2,2]
var = 'rs_rel'
variables = [('x','x'),('y','y'),('z','z'),('x','y'),('x','z'),('y','z')]
N = len(geometries)
w = 0.9/N  # the width of the bars
x = np.arange(len(runs))  # the label locations
shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

fig, axs = plt.subplots(6,1,figsize=(14,18))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v[0]][v[1]]*100.0)
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    vv = r'$_{%s}$'%(v[0]+v[1])
    axs[kk].set_ylabel(f'ARE {tau}{vv} [%]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'ARE_tau_bars.svg',dpi=300)

v_labels = {'Tix':'T_{I,x}','Tiy':'T_{I,y}','Tiz':'T_{I,z}'}
y_lims = [[0.00001,0.01],[0.00001,0.1],[0.000001,0.1]]
precision = [5,5,5]
var = 'ti'
variables = ['Tix','Tiy','Tiz']
fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v][0])
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'AE ${v_labels[v]}$ [-]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'AE_ti_bars.svg',dpi=300)

v_labels = {'Tix':'T_{I,x}','Tiy':'T_{I,y}','Tiz':'T_{I,z}'}
y_lims = [[0.01,100],[0.01,1000],[0.01,1000]]
precision = [2,2,2]
var = 'ti_rel'
variables = ['Tix','Tiy','Tiz']
fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v][0]*100)
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'ARE ${v_labels[v]}$ [%]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'ARE_ti_bars.svg',dpi=300)

v_labels = {'ux':r'\bar{u}_x','uy':r'\bar{u}_y','uz':r'\bar{u}_z'}
var = 'um'
y_lims = [[0.001,1],[0.001,1],[0.001,1]]
precision = [3,3,3]
variables = ['ux','uy','uz']
fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v][0]*100)
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'AE ${v_labels[v]}$ [cm/s]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'AE_um_bars.svg',dpi=300)


# Plot errors and rel. errors compared to true bubble time series


v_labels = {'Ux':'u_x','Uy':'u_y','Uz':'u_z'}
y_lims = [[0.000001,0.1],[0.000001,1],[0.0000001,1],[0.000001,0.1],[0.0000001,1],[0.0000001,0.01]]
precision = [5,5,5,5,5,5]
var = 'rs_b'
variables = [('x','x'),('y','y'),('z','z'),('x','y'),('x','z'),('y','z')]
N = len(geometries)
w = 0.9/N  # the width of the bars
x = np.arange(len(runs))  # the label locations
shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

fig, axs = plt.subplots(6,1,figsize=(14,18))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v[0]][v[1]])
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    vv = r'$_{%s}$'%(v[0]+v[1])
    axs[kk].set_ylabel(f'AE {tau}{vv} [m$^2$/s$^2$]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'AE_tau_bubble_bars.svg',dpi=300)

v_labels = {'Ux':'u_x','Uy':'u_y','Uz':'u_z'}
y_lims = [[0.001,100],[0.01,10000],[0.01,10000],[0.01,1000],[0.1,100000],[0.1,10000]]
precision = [3,2,2,2,2,2]
var = 'rs_rel_b'
variables = [('x','x'),('y','y'),('z','z'),('x','y'),('x','z'),('y','z')]
N = len(geometries)
w = 0.9/N  # the width of the bars
x = np.arange(len(runs))  # the label locations
shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

fig, axs = plt.subplots(6,1,figsize=(14,18))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v[0]][v[1]]*100.0)
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    vv = r'$_{%s}$'%(v[0]+v[1])
    axs[kk].set_ylabel(f'ARE {tau}{vv} [%]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'ARE_tau_bubble_bars.svg',dpi=300)

v_labels = {'Tix':'T_{I,x}','Tiy':'T_{I,y}','Tiz':'T_{I,z}'}
y_lims = [[0.0000001,0.01],[0.000001,0.1],[0.000001,0.1]]
precision = [6,5,5]
var = 'ti_b'
variables = ['Tix','Tiy','Tiz']
fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v][0])
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'AE ${v_labels[v]}$ [-]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'AE_ti_bubble_bars.svg',dpi=300)

v_labels = {'Tix':'T_{I,x}','Tiy':'T_{I,y}','Tiz':'T_{I,z}'}
y_lims = [[0.001,100],[0.01,1000],[0.01,1000]]
precision = [3,3,3]
var = 'ti_rel_b'
variables = ['Tix','Tiy','Tiz']
fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v][0]*100)
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'ARE ${v_labels[v]}$ [%]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'ARE_ti_bubble_bars.svg',dpi=300)


v_labels = {'ux':r'\bar{u}_x','uy':r'\bar{u}_y','uz':r'\bar{u}_z'}
var = 'um_b'
y_lims = [[0.0001,1],[0.00001,1],[0.0001,1]]
precision = [3,3,3]
variables = ['ux','uy','uz']
fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for run in runs:
            y.append(errors[f'{geom}_{run}'][var][v][0]*100)
        rects[ii] = axs[kk].bar(x + shifts[ii], abs(np.asarray(y)), w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'AE ${v_labels[v]}$ [cm/s]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_yscale('log')
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)
    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='center')
    axs[kk].grid()
axs[0].legend(loc=1,bbox_to_anchor=(1.12,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'AE_um_bubble_bars.svg',dpi=300)


# Plot actual value comparison

v_labels = {'Ux':r'\bar{u}_x','Uy':r'\bar{u}_y','Uz':r'\bar{u}_z'}
y_lims = [[0,25],[-0.02,0.0],[-0.015,0.01]]
variables = ['Ux','Uy','Uz']
precision = [1,4,4]
N = len(geometries)
w = 0.9/N  # the width of the bars
x = np.arange(len(runs))  # the label locations
shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for ll,run in enumerate(runs):
            axs[kk].hlines(overview[f'{geom}_{run}']['DP_bubble'][v], x[ll]+shifts[ii]-w/2, x[ll]+shifts[ii]+w/2, color='r',linestyle='-')
            axs[kk].hlines(overview[f'{geom}_{run}']['DP'][v], x[ll]+shifts[ii]-w/2, x[ll]+shifts[ii]+w/2, color='b',linestyle=':')
            y.append(overview[f'{geom}_{run}']['DP_reconstructed'][v])
        rects[ii] = axs[kk].bar(x + shifts[ii], y, w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'${v_labels[v]}$ [m/s]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)

    for key in rects:
        axs[kk].bar_label(rects[key], padding=5, rotation=90,fmt=f'%.{precision[kk]}f',label_type='edge',fontsize=8)
    axs[kk].grid()
axs[0].plot([],[], color='r',linestyle='-',label='Disp. Phase\n(bubbles)')
axs[0].plot([],[], color='b',linestyle=':',label='Disp. Phase\n(all)')
axs[0].legend(loc=1,bbox_to_anchor=(1.13,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'Overview_U_bars.svg',dpi=300)


v_labels = {'Tau_xx':r'\bar{\tau}_{xx}','Tau_yy':r'\bar{\tau}_{yy}','Tau_zz':r'\bar{\tau}_{zz}','Tau_xy':r'\bar{\tau}_{xy}','Tau_xz':r'\bar{\tau}_{xz}','Tau_yz':r'\bar{\tau}_{yz}'}
y_lims = [[0,0.08],[0,0.08],[0.0,0.04],[-0.006,0.006],[-0.01,0.0],[-0.0005,0.0005]]
variables = ['Tau_xx','Tau_yy','Tau_zz','Tau_xy','Tau_xz','Tau_yz']
precision = [5,5,5,5,5,5]
N = len(geometries)
w = 0.9/N  # the width of the bars
x = np.arange(len(runs))  # the label locations
shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

fig, axs = plt.subplots(6,1,figsize=(14,18))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for ll,run in enumerate(runs):
            axs[kk].hlines(overview[f'{geom}_{run}']['DP_bubble'][v], x[ll]+shifts[ii]-w/2, x[ll]+shifts[ii]+w/2, color='r',linestyle='-')
            axs[kk].hlines(overview[f'{geom}_{run}']['DP'][v], x[ll]+shifts[ii]-w/2, x[ll]+shifts[ii]+w/2, color='b',linestyle=':')
            y.append(overview[f'{geom}_{run}']['DP_reconstructed'][v])
        rects[ii] = axs[kk].bar(x + shifts[ii], y, w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'${v_labels[v]}$ [Pa]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)

    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='edge',fontsize=8)
    axs[kk].grid()
axs[0].plot([],[], color='r',linestyle='-',label='Disp. Phase\n(bubbles)')
axs[0].plot([],[], color='b',linestyle=':',label='Disp. Phase\n(all)')
axs[0].legend(loc=1,bbox_to_anchor=(1.13,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'Overview_RST_bars.svg',dpi=300)

# Plot actual value comparison

v_labels = {'T_Ix':r'T_{I,x}','T_Iy':r'T_{I,y}','T_Iz':r'T_{I,z}'}
y_lims = [[0,0.1],[0,0.03],[0.0,0.03]]
variables = ['T_Ix','T_Iy','T_Iz']
precision = [4,4,4]
N = len(geometries)
w = 0.9/N  # the width of the bars
x = np.arange(len(runs))  # the label locations
shifts = np.linspace(-(N*w/2.0-0.5*w),(N*w/2.0-0.5*w),N)

fig, axs = plt.subplots(3,1,figsize=(14,9))
for kk,v in enumerate(variables):
    rects = {}
    for ii,geom in enumerate(geometries):
        y=[]
        for ll,run in enumerate(runs):
            axs[kk].hlines(overview[f'{geom}_{run}']['DP_bubble'][v], x[ll]+shifts[ii]-w/2, x[ll]+shifts[ii]+w/2, color='r',linestyle='-')
            axs[kk].hlines(overview[f'{geom}_{run}']['DP'][v], x[ll]+shifts[ii]-w/2, x[ll]+shifts[ii]+w/2, color='b',linestyle=':')
            y.append(overview[f'{geom}_{run}']['DP_reconstructed'][v])
        rects[ii] = axs[kk].bar(x + shifts[ii], y, w, color=cmap(ii/N),label=labels[geom])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[kk].set_ylabel(f'${v_labels[v]}$ [-]')
    axs[kk].set_ylim(y_lims[kk])
    axs[kk].set_xticks(x)
    axs[kk].set_xticklabels(runs, rotation=60)

    for key in rects:
        axs[kk].bar_label(rects[key], padding=3, rotation=90,fmt=f'%.{precision[kk]}f',label_type='edge',fontsize=8)
    axs[kk].grid()
axs[0].plot([],[], color='r',linestyle='-',label='Disp. Phase\n(bubbles)')
axs[0].plot([],[], color='b',linestyle=':',label='Disp. Phase\n(all)')
axs[0].legend(loc=1,bbox_to_anchor=(1.13,1))
plt.subplots_adjust(left=0.07, bottom=0.09, right=0.89, top=0.98, wspace=0.2, hspace=0.3)
fig.savefig(path_out / 'Overview_TI_bars.svg',dpi=300)