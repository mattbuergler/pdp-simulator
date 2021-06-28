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

path = pathlib.Path().cwd() / 'optimization/geometry'
path_out = pathlib.Path().cwd() / 'optimization/geometry/results'

geometries = ['geom1','geom2','geom3','geom4']
#runs = ['001','002','003','004']
runs = ['001','002','003','004','005','006','007','008']

errors = {}
ux = {}
for geom in geometries:
    for run in runs:
        error_tmp = {}
        p = path / geom / run / 'run'
        u = pd.read_csv(p / 'error_summary_velocity.csv', index_col=0)
        d = pd.read_csv(p / 'error_summary_bubble_size.csv', index_col=0)
        rs = pd.read_csv(p / 'error_reynolds_stresses.csv', index_col=0)
        um = pd.read_csv(p / 'error_mean_velocity.csv', index_col=0)
        rs_rel = pd.read_csv(p / 'rel_error_reynolds_stresses.csv', index_col=0)
        um_rel = pd.read_csv(p / 'rel_error_mean_velocity.csv', index_col=0)
        error_tmp['u'] = u
        error_tmp['d'] = d
        error_tmp['rs'] = rs
        error_tmp['um'] = um
        error_tmp['rs_rel'] = rs_rel
        error_tmp['um_rel'] = um_rel
        errors[f'{geom}_{run}'] = error_tmp

        config_file = p / 'config.json'
        config = json.loads(config_file.read_bytes())
        # Velocity scale
        U = config['FLOW_PROPERTIES']['mean_velocity']
        ux[f'{geom}_{run}'] = U[0]
        # f = config['PROBE']['sampling_frequency']
        # sensors = config['PROBE']['sensors']
        # minmax = np.array([[LARGE_NUMBER,LARGE_NEG_NUMBER],
        #                 [LARGE_NUMBER,LARGE_NEG_NUMBER],
        #                 [LARGE_NUMBER,LARGE_NEG_NUMBER]])
        # for sensor in sensors:
        #     for ii in range(0,3):
        #         minmax[ii,0] = min(minmax[ii,0], sensor['relative_location'][ii])
        #         minmax[ii,1] = max(minmax[ii,1], sensor['relative_location'][ii])
        # normalized_var[f'{geom}_{run}'] = 1./np.asarray([(minmax[0,1]-minmax[0,0])*inverse_den(U[0])*f,
        #         (minmax[1,1]-minmax[1,0])*inverse_den(U[1])*f,
        #         (minmax[2,1]-minmax[2,0])*inverse_den(U[2])*f])

# Plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='6'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["figure.figsize"] = (6,6)
cmap = matplotlib.cm.get_cmap('viridis')
markers = ['^','o','s','*','<','>']
colors = ['r','g','b']
N = 3
labels = {'geom1':'Probe1','geom2':'Probe2','geom3':'Probe3','geom4':'Probe4'}
geometries = ['geom1','geom2','geom3','geom3']
runs = ['001','002','003','004']

# Plot errors as function of sampling frequency
var = 'u'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['Ux']['MARE_rec']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'MARE $u_x$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
axs.set_ylim([0.01,1])
# axs.set_xlim([min(var),max(var)])
axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=2,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'MARE_velocity_ux.svg',dpi=300)

# Plot errors as function of sampling frequency
var = 'u'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['Uy']['MARE_rec']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'MARE $u_y$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
axs.set_ylim([10,10000])
# axs.set_xlim([min(var),max(var)])
axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=2,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'MARE_velocity_uy.svg',dpi=300)

# Plot errors as function of sampling frequency
var = 'u'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['Uz']['MARE_rec']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'MARE $u_z$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
axs.set_ylim([10,10000])
# axs.set_xlim([min(var),max(var)])
axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=2,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'MARE_velocity_uz.svg',dpi=300)


# Plot errors as function of sampling frequency
var = 'um_rel'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['ux']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'Relative Error $\overline{u}_x$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
axs.set_ylim([-0.2,0.1])
# axs.set_xlim([min(var),max(var)])
# axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=8,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'RE_Mean_velocity.svg',dpi=300)

# Plot errors as function of sampling frequency
var = 'rs_rel'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['x']['x']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'Relative Error $\tau_{xx}$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
axs.set_ylim([-10,30])
# axs.set_xlim([min(var),max(var)])
# axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=2,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'RE_tau_xx.svg',dpi=300)

# Plot errors as function of sampling frequency
var = 'rs_rel'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['y']['y']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'Relative Error $\tau_{yy}$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
axs.set_ylim([-10,600])
# axs.set_xlim([min(var),max(var)])
# axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=2,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'RE_tau_yy.svg',dpi=300)

# Plot errors as function of sampling frequency
var = 'rs_rel'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['z']['z']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'Relative Error $\tau_{zz}$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
axs.set_ylim([-10,600])
# axs.set_xlim([min(var),max(var)])
# axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=2,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'RE_tau_zz.svg',dpi=300)

# Plot errors as function of sampling frequency
var = 'rs_rel'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['x']['y']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'Relative Error $\tau_{xy}$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
# axs.set_ylim([0.001,1])
# axs.set_xlim([min(var),max(var)])
# axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=1,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'RE_tau_xy.svg',dpi=300)

# Plot errors as function of sampling frequency
var = 'rs_rel'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['x']['z']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'Relative Error $\tau_{xz}$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
# axs.set_ylim([0.001,1])
# axs.set_xlim([min(var),max(var)])
# axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=2,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'RE_tau_xz.svg',dpi=300)

# Plot errors as function of sampling frequency
var = 'rs_rel'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}']) 
        y.append(errors[f'{geom}_{run}'][var]['y']['z']*100) 
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'Relative Error $\tau_{yz}$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
# axs.set_ylim([0.001,1])
# axs.set_xlim([min(var),max(var)])
# axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=1,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'RE_tau_yz.svg',dpi=300)

# Plot errors as function of sampling frequency
N = 3
var = 'd'
fig, axs = plt.subplots(1,1,figsize=(3,3))
for ii,geom in enumerate(geometries):
    x=[]
    y=[]
    for run in runs:
        x.append(ux[f'{geom}_{run}'])
        y.append(errors[f'{geom}_{run}'][var]['D']['MARE']*100)
    axs.plot(x,y,marker=markers[ii],color=cmap(ii/N),label=labels[geom])
axs.set_ylabel(r'MARE $D$ [%]')
axs.set_xlabel(r'$\overline{u}_x$ [-]')
axs.set_ylim([0.0,0.5])
# axs.set_xlim([min(var),max(var)])
# axs.set_yscale('log')
# axs.set_xscale('log')
axs.legend(loc=2,fontsize=9)
plt.tight_layout()
plt.grid()
fig.savefig(path_out / 'MARE_diameter.svg',dpi=300)
