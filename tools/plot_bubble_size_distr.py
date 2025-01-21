#!/usr/bin/env python3

import sys
import argparse
import pathlib
import json
import jsonschema
import numpy as np
import pandas as pd
import bisect
import time
import math
import matplotlib.pyplot as plt
import scipy.stats as stats
try:
    from dataio.H5Writer import H5Writer
    from dataio.H5Reader import H5Reader
    from tools.globals import *
except ImportError:
    print("Error while importing modules")
    raise

"""
    Evaluate the reconstructed data
"""
# Create parser to read in the configuration JSON-file to read from
# the command line interface (CLI)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('path', type=str,
    help="The path to the scenario directory.")
args = parser.parse_args()

# Create Posix path for OS indepency
path = pathlib.Path(args.path)
# Create a H5-file reader
reader = H5Reader(path / 'flow_data.h5')
b_size = reader.getDataSet('bubbles/size')[:]
b_size = pd.DataFrame(b_size, columns=['A','B','C'])
reader.close()
# Calculate volumen equivalent diameter
b_size['D'] = (b_size['A']*b_size['B']*b_size['C'])**(1.0/3.0)
# Create a H5-file reader
reader = H5Reader(path / 'reconstructed.h5')
b_size_rec = reader.getDataSet('bubbles/diameters')[:]
b_size_rec = np.nanmean(np.abs(b_size_rec),axis=1)
b_size_rec = pd.DataFrame(b_size_rec,columns=['D'])
reader.close()
b_size_rec = b_size_rec[~b_size_rec['D'].isnull()]

# Histogram of velocities
var_names = ['A','B','D']
var_labels = ['$A$ [mm]','$B$ [mm]','$D_{eq}$ [mm]']
b_size['E'] = b_size['A']/b_size['B']
fig, axs = plt.subplots(ncols=3,\
        figsize=(3*2,2.5))
for jj,var_name in enumerate(var_names):
    if var_name in ['A','B','D']:
        (n, bins, patches) = axs[jj].hist(b_size[var_name]*1000.0, \
            bins=np.linspace(0,20,81), density=True, color='b', alpha=1.0,\
            range=(min(b_size[var_name]*1000.0),max(b_size[var_name]*1000.0)))
    else:
        (n, bins, patches) = axs[jj].hist(b_size[var_name], \
            bins=np.linspace(0,20,81), density=True, color='b', alpha=1.0,\
            range=(min(b_size[var_name]),max(b_size[var_name])))
    if jj == 0:
        axs[jj].set_ylabel('Frequency [-]')
    if var_name in ['A','B','D']:
        axs[jj].text(2.7,0.9*max(n),f'$\mu$ = {b_size[var_name].mean()*1000.0:.2f}')
    else:
        axs[jj].text(0.5,0.9*max(n),f'$\mu$ = {b_size[var_name].mean():.2f}')
    axs[jj].set_xlabel(f'{var_labels[jj]}')
    if var_name in ['A','B','D']:
        axs[jj].set_xlim([0.0,5])
        axs[jj].set_xticks([0,1,2,3,4,5])
    else:
        axs[jj].set_xlim([0.4,1.0])
        axs[jj].set_xticks([0.4,0.6,0.8,1])
    #     axs[jj].set_xscale('log')
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=None)
fig.savefig(path / 'diameter_histrograms_input.svg',dpi=300)

# Histogram of velocities
var_names = ['A','B','D']
var_labels = ['$A$ [mm]','$B$ [mm]','$D_{eq}$ [mm]']
b_size['E'] = b_size['A']/b_size['B']
fig, axs = plt.subplots(ncols=3,\
        figsize=(3*2,2.5))
for jj,var_name in enumerate(var_names):
    if var_name in ['A','B','D']:
        (n, bins, patches) = axs[jj].hist(b_size[var_name]*1000.0, \
            bins=np.linspace(0,20,81), density=False, color='b', alpha=1.0,\
            range=(min(b_size[var_name]*1000.0),max(b_size[var_name]*1000.0)))
    else:
        (n, bins, patches) = axs[jj].hist(b_size[var_name], \
            bins=np.linspace(0,20,81), density=False, color='b', alpha=1.0,\
            range=(min(b_size[var_name]),max(b_size[var_name])))
    if jj == 0:
        axs[jj].set_ylabel('Frequency [-]')
    if var_name in ['A','B','D']:
        axs[jj].text(2.7,0.9*max(n),f'$\mu$ = {b_size[var_name].mean()*1000.0:.2f}')
    else:
        axs[jj].text(0.5,0.9*max(n),f'$\mu$ = {b_size[var_name].mean():.2f}')
    axs[jj].set_xlabel(f'{var_labels[jj]}')
    if var_name in ['A','B','D']:
        axs[jj].set_xlim([0.0,5])
        axs[jj].set_xticks([0,1,2,3,4,5])
    else:
        axs[jj].set_xlim([0.4,1.0])
        axs[jj].set_xticks([0.4,0.6,0.8,1])
    #     axs[jj].set_xscale('log')
(n_rec, bins_rec, patches_rec) = axs[2].hist(b_size_rec['D']*1000.0, \
    bins=np.linspace(0,20,81), density=False, color='r', alpha=0.7,\
    range=(min(b_size_rec['D']*1000.0),max(b_size_rec['D']*1000.0)))
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=None)
fig.savefig(path / 'diameter_histrograms_input_reconstructed.svg',dpi=300)


# Histogram of velocities
var_names = ['A','B','D']
var_labels = ['$A$ [mm]','$B$ [mm]','$D_{eq}$ [mm]']
b_size['E'] = b_size['A']/b_size['B']
fig, axs = plt.subplots(ncols=3,\
        figsize=(3*2,2.5))
for jj,var_name in enumerate(var_names):
    if var_name in ['A','B','D']:
        (n_rec, bins_rec, patches_rec) = axs[2].hist(b_size_rec['D']*1000.0, \
    bins=np.linspace(0,20,81), density=False, color='r', alpha=0.7,\
    range=(min(b_size_rec['D']*1000.0),max(b_size_rec['D']*1000.0)))
    if jj == 0:
        axs[jj].set_ylabel('Frequency [-]')
    if var_name in ['A','B','D']:
        axs[jj].text(2.7,0.9*max(n),f'$\mu$ = {b_size[var_name].mean()*1000.0:.2f}')
    else:
        axs[jj].text(0.5,0.9*max(n),f'$\mu$ = {b_size[var_name].mean():.2f}')
    axs[jj].set_xlabel(f'{var_labels[jj]}')
    if var_name in ['A','B','D']:
        axs[jj].set_xlim([0.0,5])
        axs[jj].set_xticks([0,1,2,3,4,5])
    else:
        axs[jj].set_xlim([0.4,1.0])
        axs[jj].set_xticks([0.4,0.6,0.8,1])
    #     axs[jj].set_xscale('log')
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=None)
fig.savefig(path / 'diameter_histrograms_reconstructed.svg',dpi=300)