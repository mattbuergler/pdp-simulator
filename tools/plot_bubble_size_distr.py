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
# Calculate volumen equivalent diameter
b_size['D'] = (b_size['A']*b_size['B']*b_size['C'])**(1.0/3.0)


# Histogram of velocities
var_names = ['A','B','D','E']
var_labels = ['$A$ [mm]','$B$ [mm]','$D_{eq}$ [mm]','$E$ [-]']
b_size['E'] = b_size['A']/b_size['B']
fig, axs = plt.subplots(ncols=4,\
        figsize=(4*2,2.5))
for jj,var_name in enumerate(var_names):
    if var_name in ['A','B','D']:
        (n, bins, patches) = axs[jj].hist(b_size[var_name]*1000.0, \
            bins=50, density=True, color='b', alpha=1.0,\
            range=(min(b_size[var_name]*1000.0),max(b_size[var_name]*1000.0)))
    else:
        (n, bins, patches) = axs[jj].hist(b_size[var_name], \
            bins=50, density=True, color='b', alpha=1.0,\
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
fig.savefig(path / 'diameter_histrograms.svg',dpi=300)