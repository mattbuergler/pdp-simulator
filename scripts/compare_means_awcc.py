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
import matplotlib
from matplotlib import pyplot as plt
try:
    from dataio.H5Writer import H5Writer
    from dataio.H5Reader import H5Reader
    from tools.globals import *
except ImportError:
    print("Error while importing modules")
    raise


# Create parser to read in the configuration JSON-file to read from
# the command line interface (CLI)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('path', type=str,
    help="The path to the scenario directory.")
args = parser.parse_args()
path = pathlib.Path(args.path)

nps = [5,7,10,13,15]
errors = {}
z_values = []
for np in nps:
    # Create Posix path for OS indepency
    errors_np = []
    z_values = []
    path_np = path / f'awcc_np{np}/J50_Q300_ks1'
    for z in path_np.iterdir():
        path_np_z_run = path_np / z.name / 'run'
        # Create a H5-file reader
        reader = H5Reader(path_np_z_run / 'flow_data.h5')
        mean_true = reader.getDataSet('bubbles/mean_velocity')[:]
        reader.close()
        # Calculate volumen equivalent diameter
        # Create a H5-file reader
        reader = H5Reader(path_np_z_run / 'reconstructed.h5')
        mean_rec = reader.getDataSet('bubbles/weighted_mean_velocity')[:]
        reader.close()
        error = mean_rec - mean_true
        errors_np.append(error[0])
        z_values.append(float(z.name.replace('z','')))
    error_frame = pd.DataFrame(z_values,columns=['z'])
    error_frame['error'] = errors_np
    error_frame = error_frame.sort_values('z')
    errors[np] = error_frame

cmap = matplotlib.cm.get_cmap('viridis')
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 10
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['axes.labelsize'] = 9
plt.rcParams["legend.frameon"] = False

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='6'
plt.rcParams["legend.frameon"]
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['xtick.major.pad']='6'
plt.rcParams['ytick.major.pad']='6'
plt.rcParams['mathtext.fontset'] = 'cm'

fig, ax = plt.subplots(1,1,figsize=(4,2.88))
jj = 0
for np, error_frame in errors.items():
    plt.plot(error_frame['error'],error_frame['z'],color=cmap(jj/len(nps)),label=f'$N_p$ = {np}')
    jj += 1
ax.set_ylabel(r'$z$ [mm]')
ax.set_xlabel(r'Error $\bar(u_x)$ [m/s]')
# ax.set_xlim([0.0,1.0])
# ax.set_ylim([0.0,1.0])
ax.grid()
ax.legend()
fig.savefig(path / 'error_mean_velocity.svg',dpi=300)
