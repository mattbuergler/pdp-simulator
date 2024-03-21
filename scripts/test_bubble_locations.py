#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:58:13 2023

@author: matthias
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 07:43:04 2023

@author: matthias
"""
import pathlib
import numpy as np
import random
import matplotlib
import json
import argparse
import math
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

try:
    from dataio.H5Reader import H5Reader
    from tools.globals import *
    from sbg_functions import *
except ImportError:
    print("Error while importing modules")
    raise

# Create parser to read in the configuration JSON-file to read from
# the command line interface (CLI)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('path', type=str,
    help="The path to the LDA time series file")
parser.add_argument(
    "-o", "--overwrite", metavar="OVERWRITE", default='no', help="overwrite existing data? [yes/no; default=no]",
)
args = parser.parse_args()
# Create Posix path for OS indepency
path = pathlib.Path(args.path)

# path =  pathlib.Path(f'tests/3_RA/A_Triangular_Shen/run')

reader = H5Reader(path / 'flow_data.h5')
# Read the time vector
arrival_locations = reader.getDataSet('bubbles/arrival_locations')[:]
exit_locations = reader.getDataSet('bubbles/exit_locations')[:]
arrival_times = reader.getDataSet('bubbles/arrival_times')[:]
exit_times = reader.getDataSet('bubbles/exit_times')[:]

input_file = path / 'config.json'
# Read the configuation JSON-file
config = json.loads(input_file.read_bytes())
flow_properties=config['FLOW_PROPERTIES']
probe=config['PROBE']
sensors = probe['sensors']
min_range = np.array([LARGENUMBER,LARGENUMBER,LARGENUMBER])
max_range = np.array([LARGENEGNUMBER,LARGENEGNUMBER,LARGENEGNUMBER])

for sensor in sensors:
    for ii in range(0,3):
        min_range[ii] = min(min_range[ii], sensor['relative_location'][ii])
        max_range[ii] = max(max_range[ii], sensor['relative_location'][ii])

max_probe_size = max_range - min_range
# Determine control volume (CV) size
# get the mean sphere-volume-equivalent bubble diameter
d = get_mean_bubble_sve_size(flow_properties)
V_b = math.pi/6.0*d**3
# get max bubble size
d_max = get_max_bubble_sve_size(flow_properties)
control_volume_size = np.zeros(3)
# length
control_volume_size[0] = 2*d_max + max_probe_size[0]
# width
control_volume_size[1] = 2*d + max(max_probe_size[1],max_probe_size[2])
control_volume_size[2] = 2*d + max(max_probe_size[1],max_probe_size[2])


center = (0.0, 0.0)
radius = d/2.0
circle = Circle(center, radius,ec='b',fc=None)
circle2 = Circle(center, radius,ec='b',fc=None)

f1 = plt.figure(figsize=(4,3))        
a1 = f1.gca()
sc = a1.scatter(arrival_locations[:,1],arrival_locations[:,2],marker='.',c=arrival_times,s=2,zorder=1)
a1.plot([-control_volume_size[1]/2.0,control_volume_size[1]/2.0],[-control_volume_size[2]/2.0,-control_volume_size[2]/2.0],color='k')
a1.plot([-control_volume_size[1]/2.0,control_volume_size[1]/2.0],[control_volume_size[2]/2.0,control_volume_size[2]/2.0],color='k')
a1.plot([-control_volume_size[1]/2.0,-control_volume_size[1]/2.0],[-control_volume_size[2]/2.0,control_volume_size[2]/2.0],color='k')
a1.plot([control_volume_size[1]/2.0,control_volume_size[1]/2.0],[-control_volume_size[2]/2.0,control_volume_size[2]/2.0],color='k')
a1.add_patch(circle)
a1.set_xlabel('$y$ [m]')
a1.set_ylabel('$z$ [m]')
a1.set_xlim([-0.01,0.01])
a1.set_ylim([-0.01,0.01])
plt.colorbar(sc,label='arrival time [s]')
f1.tight_layout()
plt.gca().set_aspect('equal')
f1.savefig(path / f"arrival_locations.pdf")
f1.savefig(path / f"arrival_locations.svg")

# Histogram for y- and z-axis
fig, axs = plt.subplots(ncols=2,\
        figsize=(2*2.5,2.5))
for jj,axis in enumerate(['$y$','$z$']):
    idx = jj+1
    (n, bins, patches) = axs[jj].hist(arrival_locations[:,idx], \
        bins=100, density=False, alpha=1.0,\
        range=(np.nanmin(arrival_locations[:,idx]),np.nanmax(arrival_locations[:,idx])))
    if jj == 0:
        axs[jj].set_ylabel('Frequency [-]')
    axs[jj].set_xlabel(f'{axis} [m]')
    axs[jj].set_xlim([-max(abs(bins)),max(abs(bins))])
plt.tight_layout()
fig.savefig(path / 'histograms.svg',dpi=300)
fig.savefig(path / 'histograms.pdf',dpi=300)

f1 = plt.figure(figsize=(4,3))        
a1 = f1.gca()
sc = a1.scatter(exit_locations[:,1],exit_locations[:,2],marker='.',c=exit_times,s=2,zorder=1)
a1.plot([-control_volume_size[1]/2.0,control_volume_size[1]/2.0],[-control_volume_size[2]/2.0,-control_volume_size[2]/2.0],color='k')
a1.plot([-control_volume_size[1]/2.0,control_volume_size[1]/2.0],[control_volume_size[2]/2.0,control_volume_size[2]/2.0],color='k')
a1.plot([-control_volume_size[1]/2.0,-control_volume_size[1]/2.0],[-control_volume_size[2]/2.0,control_volume_size[2]/2.0],color='k')
a1.plot([control_volume_size[1]/2.0,control_volume_size[1]/2.0],[-control_volume_size[2]/2.0,control_volume_size[2]/2.0],color='k')
a1.add_patch(circle2)
a1.set_xlabel('$y$ [m]')
a1.set_ylabel('$z$ [m]')
a1.set_xlim([-0.01,0.01])
a1.set_ylim([-0.01,0.01])
plt.colorbar(sc,label='exit time [s]')
f1.tight_layout()
plt.gca().set_aspect('equal')
f1.savefig(path / f"exit_locations.pdf")
f1.savefig(path / f"exit_locations.svg")