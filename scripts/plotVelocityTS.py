#!/usr/bin/env python3

import sys
import argparse
import pathlib
import json
import jsonschema
import math
import numpy as np
from numpy import linalg
import pandas as pd
import time
import math
import matplotlib
from matplotlib import pyplot as plt
import sys
import decimal
from scipy import signal
from decimal import *
try:
    from dataio.H5Writer import H5Writer
    from dataio.H5Reader import H5Reader
    from tools.globals import *
except ImportError:
    print("Error while importing modules")
    raise


def main():
    """
        Plot velocity time series
    """

    # Start timer
    time1 = time.time()
    # Create parser to read in the configuration JSON-file to read from
    # the command line interface (CLI)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=str,
        help="The path to the scenario directory.")
    args = parser.parse_args()

    # Create Posix path for OS indepency
    path = pathlib.Path(args.path)

    # Create the H5-file writer
    writer = H5Writer(path / 'reconstructed.h5', 'a')
    # Read the time vector
    time_reconst = writer.getDataSet('bubbles/interaction_times')[:]
    # Read the bubbles velocity
    velocity_reconst = writer.getDataSet('bubbles/velocity')[:]
    writer.close()

    cmap = plt.cm.get_cmap('viridis')
    plt.rcParams['font.size'] = 9
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

    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,0],color='k')
    plt.xlabel('$t$ [s]')
    plt.ylabel(r'$u_x$ [m/s]')
    plt.ylim([-5,20])
    plt.grid(which='major', axis='both')
    plt.tight_layout()
    fig.savefig(path /'velocity_time_series.svg',dpi=300)

    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,0],color='k')
    plt.xlabel('$t$ [s]')
    plt.ylabel(r'$u_x$ [m/s]')
    plt.grid(which='major', axis='both')
    plt.xlim([0.5*np.nanmax(time_reconst),0.5*np.nanmax(time_reconst)+0.5])
    plt.tight_layout()
    fig.savefig(path /'velocity_time_series_zoom.svg',dpi=300)

if __name__ == "__main__":
    main()