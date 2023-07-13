#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:48:27 2021

@author: matthias
"""
import argparse
import json
import numpy as np
import pandas as pd
import time
import math
import pathlib
from statsmodels.tsa.stattools import acf, pacf
from scipy import signal
import matplotlib
from matplotlib import pyplot as plt
try:
    from dataio.H5Writer import H5Writer
    from dataio.H5Reader import H5Reader
    from tools.globals import *
except ImportError:
    print("Error while importing modules")
    raise

"""
    Velocity timeseries analysis

"""

def main():
    """
        Main function of the velocity timeseries analysis

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
    reader = H5Reader(path / 'reconstructed_np5_z060.h5')
    # Read the cont. velocity
    chord_len_np5 = reader.getDataSet('bubbles/chord_lengths')[:]
    reader.close()
    # Create a H5-file reader
    reader = H5Reader(path / 'reconstructed_np15_z060.h5')
    # Read the cont. velocity
    chord_len_np15 = reader.getDataSet('bubbles/chord_lengths')[:]
    reader.close()
    print(chord_len_np5 == chord_len_np15)
    print(sum(chord_len_np5 == chord_len_np15)/len(chord_len_np15))
    print(max(chord_len_np5 - chord_len_np15))
    sys.exit()
    # Histogram of bubble velocities
    var_names = ['x']
    limits = [8,16]
    col=[0.0,0.2,0.5]
    cmap = matplotlib.cm.get_cmap('viridis')
    plt.rcParams['font.size'] = 7
    plt.rcParams['legend.fontsize'] = 7
    plt.rcParams['figure.titlesize'] = 7
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['axes.labelsize'] = 7
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

    # Calculate the magnitude of the input velocities and standard dev.
    fig, axs = plt.subplots(ncols=1,\
            figsize=(2.5,2))
    for jj,var_name in enumerate(var_names):
        (n_c, bins_c, patches_c) = axs.hist(np.clip(u_p_true[:,jj],limits[0],limits[1]), \
            bins=np.linspace(limits[0],limits[1],int((limits[1]-limits[0])/0.25+1)), density=True, color='k', alpha=1.0, label='sim.',\
            range=(limits[0],limits[1]),histtype='step')
        (n_c, bins_c, patches_c) = axs.hist(np.clip(u_p_rec[:,jj],limits[0],limits[1]), \
            bins=np.linspace(limits[0],limits[1],int((limits[1]-limits[0])/0.25+1)), density=True, alpha=0.5, label='AWCC',\
            range=(limits[0],limits[1]))
        (n_c, bins_c, patches_c) = axs.hist(np.clip(u_p_rec_roc[:,jj],limits[0],limits[1]), \
            bins=np.linspace(limits[0],limits[1],int((limits[1]-limits[0])/0.25+1)), density=True, alpha=0.5, label='AWCC+ROC',\
            range=(limits[0],limits[1]))
        if jj == 0:
            axs.set_ylabel('Frequency [-]')
        axs.set_xlabel(f'Velocity $u_{var_name}$ [m/s]')
        axs.vlines(np.nanmedian(u_p_rec_roc[:,jj])+1.3118,0,max(n_c)*10, color='r', label='$\delta_{ROC}$')
        axs.vlines(np.nanmedian(u_p_rec_roc[:,jj])-1.3118,0,max(n_c)*10, color='r')
        axs.set_xlim(limits)
        axs.set_ylim([0,1.6])
        axs.legend(loc=1,fontsize=7,ncol=1,frameon=False,fancybox=False,facecolor='w',edgecolor='k')
        # if jj == 1:
        #     axs[jj].set_yscale('log')
        # if jj == 2:
        #     axs[jj].set_yscale('log')
    plt.subplots_adjust(left=0.21, bottom=0.25, right=0.95, top=0.95, wspace=0.3, hspace=None)
    fig.savefig(path / 'velocity_histrograms.svg',dpi=300)
    fig.savefig(path / 'velocity_histrograms.png',dpi=300)

if __name__ == "__main__":
    main()