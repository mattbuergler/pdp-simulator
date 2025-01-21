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
    reader = H5Reader(path / 'flow_data.h5')
    # Read the cont. velocity
    u_p_true = reader.getDataSet('bubbles/velocity')[:]
    reader.close()
    # Create a H5-file reader
    reader = H5Reader(path / 'reconstructed_no_filter.h5')
    # Read the cont. velocity
    u_p_rec = reader.getDataSet('bubbles/velocity')[:]
    reader.close()
    # Create a H5-file reader
    reader = H5Reader(path / 'reconstructed_awcc_roc.h5')
    # Read the cont. velocity
    u_p_rec_flt = reader.getDataSet('bubbles/velocity')[:]
    reader.close()
    # Create a H5-file reader
    reader = H5Reader(path / 'reconstructed_awcc_roc_roc.h5')
    # Read the cont. velocity
    u_p_rec_flt_roc = reader.getDataSet('bubbles/velocity')[:]
    reader.close()


    # Histogram of bubble velocities
    var_names = ['x']
    limits = [10.5,15.5]
    col=[0.0,0.2,0.5]
    bin_width= 0.2
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
            figsize=(3.5,2))
    for jj,var_name in enumerate(var_names):
        (n_c, bins_c, patches_c) = axs.hist(np.clip(u_p_true[:,jj],limits[0],limits[1]), \
            bins=np.linspace(limits[0],limits[1],int((limits[1]-limits[0])/bin_width+1)), density=True, color='k', alpha=1.0, label='synthetic',\
            range=(limits[0],limits[1]),histtype='step')
        (n_c, bins_c, patches_c) = axs.hist(np.clip(u_p_rec_flt_roc[:,jj],limits[0],limits[1]), \
            bins=np.linspace(limits[0],limits[1],int((limits[1]-limits[0])/bin_width+1)), density=True, alpha=0.5, label='SPA+FLT+ROC',\
            range=(limits[0],limits[1]))
        (n_c, bins_c, patches_c) = axs.hist(np.clip(u_p_rec_flt[:,jj],limits[0],limits[1]), \
            bins=np.linspace(limits[0],limits[1],int((limits[1]-limits[0])/bin_width+1)), density=True, alpha=0.5, label='SPA+FLT',\
            range=(limits[0],limits[1]))
        (n_c, bins_c, patches_c) = axs.hist(np.clip(u_p_rec[:,jj],limits[0],limits[1]), \
            bins=np.linspace(limits[0],limits[1],int((limits[1]-limits[0])/bin_width+1)), density=True, alpha=0.5, label='SPA',\
            range=(limits[0],limits[1]))
        if jj == 0:
            axs.set_ylabel('Frequency [-]')
        axs.set_xlabel(f'Velocity $u_{var_name}$ [m/s]')
        axs.vlines(np.nanmedian(u_p_rec_flt[:,jj])+1.9835,0,max(n_c)*10, color='b', label='$\delta_{FLT}$')
        axs.vlines(np.nanmedian(u_p_rec_flt[:,jj])-1.9835,0,max(n_c)*10, color='b')
        axs.vlines(np.nanmedian(u_p_rec_flt_roc[:,jj])+0.6471,0,max(n_c)*10, color='r', label='$\delta_{ROC}$')
        axs.vlines(np.nanmedian(u_p_rec_flt_roc[:,jj])-0.6471,0,max(n_c)*10, color='r')
        axs.set_xlim(limits)
        axs.set_ylim([0,3.5])
        axs.text(limits[0]+0.1,3.2,'b)')
        axs.legend(loc=1,fontsize=7,ncol=1,frameon=False,fancybox=False,facecolor='w',edgecolor='k', bbox_to_anchor=(1.57,1.05))
        # if jj == 1:
        #     axs[jj].set_yscale('log')
        # if jj == 2:
        #     axs[jj].set_yscale('log')
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.7, top=0.95, wspace=0.3, hspace=None)
    fig.savefig(path / 'velocity_histrograms_shen.svg',dpi=300)
    fig.savefig(path / 'velocity_histrograms_shen.png',dpi=300)

if __name__ == "__main__":
    main()