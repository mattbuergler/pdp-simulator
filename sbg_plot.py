#!/usr/bin/env python3

import argparse
import pathlib
import json
import jsonschema
import numpy as np
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt

try:
    from dataio.H5Reader import H5Reader
    import sbg_functions
except ImportError:
    print("")
    raise


"""
    Stochastic Bubble Generator (SBG)

    The user-defined parameters are passed via the input_file (JSON).
    A time series is generated and saved to a file.
"""

def plot_results(path, flow_properties):
    # Plot parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['xtick.major.pad']='6'
    plt.rcParams['ytick.major.pad']='6'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams["figure.figsize"] = (6,6)


    # Parse velocity time series
    vel_path = path / 'flow_data.h5'
    if vel_path.is_file():
        # Create a H5-file reader
        reader = H5Reader(vel_path)
        # Read the time vector
        vel = reader.getDataSet('fluid/velocity')[:]
        # Read the trajectory
        t_vel = reader.getDataSet('fluid/time')[:]
        reader.close()



        # Plot velocity time series
        color = [0.2,0.2,0.2]
        lw = 0.2
        fig, axs = plt.subplots(3,figsize=(6,4))
        axs[0].plot(t_vel,vel[:,0],color=color,lw=lw)
        axs[1].plot(t_vel,vel[:,1],color=color,lw=lw)
        axs[2].plot(t_vel,vel[:,2],color=color,lw=lw)
        axs[0].set_ylabel('$u_x$ [m/s]')
        axs[1].set_ylabel('$u_y$ [m/s]')
        axs[2].set_ylabel('$u_z$ [m/s]')
        axs[2].set_xlabel('time $t$ [s]')
        axs[0].set_xlim([0,flow_properties['duration']])
        axs[1].set_xlim([0,flow_properties['duration']])
        axs[2].set_xlim([0,flow_properties['duration']])
        plt.tight_layout()
        fig.savefig(path / 'velocity.svg',dpi=300)
