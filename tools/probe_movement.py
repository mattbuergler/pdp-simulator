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

def main():
    """
        Main function of the the evaluation
    """
    # Start timer
    t_start = time.time()
    # Create parser to read in the configuration JSON-file to read from
    # the command line interface (CLI)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=str,
        help="The path to the scenario directory.")
    args = parser.parse_args()

    # Create Posix path for OS indepency
    path = pathlib.Path(args.path)
    config_file = path / 'config.json'
    # Read the configuation JSON-file
    config = json.loads(config_file.read_bytes())
    # Load the schema
    schema_file = pathlib.Path(sys.argv[0]).parents[0] / 'schemadef' / 'config_schema.json'
    schema = json.loads(schema_file.read_bytes())
    # Validate the configuration file
    jsonschema.validate(instance=config, schema=schema)
    # Get mean flow velocity and duration
    fp_vel = config['FLOW_PROPERTIES']['mean_velocity']
    fp_dur = config['FLOW_PROPERTIES']['duration']
    fp_ti = config['FLOW_PROPERTIES']['turbulent_intensity']
    n_sensors = len(config['PROBE']['sensors'])
    # Create a H5-file reader
    reader = H5Reader(path / 'reconstructed.h5')
    # Read the reconstructed velocity time series
    vel_rec = reader.getDataSet('bubbles/velocity')[:]
    # Read the mean velocity and weigthed mean velocity
    vel_mean_rec = reader.getDataSet('bubbles/mean_velocity')[:]
    vel_weighted_mean_rec = reader.getDataSet('bubbles/weighted_mean_velocity')[:]
    # Read the Reynolds stresses
    rs_rec = reader.getDataSet('bubbles/reynold_stresses')[:]
    # Read the turbulent intensity
    Ti_rec = reader.getDataSet('bubbles/turbulent_intensity')[:]
    # Read the reconstructed time bubbles
    t_rec = reader.getDataSet('bubbles/interaction_times')[:]
    # Read the bubble sizecd t
    b_size_rec = reader.getDataSet('bubbles/diameters')[:]
    reader.close()

    # Parse velocity time series
    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the 'true' continuous phase velocity time series
    vel_true_fluid = reader.getDataSet('fluid/velocity')[:]
    # Read the mean velocity
    vel_mean_true_fluid = reader.getDataSet('fluid/mean_velocity')[:]
    # Read the Reynolds stresses
    rs_true_fluid = reader.getDataSet('fluid/reynold_stresses')[:]
    # Read the turbulent intensity
    Ti_true_fluid = reader.getDataSet('fluid/turbulent_intensity')[:]
    # Read the 'true' velocity time series
    vel_true = reader.getDataSet('bubbles/velocity')[:]
    # Read the mean velocity
    vel_mean_true = reader.getDataSet('bubbles/mean_velocity')[:]
    # Read the Reynolds stresses
    rs_true = reader.getDataSet('bubbles/reynold_stresses')[:]
    # Read the turbulent intensity
    Ti_true = reader.getDataSet('bubbles/turbulent_intensity')[:]
    # Read the time vector
    t_true = reader.getDataSet('bubbles/time')[:]
    # Read the bubble size
    b_size = reader.getDataSet('bubbles/size')[:]
    # Read the time vector
    t_b_size = reader.getDataSet('bubbles/arrival_time')[:]
    if "UNCERTAINTY_QUANTIFICATION" in config:
        # Read the time vector
        t_probe = reader.getDataSet('probe/time')[:]
        # Read the probe location
        c_probe = reader.getDataSet('probe/location')[:]
        vib = config["UNCERTAINTY_QUANTIFICATION"]["VIBRATION"]
        vib_mags = vib['amplitudes']
        np.random.seed(42)
        rand = np.sort(np.random.uniform(0, len(t_rec), 3).astype('int'))
        # Plot velocity time series
        color = 'k'
        dt = t_rec[rand,1]-t_rec[rand,0]
        mult = 10
        lw = 1
        fig, axs = plt.subplots(3,figsize=(6,6))
        axs[0].plot(t_probe,c_probe[:,1],color=color,lw=lw)
        axs[1].plot(t_probe,c_probe[:,1],color=color,lw=lw)
        axs[2].plot(t_probe,c_probe[:,1],color=color,lw=lw)
        axs[0].set_ylabel('$y(t)$ [m]')
        axs[1].set_ylabel('$y(t)$ [m]')
        axs[2].set_ylabel('$y(t)$ [m]')
        axs[0].set_xlabel('$t$ [s]')
        axs[1].set_xlabel('$t$ [s]')
        axs[2].set_xlabel('$t$ [s]')
        axs[0].axvspan(t_rec[rand[0],0], t_rec[rand[0],1], alpha=0.5, color='red')
        axs[1].axvspan(t_rec[rand[1],0], t_rec[rand[1],1], alpha=0.5, color='red')
        axs[2].axvspan(t_rec[rand[2],0], t_rec[rand[2],1], alpha=0.5, color='red')
        axs[0].set_xlim([t_rec[rand[0],0]-mult*dt[0],t_rec[rand[0],1]+mult*dt[0]])
        axs[1].set_xlim([t_rec[rand[1],0]-mult*dt[1],t_rec[rand[1],1]+mult*dt[1]])
        axs[2].set_xlim([t_rec[rand[2],0]-mult*dt[2],t_rec[rand[2],1]+mult*dt[2]])
        axs[0].set_ylim([-vib_mags[0],vib_mags[0]])
        axs[1].set_ylim([-vib_mags[1],vib_mags[1]])
        axs[2].set_ylim([-vib_mags[1],vib_mags[1]])
        axs[0].grid(which='major', axis='both')
        axs[1].grid(which='major', axis='both')
        axs[2].grid(which='major', axis='both')
        axs[0].set_title(f'Bubble #{rand[0]}')
        axs[1].set_title(f'Bubble #{rand[1]}')
        axs[2].set_title(f'Bubble #{rand[2]}')
        plt.tight_layout()
        fig.savefig(path / 'vibrations.svg',dpi=300)

        length = 10000
        fig, axs = plt.subplots(3,figsize=(6,6))
        axs[0].plot(t_probe[0:length],c_probe[0:length,0],color=color,lw=lw)
        axs[1].plot(t_probe[0:length],c_probe[0:length,1],color=color,lw=lw)
        axs[2].plot(t_probe[0:length],c_probe[0:length,2],color=color,lw=lw)
        axs[0].set_ylabel('$x(t)$ [m]')
        axs[1].set_ylabel('$y(t)$ [m]')
        axs[2].set_ylabel('$z(t)$ [m]')
        axs[0].set_xlabel('$t$ [s]')
        axs[1].set_xlabel('$t$ [s]')
        axs[2].set_xlabel('$t$ [s]')
        # axs[0].set_xlim([t_rec[rand[0],0]-mult*dt[0],t_rec[rand[0],1]+mult*dt[0]])
        # axs[1].set_xlim([t_rec[rand[1],0]-mult*dt[1],t_rec[rand[1],1]+mult*dt[1]])
        # axs[2].set_xlim([t_rec[rand[2],0]-mult*dt[2],t_rec[rand[2],1]+mult*dt[2]])
        axs[0].set_ylim([-vib_mags[0],vib_mags[0]])
        axs[1].set_ylim([-vib_mags[1],vib_mags[1]])
        axs[2].set_ylim([-vib_mags[1],vib_mags[1]])
        axs[0].grid(which='major', axis='both')
        axs[1].grid(which='major', axis='both')
        axs[2].grid(which='major', axis='both')
        plt.tight_layout()
        fig.savefig(path / 'vibrations_xyz.svg',dpi=300)
        
    reader.close()

if __name__ == "__main__":
    main()