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
import matplotlib as plt
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


"""
    Robust outlier cutoff based on the maximum absolute deviation and the 
    universal threshold."""

def inverse_den(x):
    """
    Calculate inverse of a number.
    """
    if abs(x) < Decimal(1.e-24):
        return Decimal(0.0)
    else:
        return Decimal(1.0) / x

def roc(u):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
    """

    k = 1.483; # based on a normal distro, see Rousseeuw and Croux (1993)

    # robust estimation of the variance:
    # expected value estimated through MED
    u_med = np.nanmedian(u,axis=0)
    # ust estimated through MED
    u_std = k * np.nanmedian(abs(u - u_med),axis=0)
    # universal threshold:
    N = len(u)
    lambda_u = math.sqrt(2*math.log(N))
    ku_std = lambda_u*u_std
    ku_std[ku_std == 0.0] = np.nan

    u_filt = u
    for ii in range(0,N):
        u_filt[ii,(abs((u[ii,:]-u_med)/ku_std) > 1.0)] = np.nan


    return u_filt


def main():
    """
        Main function of the Robust Outlier Cutoff (ROC)

        In this function, the configuration JSON-File and the the signal data
        are parsed. The robut outlier cutoff function is called.
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

    # data filtering
    # set negative values to NaN
    velocity_reconst[velocity_reconst < 0.0] = np.nan
    # ROC filtering, R12 and SPR filtering implemented in previous loop
    print('Performing robust outlier cutoff.\n')
    while sum(sum(np.isnan(velocity_reconst))) < sum(sum(np.isnan(roc(velocity_reconst)))):
        velocity_reconst = roc(velocity_reconst)

    spikesloop=(sum(sum(np.isnan(velocity_reconst)))/(len(velocity_reconst)*3))*100
    print(f'Discarded data: {spikesloop:.6f} %\n')
    # Re-calculate the other properties.
    # Generate a reconstructed velocity time-series from individual
    # Initialize bubble velocities, weighted mean velocity
    weighted_mean_velocity_reconst = np.empty(3)
    weighted_mean_velocity_reconst[0] = np.nansum(velocity_reconst[:,0] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.nansum(time_reconst[:,1]-time_reconst[:,0],axis=0)
    weighted_mean_velocity_reconst[1] = np.nansum(velocity_reconst[:,1] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.nansum(time_reconst[:,1]-time_reconst[:,0],axis=0)
    weighted_mean_velocity_reconst[2] = np.nansum(velocity_reconst[:,2] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.nansum(time_reconst[:,1]-time_reconst[:,0],axis=0)
    # Calculate mean velocity
    mean_velocity_reconst = np.nanmean(velocity_reconst, axis=0)
    # Initialize the reynolds stress tensors time series
    reynolds_stress = np.empty((len(velocity_reconst),3,3))
    for ii in range(0,len(velocity_reconst)):
        # Calculate velocity fluctuations
        velocity_fluct_reconst = velocity_reconst[ii,:] - mean_velocity_reconst
        # Reynolds stresses as outer product of fluctuations
        reynolds_stress[ii,:,:] = np.outer(velocity_fluct_reconst, \
                                    velocity_fluct_reconst)
    # Calculate mean Reynolds stresses
    mean_reynolds_stress = np.nanmean(reynolds_stress,axis=0)
    # Calculate turbulent intensity with regard to mean x-velocity
    turbulent_intensity = np.sqrt(np.array([
            mean_reynolds_stress[0,0], \
            mean_reynolds_stress[1,1], \
            mean_reynolds_stress[2,2], \
            ])) / np.sqrt(mean_velocity_reconst.dot(mean_velocity_reconst))

    # Create the velocity data set
    writer.write2DataSet('bubbles/velocity', velocity_reconst, 0, 0)
    writer.write2DataSet('bubbles/mean_velocity', \
        mean_velocity_reconst, 0, 0)
    writer.write2DataSet('bubbles/weighted_mean_velocity', \
        weighted_mean_velocity_reconst, 0, 0)
    writer.write2DataSet('bubbles/reynold_stresses', \
        mean_reynolds_stress, 0, 0)
    writer.write2DataSet('bubbles/turbulent_intensity', \
        turbulent_intensity, 0, 0)
    writer.close()
    time2 = time.time()
    print(f'Successfully run the robust outlier cutoff algorithm')
    print(f'Finished in {time2-time1:.2f} seconds\n')

if __name__ == "__main__":
    main()