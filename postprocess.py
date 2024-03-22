#!/usr/bin/env python3

"""
    Filename: postprocess.py
    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
    Date created: January 1, 2024
    Description:

    Function to post-process a reconstructed pseudo-instantaneous
    velocity timeseries by filtering.

"""

# (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero,
# Benjamin Hohermuth, David F. Vetsch, Robert M. Boes,
# D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)
# This software is released under the the GNU General Public License v3.0.
# https://https://opensource.org/license/gpl-3-0

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
    u_med = np.nanmedian(u[:,0],axis=0)
    # ust estimated through MED
    u_std = k * np.nanmedian(abs(u[:,0] - u_med),axis=0)
    # universal threshold:
    # take only the number of non-nan values
    N = len(u[~np.isnan(u[:,0])])
    lambda_u = math.sqrt(2*math.log(N))
    ku_std = lambda_u*u_std
    u_filt = np.zeros((len(u),3))
    for ii in range(0,len(u)):
        if (abs((u[ii,0]-u_med)/ku_std) > 1.0):
            u_filt[ii,:] = np.nan
        else:
            u_filt[ii,:] = u[ii,:]
    return u_filt

def cutoff_roc(u):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
    """

    k = 1.483; # based on a normal distro, see Rousseeuw and Croux (1993)

    # robust estimation of the variance:
    # expected value estimated through MED
    u_med = np.nanmedian(u[:,0],axis=0)
    # ust estimated through MED
    u_std = k * np.nanmedian(abs(u[:,0] - u_med),axis=0)
    # universal threshold:
    # take only the number of non-nan values
    N = len(u[~np.isnan(u[:,0])])
    lambda_u = math.sqrt(2*math.log(N))
    ku_std = lambda_u*u_std
    return ku_std

def filter_awcc_roc(u, u_mean, u_rms):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
        u_mean:             given mean value
        u_rms:              given root mean square deviation

    """

    u_std = u_rms
    # universal threshold:
    N = len(u[~np.isnan(u[:,0])])
    lambda_u = math.sqrt(2*math.log(N))
    ku_std = lambda_u*u_std
    u_filt = np.zeros((len(u),3))
    for ii in range(0,len(u)):
        if (abs((u[ii,0]-u_mean)/ku_std) > 1.0):
            u_filt[ii,:] = np.nan
        else:
            u_filt[ii,:] = u[ii,:]

    return u_filt

def cutoff_awcc_roc(u, u_mean, u_rms):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
        u_mean:             given mean value
        u_rms:              given root mean square deviation

    """

    u_std = u_rms
    # universal threshold:
    N = len(u[~np.isnan(u[:,0])])
    lambda_u = math.sqrt(2*math.log(N))
    ku_std = lambda_u*u_std
    return ku_std

def filter_max_roc(u, u_mean):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
    """

    k = 1.483; # based on a normal distro, see Rousseeuw and Croux (1993)

    # robust estimation of the variance:
    # expected value estimated through MED
    u_med = u_mean
    # ust estimated through MED
    u_std = k * np.nanmedian(abs(u[u[:,0] > u_mean][:,0] - u_med),axis=0)
    # universal threshold:
    N = len(u[~np.isnan(u[:,0])])
    lambda_u = math.sqrt(2*math.log(N))
    ku_std = lambda_u*u_std

    u_filt = np.zeros((len(u),3))
    for ii in range(0,len(u)):
        if (abs((u[ii,0]-u_med)/ku_std) > 1.0):
            u_filt[ii,:] = np.nan
        else:
            u_filt[ii,:] = u[ii,:]
    return u_filt
    
def cutoff_max_roc(u, u_mean):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
    """

    k = 1.483; # based on a normal distro, see Rousseeuw and Croux (1993)

    # robust estimation of the variance:
    # expected value estimated through MED
    u_med = u_mean
    # ust estimated through MED
    u_std = k * np.nanmedian(abs(u[u[:,0] > u_mean][:,0] - u_med),axis=0)
    # universal threshold:
    N = len(u[~np.isnan(u[:,0])])
    lambda_u = math.sqrt(2*math.log(N))
    ku_std = lambda_u*u_std
    return ku_std

def filter_max(u, u_mean):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
        u_mean:             given mean value
        u_rms:              given root mean square deviation

    """
    u_pre_filter = u[u[:,0] < np.nanpercentile(u[:,0],99)]
    u_std = np.sqrt(np.sum(np.square(u_pre_filter[u_pre_filter[:,0] > u_mean][:,0] - u_mean))/len(u_pre_filter[u_pre_filter[:,0] > u_mean][:,0]))
    # universal threshold:
    N = len(u)

    u_filt = np.zeros((N,3))
    for ii in range(0,N):
        if (abs((u[ii,0]-u_mean)/u_std) > 1.0):
            u_filt[ii,:] = np.nan
        else:
            u_filt[ii,:] = u[ii,:]

    return u_filt

def filter_max_med(u, u_mean):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
        u_mean:             given mean value
        u_rms:              given root mean square deviation

    """
    u_std = np.nanmedian(u[u[:,0] > u_mean][:,0] - u_mean,axis=0)
    # universal threshold:
    N = len(u)

    u_filt = np.zeros((N,3))
    for ii in range(0,N):
        if (abs((u[ii,0]-u_mean)/u_std) > 1.0):
            u_filt[ii,:] = np.nan
        else:
            u_filt[ii,:] = u[ii,:]

    return u_filt

def filter(u, u_med, max_dev):
    """
        Filtering based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
        u_med:              expected value
    """

    # ust estimated through MED
    u_std = (max_dev/100.0)*u_med
    N = len(u)
    u_filt = np.zeros((N,3))
    for ii in range(0,N):
        if (abs((u[ii,0]-u_med)/u_std) > 1.0):
            u_filt[ii,:] = np.nan
        else:
            u_filt[ii,:] = u[ii,:]

    return u_filt


def main():
    # Start timer
    time1 = time.time()
    # Create parser to read in the configuration JSON-file to read from
    # the command line interface (CLI)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=str,
        help="The path to the scenario directory.")
    parser.add_argument('-ft', '--filter-type', metavar="TYPE", default='None',
        help="the following options are available:\n"
        + "awcc_roc   - based on mean and RMS velocity from AWCC\n"
        + "max_roc    - ROC based on mean from awcc and median deviation of velocity time series values larger than the mean.\n"
        + "max        - based on mean from awcc and standard deviation of velocity time series values larger than the mean.\n"
        + "max_med    - based on mean from awcc and median deviation of velocity time series values larger than the mean.\n"
        + "none       - no filtering.\n")
    parser.add_argument('-roc', '--ROC', default='False',
        help='Perform robust outlier cutoff (ROC) based on the maximum' + 
                'absolute deviation and the universal threshold (True/False).')
    args = parser.parse_args()

    # Create Posix path for OS indepency
    path = pathlib.Path(args.path)

    print('\nRunning post-processing\n')

    # Create the H5-file writer
    writer = H5Writer(path / 'reconstructed.h5', 'a')
    # Read the time vector
    time_reconst = writer.getDataSet('bubbles/interaction_times')[:]
    # Read the bubbles velocity
    velocity_reconst = writer.getDataSet('bubbles/velocity')[:]
    # Read the bubbles diameters
    bubble_diam_reconst = writer.getDataSet('bubbles/diameters')[:]
    # Read the mean velocity from awcc
    mean_vel_awcc = writer.getDataSet('bubbles/mean_velocity_awcc')[:][0]
    # calculate u_rms from awcc
    u_rms_awcc = math.sqrt(writer.getDataSet('bubbles/reynold_stresses_awcc')[:][0,0])

    # Do the actual post-processing here
    # data filtering
    if args.filter_type == 'awcc_roc':
        while sum(sum(np.isnan(velocity_reconst))) < sum(sum(np.isnan(filter_awcc_roc(velocity_reconst, mean_vel_awcc, u_rms_awcc)))):
            velocity_reconst = filter_awcc_roc(velocity_reconst, mean_vel_awcc, u_rms_awcc)
        cutoff = cutoff_awcc_roc(velocity_reconst, mean_vel_awcc, u_rms_awcc)
        print(f"\nRun filtering based on mean velocity of {mean_vel_awcc:.4f} m/s and u_rms of {u_rms_awcc:.4f} m/s. Cutoff is {cutoff:.4f} m/s\n")
    elif args.filter_type == 'max_roc':
        while sum(sum(np.isnan(velocity_reconst))) < sum(sum(np.isnan(filter_max_roc(velocity_reconst, mean_vel_awcc)))):
            velocity_reconst = filter_max_roc(velocity_reconst, mean_vel_awcc)
        cutoff = cutoff_max_roc(velocity_reconst, mean_vel_awcc)
        print(f"\nRun filtering 'filter_max_roc' based on mean velocity of {mean_vel_awcc:.4f} m/s and cutoff of {cutoff:.4f} m/s\n")
    elif args.filter_type == 'max':
        print(f"\nRun filtering 'filter_max' based on mean velocity of {mean_vel_awcc:.4f} m/s\n")
        velocity_reconst = filter_max(velocity_reconst, mean_vel_awcc)
    elif args.filter_type == 'max_med':
        print(f"\nRun filtering 'filter_max_med' based on mean velocity of {mean_vel_awcc:.4f} m/s\n")
        velocity_reconst = filter_max_med(velocity_reconst, mean_vel_awcc)
    elif args.filter_type == 'none':
        print('No filtering.')
    else:
        print(f"Filter of type <{args.filter_type}> does not exist.")
        sys.exit()
    discarded=(sum(sum(np.isnan(velocity_reconst)))/(len(velocity_reconst)*3))*100
    print(f'Discarded data: {discarded:.2f} %\n')

    if args.ROC == 'True':
        # ROC filtering
        print('Performing robust outlier cutoff.\n')
        while sum(sum(np.isnan(velocity_reconst))) < sum(sum(np.isnan(roc(velocity_reconst)))):
            velocity_reconst = roc(velocity_reconst)
        cutoff = cutoff_roc(velocity_reconst)
        discarded=(sum(sum(np.isnan(velocity_reconst)))/(len(velocity_reconst)*3))*100
        print(f'Discarded data: {discarded:.2f} %, with a cutoff of {cutoff:.4f} m/s.\n')


    # Re-calculate the other properties.
    # Generate a reconstructed velocity time-series from individual
    # Initialize bubble velocities, weighted mean velocity
    weighted_mean_velocity_reconst = np.empty(3)
    time_deltas = (time_reconst[:,1]-time_reconst[:,0])
    weighted_mean_velocity_reconst[0] = np.nansum(velocity_reconst[:,0] * time_deltas,axis=0) \
                                    / np.nansum(time_deltas[~np.isnan(velocity_reconst[:,0])],axis=0)
    weighted_mean_velocity_reconst[1] = np.nansum(velocity_reconst[:,1] * time_deltas,axis=0) \
                                    / np.nansum(time_deltas[~np.isnan(velocity_reconst[:,1])],axis=0)
    weighted_mean_velocity_reconst[2] = np.nansum(velocity_reconst[:,2] * time_deltas,axis=0) \
                                    / np.nansum(time_deltas[~np.isnan(velocity_reconst[:,2])],axis=0)
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

    # set diameters of bubbles with no velocity to nan
    bubble_diam_reconst[np.isnan(velocity_reconst[:,0]),0] = np.nan
    bubble_diam_reconst[np.isnan(velocity_reconst[:,0]),1] = np.nan

    # Create the velocity data set
    writer.write2DataSet('bubbles/velocity', velocity_reconst, 0, 0)
    writer.write2DataSet('bubbles/mean_velocity', \
        mean_velocity_reconst, 0, 0)
    writer.write2DataSet('bubbles/weighted_mean_velocity', \
        weighted_mean_velocity_reconst, 0, 0)
    #writer.write2DataSet('bubbles/mean_vel_awcc', np.array([mean_vel_awcc]), 0, 0)
    writer.writeDataSet('bubbles/mean_vel_awcc', np.array([mean_vel_awcc]), 'float64')

    writer.write2DataSet('bubbles/reynold_stresses', \
        mean_reynolds_stress, 0, 0)
    writer.write2DataSet('bubbles/turbulent_intensity', \
        turbulent_intensity, 0, 0)
    writer.write2DataSet('bubbles/diameters', bubble_diam_reconst, 0, 0)
    writer.write2DataSet('bubbles/data_rate', np.array([len(velocity_reconst[~np.isnan(velocity_reconst[:,0])])/(time_reconst[-1,1]-time_reconst[0,0])]), 0, 0)

    writer.close()
    time2 = time.time()
    print(f'Successfully run the post-processing')
    print(f'Finished in {time2-time1:.2f} seconds\n')

if __name__ == "__main__":
    main()