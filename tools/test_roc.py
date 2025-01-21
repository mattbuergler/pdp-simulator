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

    i_rep = np.zeros((N,3))    #which are to be replaced (by NaN)
    u_filt = np.zeros((N,3))
    for ii in range(0,N):
        if (abs((u[ii,:]-u_med)/ku_std) > 1.0).any():
            u_filt[ii,:] = np.nan
            i_rep[ii,:] = 1
        else:
            u_filt[ii,:] = u[ii,:]

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

    # Create the H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the bubbles velocity
    velocity_reconst = reader.getDataSet('bubbles/velocity')[:]
    reader.close()
    print(f'DEBUG: velocity_reconst = {velocity_reconst}')
    # data filtering
    # ROC filtering, R12 and SPR filtering implemented in previous loop
    print('Performing robust outlier cutoff.\n')
    while sum(sum(np.isnan(velocity_reconst))) < sum(sum(np.isnan(roc(velocity_reconst)))):
        velocity_reconst = roc(velocity_reconst)

    spikesloop=(sum(np.isnan(velocity_reconst)[:,0])/len(velocity_reconst))*100
    print(f'Discarded data: {spikesloop:.6f} %\n')

    time2 = time.time()
    print(f'Successfully run the robust outlier cutoff algorithm')
    print(f'Finished in {time2-time1:.2f} seconds\n')
    N = 10000
    velocity_reconst = np.zeros((N,3))
    velocity_reconst[:,0] = np.random.normal(loc=5, scale=0.5, size=N)
    velocity_reconst[:,1] = np.random.normal(loc=0, scale=0.25, size=N)
    velocity_reconst[:,2] = np.random.normal(loc=0, scale=0.25, size=N)

    while sum(sum(np.isnan(velocity_reconst))) < sum(sum(np.isnan(roc(velocity_reconst)))):
        velocity_reconst = roc(velocity_reconst)

    spikesloop=(sum(np.isnan(velocity_reconst)[:,0])/len(velocity_reconst))*100
    print(f'Discarded data: {spikesloop:.6f} %\n')


if __name__ == "__main__":
    main()

