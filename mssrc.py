#!/usr/bin/env python3

import sys
import argparse
import pathlib
import json
import jsonschema
import numpy as np
from numpy import linalg
import pandas as pd
import time
import math
import matplotlib as plt
import sys
import decimal
from decimal import *
try:
    from dataio.H5Writer import H5Writer
    from dataio.H5Reader import H5Reader
    from tools.globals import *
except ImportError:
    print("Error while importing modules")
    raise


"""
    Multi-Sensor Signal ReConstructor (MSSRC)

    The user-defined parameters are passed via the input_file (JSON).
    A time series is generated and saved to a file.

    Literature:
    Shen, X., Saito, Y., Mishima, K., & Nakamura, H. (2005). Methodological improvement of an intrusive four-sensor probe for the multi-dimensional two-phase flow measurement. International Journal of Multiphase Flow, 31(5), 593–617. https://doi.org/10.1016/j.ijmultiphaseflow.2005.02.003

    Shen, X., & Nakamura, H. (2014). Spherical-bubble-based four-sensor probe signal processing algorithm for two-phase flow measurement. International Journal of Multiphase Flow, 60, 11–29. https://doi.org/10.1016/j.ijmultiphaseflow.2013.11.010
"""

# Define global variables
COEFF_0 = Decimal(0.3)   # Eq. (43) in Shen et al. (2005): low limit constant

def inverse_den(x):
    """
    Calculate inverse of a number.
    """
    if abs(x) < Decimal(1.e-24):
        return Decimal(0.0)
    else:
        return Decimal(1.0) / x

def calc_det(mat):
    """
        Takes a matrix as and argument and returns the determinant

        In this function, the configuration JSON-File and the the signal data are parsed. The from the signal time series, the local velocities, the void fraction, the bubble diameters and the interfacial area density are recovered by a reconstruction algorithm.
    """
    a = mat[0][0]
    b = mat[0][1]
    c = mat[0][2]
    d = mat[1][0]
    e = mat[1][1]
    f = mat[1][2]
    g = mat[2][0]
    h = mat[2][1]
    i = mat[2][2]
    det = a*e*i + b*f*g + c*d*h - c*e*g - b*d*i - a*f*h
    return det

def main():
    """
        Main function of the Stochastic Bubble Generator (SBG)

        In this function, the configuration JSON-File and the the signal data are parsed. The from the signal time series, the local velocities, the void fraction, the bubble diameters and the interfacial area density are recovered by a reconstruction algorithm.
    """

    # Start timer
    time1 = time.time()
    # Create parser to read in the configuration JSON-file to read from
    # the command line interface (CLI)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=str,
        help="The path to the scenario directory.")
    parser.add_argument('-vel', '--velocity', default=1,
        help='A rough estimate for the mean velocity.')
    parser.add_argument('-p', '--progress', action='store_true',
        help='Show progress bar.')
    args = parser.parse_args()

    # Eq. (43) in Shen et al. (2005): estimated gas velocity
    V_GAS = Decimal(args.velocity)

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

    # Get sensors from config file
    # Eq. (1) Shen and Nakamura (2014): distance vectors S_0k from the central
    # front sensor tip (0) to any of the three peripheral rear sensor tips k
    S_0k = {}
    S_0k_mag = {}
    cos_eta_0k = {}
    sensor_ids = []
    aux_sensor_ids = []
    max_t_k = {}
    sensors = config['PROBE']['sensors']
    for sensor in sensors:
        # Get the sensor ID
        s_id = sensor['id']
        sensor_ids.append(s_id)
        if s_id != 0:
            aux_sensor_ids.append(s_id)
            # Set the distance vectors
            S_0k[s_id] = np.asarray(sensor['relative_location'],dtype='str')
            # Calculate magnitudes
            S_0k_mag[s_id] = (Decimal(S_0k[s_id][0]) * Decimal(S_0k[s_id][0])
                            + Decimal(S_0k[s_id][1]) * Decimal(S_0k[s_id][1])
                            + Decimal(S_0k[s_id][2]) * Decimal(S_0k[s_id][2])).sqrt()
            # Calculate cos of angles between vector S0k and (x,y,z)-axis
            tmp = []
            for ii in range(0,3):
                tmp.append(Decimal(S_0k[s_id][ii]) / S_0k_mag[s_id])
            cos_eta_0k[s_id] = tmp
            # Determine valid range for lag between signal of sensors 0
            # and signal of sensor k from Eq. (43) in Shen et al. (2005)
            max_t_k[s_id] = S_0k_mag[s_id] / (COEFF_0 * V_GAS)

    # Create a H5-file reader
    reader = H5Reader(path / 'binary_signal.h5')
    # Read the time vector
    t_signal_str = (reader.getDataSet('time')[:]).astype(str)
    t_signal = reader.getDataSet('time')[:]
    # Read the signal time series
    ds_signal = reader.getDataSet('signal')
    # Get the signal
    signal = np.array(ds_signal, dtype='int8')
    # Get the corresponding sensor ids
    sensor_ids = ds_signal.attrs['sensor_id']
    reader.close()
    # Create interface-detection (IFD) signal by first order difference
    signal_ifd = signal[1:len(signal)] - signal[0:(len(signal)-1)]
    # Get average time for each difference
    # Multiply and divide by the conjugate expression to avoid loss
    # of significance
    t_signal_ifd = t_signal[0:(len(t_signal)-1)] \
            + (t_signal[1:len(t_signal)]-t_signal[0:(len(t_signal)-1)])/2.0
    t_signal_ifd_list = []
    for ii in range(0,len(t_signal_ifd)):
        t_signal_ifd_list.append(str(Decimal(t_signal_str[ii]) \
            + (Decimal(t_signal_str[ii+1])-Decimal(t_signal_str[ii]))/Decimal(2.0)))

    # Interface-pairing signal-processing scheme
    # Shen et al. (2005)
    # Get the column number of the main sensor 0, just in case the order gets
    # messed up along the process
    id0 = np.where(sensor_ids == 0)[0][0]
    # Indices of the rising IFD signals for main sensor 0
    signal_ifd_rise_0 = np.where(signal_ifd[:,id0] > 0.0)[0]
    # Initialize lists to store complete and incomplete bubbles
    bubbles_complete = []
    bubbles_incomplete = []
    print('\nChecking for complete bubbles....\n')
    # Loop over rising signal fronts of main sensor 0 and do interface pairing
    for ii,idx_rise_0 in enumerate(signal_ifd_rise_0):
        # DataFrame with time of IFD of front (2h) and rear (2h+1) bubble
        # interface for each sensor
        ifd_times = pd.DataFrame(index=sensor_ids, columns=['t_2h','t_2h+1'])
        # Main Sensor 0
        # Write t_2h of sensor 0
        ifd_times['t_2h'][0] = t_signal_ifd_list[idx_rise_0]

        # Search for t_2h+1
        # Indices of the falling IFD signals for main sensor 0
        signal_ifd_fall_0 = np.where(signal_ifd[:,id0] < 0.0)[0]
        # Index of the falling IFD signal for main sensor 0
        idx_fall_0 = min(signal_ifd_fall_0[signal_ifd_fall_0 > idx_rise_0])
        # Write t_b+1 of sensor 0
        ifd_times['t_2h+1'][0] = t_signal_ifd_list[idx_fall_0]
        # Auxillary sensors k
        for k in aux_sensor_ids:
            idk = np.where(sensor_ids == k)[0][0]
            # Check if sensor is currently in air (1) or water (0) to determine
            # the search direction in time (backward or forward)
            phase = signal[idx_rise_0, idk]
            if phase == 0:
                # Sensor k is currently in water phase -> search forward
                # Rising IFD signal
                # Indices of the rising IFD signals for aux. sensor k
                signal_ifd_rise_k = np.where(signal_ifd[:,idk] > 0.0)[0]
                # Ahead of rising IFD signal of sensor 0
                signal_ifd_rise_k = signal_ifd_rise_k[signal_ifd_rise_k > idx_rise_0]
                # Index of the rising IFD signal for aux. sensor k closest to
                # idx_rise_0
                if len(signal_ifd_rise_k) > 0:
                    idx_rise_k = min(signal_ifd_rise_k)
                else:
                    # no rising IFD signal before t_2h of main sensor, NaN
                    idx_rise_k = np.nan

            elif phase == 1:
                # Sensor k is currently in air phase -> search backward
                # Rising IFD signal
                # Indices of the rising IFD signals for aux. sensor k
                signal_ifd_rise_k = np.where(signal_ifd[:,idk] > 0.0)[0]
                # Behind of rising IFD signal of sensor 0
                signal_ifd_rise_k = signal_ifd_rise_k[signal_ifd_rise_k <= idx_rise_0]

                # Index of the rising IFD signal for aux. sensor k closest to
                # idx_rise_0
                if len(signal_ifd_rise_k) > 0:
                    idx_rise_k = max(signal_ifd_rise_k)
                else:
                    # no rising IFD signal before t_2h of main sensor, NaN
                    idx_rise_k = np.nan
            # Search for t_2h+1 of sensor k
            if not np.isnan(idx_rise_k):
                # Indices of the falling IFD signals for main sensor k
                signal_ifd_fall_k = np.where(signal_ifd[:,idk] < 0.0)[0]
                signal_ifd_fall_k = signal_ifd_fall_k[signal_ifd_fall_k > idx_rise_k]
                # Index of the falling IFD signal for main sensor k
                idx_fall_k = min(signal_ifd_fall_k)
            else:
                idx_fall_k = np.nan

            # Store the time of IFD of front (2h) and rear (2h+1) bubble
            # interface for each sensor
            # Write t_2h of sensor k
            if not np.isnan(idx_rise_k):
                ifd_times['t_2h'][k] = t_signal_ifd_list[idx_rise_k]
            else:
                ifd_times['t_2h'][k] = np.nan
            # Write t_2h+1 of sensor k
            if not np.isnan(idx_fall_k):
                ifd_times['t_2h+1'][k] = t_signal_ifd_list[idx_fall_k]
            else:
                ifd_times['t_2h+1'][k] = np.nan

        # Check for each sensor if IFD times of front (2h) and rear (2h+1)
        # bubble fall within the range for the lag between signal of sensors 0
        # and signal of sensor k from Eq. (42) in Shen et al. (2005)
        for k in aux_sensor_ids:
            # first interface (2h)
            lag_2h = abs(Decimal(ifd_times['t_2h'][0]) - Decimal(ifd_times['t_2h'][k]))
            if lag_2h.compare(max_t_k[k]) == 1:
                # lag too large
                ifd_times['t_2h'][k] = np.nan
            # second interface (2h+1)
            lag_2hp1 = abs(Decimal(ifd_times['t_2h+1'][0]) - Decimal(ifd_times['t_2h+1'][k]))
            if lag_2hp1.compare(max_t_k[k]) == 1:
                ifd_times['t_2h+1'][k] = np.nan

        # Store the interface detection times for complete and
        # incomplete bubbles
        bubble_props = {'ifd_times':ifd_times}

        if ifd_times.isnull().values.any():
            bubbles_incomplete.append(bubble_props)
        else:
            bubbles_complete.append(bubble_props)
        # Display progress
        if args.progress:
            printProgressBar(ii, len(signal_ifd_rise_0), prefix = 'Progress:', suffix = 'Complete', length = 50)

    print(f"\nDetected {len(bubbles_complete)} complete bubble signals.")
    print(f"Detected {len(bubbles_incomplete)} incomplete bubble signals.\n")
    print('\nRunning reconstruction algorithm....\n')
    # Initialize the Interfacial Area Concentration (IAC)
    iac = Decimal(0.0)
    # Loop over complete bubbles
    for ii,bubble_props in enumerate(bubbles_complete):
        ifd_times = bubble_props['ifd_times']
        # Time differences between sensors 0 and k for interfaces 2h and 2h+1
        delta_t_0k_2h = {}
        delta_t_0k_2hp1 = {}
        # Time differences k=1,2,3,4 for interfaces 2h and 2h+1
        delta_t_k_h = {}
        delta_t_k_h[0] = Decimal(ifd_times['t_2h+1'][0]) - Decimal(ifd_times['t_2h'][0])
        # 1. Reconstruct instantaneous local velocity
        # Inverse of measurable displacement velocity of the h-th bubble
        # for sensor 0 and k
        V_m0k_h_inv = {}
        E_0k_2h = {}
        E_0k_2hp1 = {}
        A_0 = []
        A_01_h = []
        A_02_h = []
        A_03_h = []
        B_01_2h = []
        B_02_2h = []
        B_03_2h = []
        B_01_2hp1 = []
        B_02_2hp1 = []
        B_03_2hp1 = []

        for k in aux_sensor_ids:
            # Time differences
            # Eq. (2) from Shen and Nakamura (2014)
            delta_t_0k_2h[k] = Decimal(ifd_times['t_2h'][k]) \
                             - Decimal(ifd_times['t_2h'][0])
            delta_t_0k_2hp1[k] = Decimal(ifd_times['t_2h+1'][k]) \
                               - Decimal(ifd_times['t_2h+1'][0])
            # Eq. (3) from Shen and Nakamura (2014)
            delta_t_k_h[k] = Decimal(ifd_times['t_2h+1'][k]) \
                           - Decimal(ifd_times['t_2h'][k])
            # Measurable displacement velocity
            # Eq. (30) from Shen and Nakamura (2014)
            V_m0k_h_inv[k] = (delta_t_0k_2h[k] + delta_t_0k_2hp1[k]) \
                        / (Decimal(2.0) * Decimal(S_0k_mag[k]))
            A_0.append(cos_eta_0k[k])
            A_01_h.append(np.array([V_m0k_h_inv[k], cos_eta_0k[k][1], cos_eta_0k[k][2]]))
            A_02_h.append(np.array([cos_eta_0k[k][0], V_m0k_h_inv[k], cos_eta_0k[k][2]]))
            A_03_h.append(np.array([cos_eta_0k[k][0], cos_eta_0k[k][1], V_m0k_h_inv[k]]))

        # Calculate matrix determinants
        # Eq. (32) to (35) from Shen and Nakamura (2014)
        A_0_det = calc_det(A_0)
        A_01_h_det = calc_det(A_01_h)
        A_02_h_det = calc_det(A_02_h)
        A_03_h_det = calc_det(A_03_h)
        # Calculate instantaneous velocity magnitude and components
        dummy = (A_01_h_det* A_01_h_det + A_02_h_det*A_02_h_det + A_03_h_det*A_03_h_det).sqrt()
        # Eq. (37) from Shen and Nakamura (2014)
        V_bh_mag =  abs(A_0_det) * inverse_den(dummy)
        # Eq. (36) from Shen and Nakamura (2014)
        V_bh_x = V_bh_mag**2 * A_01_h_det * inverse_den(A_0_det)
        V_bh_y = V_bh_mag**2 * A_02_h_det * inverse_den(A_0_det)
        V_bh_z = V_bh_mag**2 * A_03_h_det * inverse_den(A_0_det)
        bubble_props['velocity'] = np.array([V_bh_x, V_bh_y, V_bh_z])

        # 2. Reconstruct bubble diameter and interfacial normal unit vector
        # measurement
        E_0k_2h = {}
        E_0k_2hp1 = {}
        B_01_2h = []
        B_02_2h = []
        B_03_2h = []
        B_01_2hp1 = []
        B_02_2hp1 = []
        B_03_2hp1 = []
        for k in aux_sensor_ids:
            # Chord lengths of interfaces 2h and 2h+1
            # Eq. (43) and (44) in Shen and Nakamura (2014)
            E_0k_2h[k] = V_bh_mag*V_bh_mag * delta_t_0k_2h[k] \
                        * (delta_t_0k_2hp1[k] + delta_t_k_h[0]) / S_0k_mag[k] \
                        - S_0k_mag[k]
            E_0k_2hp1[k] = V_bh_mag*V_bh_mag * delta_t_0k_2hp1[k] \
                        * (delta_t_0k_2h[k] - delta_t_k_h[0]) / S_0k_mag[k] \
                        - S_0k_mag[k]
            # Eq. (46) and (48) in Shen and Nakamura (2014)
            B_01_2h.append(np.array([E_0k_2h[k], cos_eta_0k[k][1], cos_eta_0k[k][2]]))
            B_02_2h.append(np.array([cos_eta_0k[k][0], E_0k_2h[k], cos_eta_0k[k][2]]))
            B_03_2h.append(np.array([cos_eta_0k[k][0], cos_eta_0k[k][1], E_0k_2h[k]]))
            B_01_2hp1.append(np.array([E_0k_2hp1[k], cos_eta_0k[k][1], cos_eta_0k[k][2]]))
            B_02_2hp1.append(np.array([cos_eta_0k[k][0], E_0k_2hp1[k], cos_eta_0k[k][2]]))
            B_03_2hp1.append(np.array([cos_eta_0k[k][0], cos_eta_0k[k][1], E_0k_2hp1[k]]))
        # Calculate matrix determinants
        # Eq. (46) and (48) in Shen and Nakamura (2014)
        B_01_2h_det = calc_det(B_01_2h)
        B_02_2h_det = calc_det(B_02_2h)
        B_03_2h_det = calc_det(B_03_2h)
        B_01_2hp1_det = calc_det(B_01_2hp1)
        B_02_2hp1_det = calc_det(B_02_2hp1)
        B_03_2hp1_det = calc_det(B_03_2hp1)

        # Calculate the bubble diameter
        dummy_2h = (B_01_2h_det*B_01_2h_det \
                    + B_02_2h_det*B_02_2h_det \
                    + B_03_2h_det*B_03_2h_det).sqrt()
        dummy_2hp1 = (B_01_2hp1_det*B_01_2hp1_det \
                    + B_02_2hp1_det*B_02_2hp1_det \
                    + B_03_2hp1_det*B_03_2hp1_det).sqrt()
        # Eq. (50) from Shen and Nakamura (2014)
        D_h_2h = dummy_2h / A_0_det
        D_h_2hp1 = dummy_2hp1 / A_0_det
        bubble_props['diameter'] = np.array([D_h_2h, D_h_2hp1])

        # Calculate interfacial normal unit vectors
        # Eq. (51) in Shen and Nakamura (2014)
        cos_eta_i_2h = np.array([Decimal(1.0), Decimal(1.0), Decimal(1.0)]) * A_0_det \
                         * inverse_den(abs(A_0_det)) \
                         * inverse_den(dummy_2h)
        cos_eta_i_2h = cos_eta_i_2h \
                    * np.array([B_01_2h_det, B_02_2h_det, B_03_2h_det])
        cos_eta_i_2hp1 = np.array([Decimal(1.0), Decimal(1.0), Decimal(1.0)]) * A_0_det \
                        * inverse_den(abs(A_0_det)) \
                        * inverse_den(dummy_2hp1)
        cos_eta_i_2hp1 = cos_eta_i_2hp1 \
                    * np.array([B_01_2hp1_det, B_02_2hp1_det, B_03_2hp1_det])

        bubble_props['if_norm_unit_vecs'] = [cos_eta_i_2h, cos_eta_i_2hp1]

        # Calculate the local Interfacial Area Concentration (IAC)
        # Eq. (54) in Shen and Nakamura (2014)
        # Sampling duration Omega
        Omega = Decimal(t_signal[-1]) - Decimal(t_signal[0])
        dummy_A = A_01_h_det*A_01_h_det + A_02_h_det*A_02_h_det + A_03_h_det*A_03_h_det
        dummy_B = (B_01_2h_det*B_01_2h_det \
                    + B_02_2h_det*B_02_2h_det \
                    + B_03_2h_det*B_03_2h_det).sqrt()
        dummy_AB = abs(A_0_det * (A_01_h_det*B_01_2h_det \
                                + A_02_h_det*B_02_2h_det \
                                + A_03_h_det*B_03_2h_det))
        iac_h = dummy_A * dummy_B * inverse_den(dummy_AB)
        iac += Decimal(2.0) / Omega * iac_h
        # Display progress
        if args.progress:
            printProgressBar(ii, len(bubbles_complete), prefix = 'Progress:', suffix = 'Complete', length = 50)

    print('\nSaving results....\n')
    # Generate a reconstructed velocity time-series from individual
    # Initialize bubble velocities, weighted mean velocity
    velocity_reconst = np.empty((len(bubbles_complete),3))
    weighted_mean_velocity_reconst = np.empty(3)
    time_reconst = np.empty((len(bubbles_complete),2))
    # Collect bubble properties, such as diameter
    bubble_diam_reconst = np.empty((len(bubbles_complete),2))
    for ii,bubble_props in enumerate(bubbles_complete):
        velocity_reconst[ii,:] = np.array([bubble_props['velocity'][0], \
                                           bubble_props['velocity'][1], \
                                           bubble_props['velocity'][2]
                                          ])
        time_reconst[ii,0] = np.min(bubble_props['ifd_times'].to_numpy().astype('float64'))
        time_reconst[ii,1] = np.max(bubble_props['ifd_times'].to_numpy().astype('float64'))
        bubble_diam_reconst[ii,:] = bubble_props['diameter']
    weighted_mean_velocity_reconst[0] = np.sum(velocity_reconst[:,0] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.sum(time_reconst[:,1]-time_reconst[:,0],axis=0)
    weighted_mean_velocity_reconst[1] = np.sum(velocity_reconst[:,1] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.sum(time_reconst[:,1]-time_reconst[:,0],axis=0)
    weighted_mean_velocity_reconst[2] = np.sum(velocity_reconst[:,2] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.sum(time_reconst[:,1]-time_reconst[:,0],axis=0)
    # Calculate mean velocity
    mean_velocity_reconst = velocity_reconst.mean(axis=0)
    # Initialize the reynolds stress tensors time series
    reynolds_stress = np.empty((len(velocity_reconst),3,3))
    for ii in range(0,len(velocity_reconst)):
        # Calculate velocity fluctuations
        velocity_fluct_reconst = velocity_reconst[ii,:] - mean_velocity_reconst
        # Reynolds stresses as outer product of fluctuations
        reynolds_stress[ii,:,:] = np.outer(velocity_fluct_reconst, \
                                    velocity_fluct_reconst)
    # Calculate mean Reynolds stresses
    mean_reynolds_stress = reynolds_stress.mean(axis=0)
    # Calculate turbulent intensity with regard to mean x-velocity
    turbulent_intensity = np.sqrt(np.array([
            mean_reynolds_stress[0,0], \
            mean_reynolds_stress[1,1], \
            mean_reynolds_stress[2,2], \
            ])) / np.sqrt(mean_velocity_reconst.dot(mean_velocity_reconst))
    # Create the H5-file writer
    writer = H5Writer(path / 'reconstructed.h5', 'w')
    # Create the velocity data set
    writer.writeDataSet('bubbles/velocity', velocity_reconst, 'float64')
    writer.writeDataSet('bubbles/interaction_times', time_reconst, 'float64')
    writer.writeDataSet('bubbles/mean_velocity', \
        mean_velocity_reconst, 'float64')
    writer.writeDataSet('bubbles/weighted_mean_velocity', \
        weighted_mean_velocity_reconst, 'float64')
    writer.writeDataSet('bubbles/reynold_stresses', \
        mean_reynolds_stress, 'float64')
    writer.writeDataSet('bubbles/turbulent_intensity', \
        turbulent_intensity, 'float64')
    ds_vel = writer.getDataSet('bubbles/velocity')
    # Add the attributes
    ds_vel.attrs['labels'] = ['Ux','Uy','Uz']
    # Create the dataset for the bubble diameter
    writer.writeDataSet('bubbles/diameters', bubble_diam_reconst, 'float64')
    ds_d = writer.getDataSet('bubbles/diameters')
    # Add the attributes
    ds_d.attrs['labels'] = ['D_h_2h', 'D_h_2hp1']
    # Create the IAC and void_fraction datasets
    writer.writeDataSet('IAC', np.array([iac],dtype='float64'), 'float64')
    writer.writeDataSet('voidFraction', np.array([np.mean(signal)]), 'float64')
    writer.close()
    time2 = time.time()
    print(f'Successfully run the reconstruction algorithm')
    print(f'Finished in {time2-time1} seconds\n')

if __name__ == "__main__":
    main()