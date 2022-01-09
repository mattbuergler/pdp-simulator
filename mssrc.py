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
import matplotlib.pyplot as plt
import sys
import decimal
from scipy import signal
from decimal import *
from itertools import combinations
from joblib import Parallel, delayed

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
V_GAS = Decimal(0.0)     # Eq. (43) in Shen et al. (2005): dummy value, calculated later


def inverse_den(x):
    """
    Calculate inverse of a number.
    """
    if x.is_nan():
        return Decimal('nan')
    else:
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

def chord(signal, duration, progress=False):
    """
        Calculation of chord lengths and bubble/droplet count rate

        signal:             binary signal of the leading tip
        duration:           sampling duration
    """


    chord_water = []
    chord_air = []
    length_air = 1
    length_water = 1
    # scanning signal
    for ii in range(0,len(signal)-1):
        if (signal[ii] == 1):               # Tip is in the air...
            if (signal[ii+1] == 1):         # and will keep on being in the air
                length_air = length_air + 1
                if (ii == len(signal)-2):   # last chord
                    chord_air.append(length_air)
            elif (signal[ii+1] == 0):       # and gets wet in next measurement
                chord_air.append(length_air)
                length_air = 1
        elif (signal[ii] == 0):             # Tip is in the water...
            if (signal[ii+1] == 0):         # and will keep on being in the water
                length_water = length_water+1
                if (ii == len(signal)-2):   # last chord
                    chord_water.append(length_water)
            elif (signal[ii+1] == 1):       # and gets dry in next measurement
                chord_water.append(length_water)
                length_water = 1
        # Display progress
        if progress:
            printProgressBar(ii, len(signal)-1, prefix = 'Progress:', suffix = 'Complete', length = 50)
    # particle count rate (1/s)
    F = len(chord_air)/duration

    return chord_water,chord_air,F

def windows(chord_air, chord_water, n_particles, f_sample, progress):
    """
        This function returns the windows start/stop, their computation depends
        upon nParticles

        chord_air:          air chords, based on the first tip.
        chord_water:        water chords, based on the first tip.
        n_particles:        number of particles making each signal subsegment.
        f_sample:           sampling rate
    """

    n_windows = min(
                math.floor(len(chord_air)/n_particles),
                math.floor(len(chord_water)/n_particles)
                )

    start = np.empty((n_windows),dtype='int')
    stop = np.empty((n_windows),dtype='int')

    # adaptive windows
    for ii in range(0,n_windows):
        start[ii] = sum(chord_air[0:n_particles*(ii+1)])+sum(chord_water[1:n_particles*(ii+1)])
        stop[ii] = sum(chord_air[0:n_particles*(ii+1)])+sum(chord_water[1:n_particles*(ii+1)])
        # Display progress
        if progress:
            printProgressBar(ii, n_windows, prefix = 'Progress:', suffix = 'Complete', length = 50)

    start = np.roll(start, 1)
    start[0] = 0;                                           # first time window
    stop[len(stop)-1] = sum(chord_water)+sum(chord_air)-1   #+1; #adding the last segment

    t = np.round((start+stop)/2.0)/f_sample                    #calculation of time window centres

    return n_windows,start,stop,t

def velocity(n_lags, delta_x, f_sample, S1, S2):
    """
        Robust velocity estimation including filtering criteria SPRthres and Rxymaxthres

        n_lags:             lags corresponding to time windows
        delta_x:            streamwise distance between tips
        f_sample:           sampling frequency of the sensor [1/s]
        S1:                 binary signal of the trailing tip
        S2:                 binary signal of the trailing tip
    """

    # calculate cross-correlation
    corr = signal.correlate(S2-np.mean(S2),S1-np.mean(S1))
    # calculate auto-correlation for signal 1
    auto_corr_S1 = signal.correlate(S1-np.mean(S1),S1-np.mean(S1))
    # calculate auto-correlation for signal 2
    auto_corr_S2 = signal.correlate(S2-np.mean(S2),S2-np.mean(S2))
    # check if some correlation detected
    if (sum(np.absolute(corr)) > 0.0):
        # normalize
        corr = corr/math.sqrt(auto_corr_S1[n_lags-1]*auto_corr_S2[n_lags-1])
        lags = signal.correlation_lags(len(S2), len(S1))
        # convert to time domain
        tau = lags/f_sample

        # find peaks of the cross-correlation function
        pks = signal.find_peaks(corr)[0]
        prom = signal.peak_prominences(corr, pks)[0]
        pks = corr[pks]
        idx = np.argmax(pks)
        # find peak locations
        peaks = pks[prom > 0.3*prom[idx]]
        SPR_nofilter = 0
        if (len(peaks) == 1):
            #SPR is zero if only one peak was detected
            SPR = 0
            SPR_nofilter = 0
        else:
            #otherwise second highest peak divided by the highest peak
            SPR = max(peaks[peaks<max(peaks)])/max(peaks)
            SPR_nofilter = max(peaks[peaks<max(peaks)])/max(peaks)

        # lag with max. cross-correlation coefficient
        lagsRxymax = np.argmax(corr)
        # maximum cross-correlation coefficient 
        Rxymax = corr[lagsRxymax]
        Rxymax_nofilter = corr[lagsRxymax]
        Vx_nofilter = (delta_x)/tau[lagsRxymax]
        # filtering based on SPRthres and Rxymaxthres
        if  (Rxymax > ((SPR**2.0 + 1.0)*0.4)):              # length(lagsRxymax)==1
            # pseudo-instantaneous velocity (m/s)
            Vx = (delta_x)/tau[lagsRxymax]
        else:
            Rxymax = np.nan
            Vx = np.nan
            SPR = np.nan
    else:
        Rxymax = np.nan
        Vx = np.nan
        SPR = np.nan
        Rxymax_nofilter = np.nan
        Vx_nofilter = np.nan
        SPR_nofilter = np.nan
    return Vx,Rxymax,SPR,Vx_nofilter,Rxymax_nofilter,SPR_nofilter

def evaluate_filtering(path, SPR, Rxymax, uinst):
    """
        Evaluate the cross-correlation based filtering approach by plotting the
        figure.

        path:       the path to store files
        SPR:        Secondary peak ratios
        Rxymax:     Maximum cross-correlations
        uinst:      Velocities 
    """

    spr_ineq = np.linspace(0.0,1.0,1000)
    rmax_ineq = 0.4*(np.square(spr_ineq) + 1.0)
    fig, ax = plt.subplots(1,1,figsize=(4,2.88))
    sc = plt.scatter(SPR,Rxymax,c=uinst,cmap='viridis')
    plt.plot(spr_ineq,rmax_ineq,color='k')
    ax.set_ylabel(r'$R_{\mathrm{12},i,\mathrm{max}}$ [-]')
    ax.set_xlabel(r'SPR$_i$ [-]')
    ax.set_xlim([0.0,1.0])
    ax.set_ylim([0.0,1.0])
    ax.grid()
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$U_x$ [m/s]', rotation=90)
    plt.subplots_adjust(left=0.16, bottom=0.15, right=0.88, top=0.98, wspace=0.2, hspace=0.3)
    fig.savefig(path / 'SPR_R12max.svg',dpi=300)

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

def roc_mod(u, u_med):
    """
        Robust outlier cutoff based on the maximum absolute deviation and the 
        universal threshold.

        u:                  the velocity time series
        u_med:              expected value
    """

    # ust estimated through MED
    u_std = 0.5*u_med
    N = len(u)
    u_filt = np.zeros((N,3))
    for ii in range(0,N):
        if (abs((u[ii,0]-u_med)/u_std) > 1.0):
            u_filt[ii,:] = np.nan
        else:
            u_filt[ii,:] = u[ii,:]
    return u_filt

def calc_velocity_awwcc(ii, start, stop, signal, f_sample, delta_x, args):
    n_lags = int(stop[ii] - start[ii])              # lags correspond to time windows
    S1 = signal[start[ii]:stop[ii],0]               # raw signals leading tip
    S2 = signal[start[ii]:stop[ii],1]               # raw signals trailing tip
    c_inst = np.mean(S1)                            # void fraction inside a window
    _,_,f_inst = chord(S1,(n_lags/f_sample))        # get F inside a window
    u_inst,Rxymax,SPR,u_inst_nofilter,Rxymax_nofilter,SPR_nofilter = velocity(n_lags,delta_x,f_sample,S1,S2) # pseudo-instantaneous velocities
    tmp = pd.DataFrame(np.array([n_lags,u_inst,f_inst,c_inst,Rxymax,SPR,u_inst_nofilter,Rxymax_nofilter,SPR_nofilter])[np.newaxis, :], index=[ii], columns=['n_lags','u_inst','f_inst','c_inst','Rxymax','SPR','u_inst_nofilter','Rxymax_nofilter','SPR_nofilter'])
    # Display progress
    if args.progress:
        printProgressBar(ii, n_windows, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return tmp

def run_sig_proc_awcc(path, args, config, sensor_ids, t_signal, signal):
    """
        Averaging Windows Cross-Correlation signal processing for dual-tip
        probes.

        path:       the velocity time series
        args:       the arguments from the CL-parser
        config:     the configuration JSON-file
        sensor_ids: IDs of the sensors
        t_signal:   the time series of the signal
        signal:     the signal
    """

    # get the number of particles per averaging windows
    n_particles = config['RECONSTRUCTION']['n_particles']
    # the sampling frequency
    f_sample = config['PROBE']['sampling_frequency']
    sensors = config['PROBE']['sensors']
    delta_x = abs(sensors[1]['relative_location'][0] - sensors[0]['relative_location'][0])
    # the sampling duration
    duration = Decimal(t_signal[-1])-Decimal(t_signal[0])
    # get the chord lengths in terms of number of samples
    # chord length [s] = chord length / f_sample
    print('Determining the air and water chord.\n')
    chord_w,chord_a,F1 = chord(signal[:,0],duration, progress=args.progress)
    print('Determining the windows.\n')
    n_windows,start,stop,t = windows(chord_a,chord_w,n_particles,f_sample, progress=args.progress)

    results = Parallel(n_jobs=int(args.nthreads),backend='multiprocessing')(delayed(calc_velocity_awwcc)(ii, start, stop, signal, f_sample, delta_x, args) for ii in range(0,n_windows))
    results = pd.concat(results)

    # remove velocities <= 0
    results['u_inst'] = np.where(results['u_inst'] >= 0, results['u_inst'], results['u_inst']*np.nan)

    # plot the figure for evaluation of the filtering
    evaluate_filtering(path, results['SPR_nofilter'], results['Rxymax_nofilter'], results['u_inst_nofilter'])

    # save some data for further use
    np.savetxt(path / 'SPR.csv',results['SPR_nofilter'], delimiter=',')
    np.savetxt(path / 'R12max.csv', results['Rxymax_nofilter'], delimiter=',')
    np.savetxt(path / 'U.csv', results['u_inst_nofilter'], delimiter=',')

    # store as bubble properties
    bubbles = []
    for ii in range(0,len(results['u_inst'])):
        ifd_times = pd.DataFrame(index=sensor_ids, columns=['t_2h','t_2h+1'])
        ifd_times['t_2h'][0] = start[ii]/f_sample
        ifd_times['t_2h+1'][0] = stop[ii]/f_sample
        ifd_times['t_2h'][1] = start[ii]/f_sample
        ifd_times['t_2h+1'][1] = stop[ii]/f_sample
        bubble_props = {'ifd_times':ifd_times}
        bubble_props['velocity'] = np.array([results['u_inst'][ii],0.0, 0.0])
        bubble_props['diameter'] = np.array([0.0, 0.0])
        bubble_props['if_norm_unit_vecs'] = [np.array([0.0, 0.0, 0.0]), \
                                            np.array([0.0, 0.0, 0.0])]
        bubbles.append(bubble_props)

    return bubbles

def get_sensor_distance_vectors(config):
    """
        Calculates the distance vectors between the leading tip (0) and the
        trailing tips (k).

        config:     the configuration JSON-file
    """

    # Get sensors from config file
    # Eq. (1) Shen and Nakamura (2014): distance vectors S_0k from the central
    # front sensor tip (0) to any of the three peripheral rear sensor tips k
    S_k = {}
    S_0k = {}
    S_0k_mag = {}
    cos_eta_0k = {}
    aux_sensor_ids = []
    max_t_k = {}
    sensors = config['PROBE']['sensors']
    for sensor in sensors:
        # Get the sensor ID
        s_id = sensor['id']
        # Get relative location vectors
        S_k[s_id] = np.asarray(sensor['relative_location'],dtype='str')
    for sensor in sensors:
        # Get the sensor ID
        s_id = sensor['id']
        if s_id != 0:
            aux_sensor_ids.append(s_id)
            # Set the distance vectors
            S_0k[s_id] = np.array([Decimal(S_k[s_id][0])-Decimal(S_k[0][0]),
                                   Decimal(S_k[s_id][1])-Decimal(S_k[0][1]),
                                   Decimal(S_k[s_id][2])-Decimal(S_k[0][2])],
                                  dtype='str')
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

    return aux_sensor_ids, max_t_k, S_0k_mag, S_0k, cos_eta_0k

def run_interface_pairing(idx_rise_0, id0, max_t_k, signal, signal_ifd, sensor_ids, aux_sensor_ids, t_signal_ifd_list, args):
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
    if len(signal_ifd_fall_0[signal_ifd_fall_0 > idx_rise_0]) > 0:
        idx_fall_0 = min(signal_ifd_fall_0[signal_ifd_fall_0 > idx_rise_0])
        # Write t_b+1 of sensor 0
        ifd_times['t_2h+1'][0] = t_signal_ifd_list[idx_fall_0]
    else:
        ifd_times['t_2h+1'][0] = np.nan

    # Auxillary sensors k
    for k in aux_sensor_ids:
        idk = np.where(sensor_ids == k)[0][0]
        # edit mb: ONLY forward search, due to basic assumption of flow direction
        # # Check if sensor is currently in air (1) or water (0) to determine
        # # the search direction in time (backward or forward)
        # phase = signal[idk]
        # if phase == 0:
        #     # Sensor k is currently in water phase -> search forward
        #     # Rising IFD signal
        #     # Indices of the rising IFD signals for aux. sensor k
        #     signal_ifd_rise_k = np.where(signal_ifd[:,idk] > 0.0)[0]
        #     # Ahead of rising IFD signal of sensor 0
        #     signal_ifd_rise_k = signal_ifd_rise_k[signal_ifd_rise_k > idx_rise_0]
        #     # Index of the rising IFD signal for aux. sensor k closest to
        #     # idx_rise_0
        #     if len(signal_ifd_rise_k) > 0:
        #         idx_rise_k = min(signal_ifd_rise_k)
        #     else:
        #         # no rising IFD signal before t_2h of main sensor, NaN
        #         idx_rise_k = np.nan

        # elif phase == 1:
        #     # Sensor k is currently in air phase -> search backward
        #     # Rising IFD signal
        #     # Indices of the rising IFD signals for aux. sensor k
        #     signal_ifd_rise_k = np.where(signal_ifd[:,idk] > 0.0)[0]
        #     # Behind of rising IFD signal of sensor 0
        #     signal_ifd_rise_k = signal_ifd_rise_k[signal_ifd_rise_k <= idx_rise_0]

        #     # Index of the rising IFD signal for aux. sensor k closest to
        #     # idx_rise_0
        #     if len(signal_ifd_rise_k) > 0:
        #         idx_rise_k = max(signal_ifd_rise_k)
        #     else:
        #         # no rising IFD signal before t_2h of main sensor, NaN
        #         idx_rise_k = np.nan

        # ONLY DO FORWARD SEARCH
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

        phase = signal[idk]

        # Search for t_2h+1 of sensor k
        if not np.isnan(idx_rise_k):
            # Indices of the falling IFD signals for main sensor k
            signal_ifd_fall_k = np.where(signal_ifd[:,idk] < 0.0)[0]
            signal_ifd_fall_k = signal_ifd_fall_k[signal_ifd_fall_k > idx_rise_k]
            # Index of the falling IFD signal for main sensor k
            if len(signal_ifd_fall_k) > 0:
                if phase == 0:
                    idx_fall_k = min(signal_ifd_fall_k)
                elif phase == 1:
                    if len(signal_ifd_fall_k) > 1:
                        idx_fall_k = signal_ifd_fall_k[1]
                    else:
                        idx_fall_k = np.nan
            else:
                idx_fall_k = np.nan
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
    # interface fall within the range for the lag between signal of sensors 0
    # and signal of sensor k from Eq. (42) in Shen et al. (2005)
    aux_sensors_complete = []
    for k in aux_sensor_ids:
        ok = True
        # first interface (2h)
        lag_2h = abs(Decimal(ifd_times['t_2h'][0]) - Decimal(ifd_times['t_2h'][k]))
        if lag_2h.compare(max_t_k[k]) == 1:
            # lag too large
            ifd_times['t_2h'][k] = np.nan
            ok = False
        # second interface (2h+1)
        lag_2hp1 = abs(Decimal(ifd_times['t_2h+1'][0]) - Decimal(ifd_times['t_2h+1'][k]))
        if lag_2hp1.compare(max_t_k[k]) == 1:
            # lag too large
            ifd_times['t_2h+1'][k] = np.nan
            ok = False
        if math.isnan(lag_2h) | math.isnan(lag_2h):
            ok = False
            ifd_times['t_2h'][k] = np.nan
            ifd_times['t_2h+1'][k] = np.nan
        # store IDs of sensors which recorded a valid signal
        if ok:
            aux_sensors_complete.append(k)
    # store sensors with valid signal as bubble property
    bubble_props = {'aux_sensors_complete':aux_sensors_complete}

    # Store the interface detection times for complete and
    # incomplete bubbles
    bubble_props['ifd_times'] = ifd_times

    # Display progress
    # if args.progress:
    #     printProgressBar(ii, len(signal_ifd_rise_0), prefix = 'Progress:', suffix = 'Complete', length = 50)

    return bubble_props


def run_event_detection(args, aux_sensor_ids, max_t_k, sensor_ids, t_signal, signal):
    """
        This is the event-detection (ED) algorithm, based on interface pairing.
        The algorithm matches the interface detection by the leading and 
        trailing tip(s) of the probe for each individual particle-probe 
        interaction.

        args:           the arguments from the CL-parser
        aux_sensor_ids: the auxillary (or trailing) sensor ids
        max_t_k:        valid range for lags between signal of sensors 0 and signal of sensor k
        sensor_ids:     IDs of the sensors
        t_signal:       the time series of the signal
        signal:         the signal
    """

    print('\nStarting event-detection algorithm....\n')
    # Create interface-detection (IFD) signal by first order difference
    signal_ifd = signal[1:len(signal)] - signal[0:(len(signal)-1)]

    # Get average time for each difference
    t_signal_ifd_list = []
    for ii in range(0,len(t_signal)-1):
        t_signal_ifd_list.append(Decimal(t_signal[ii]) \
            + (Decimal(t_signal[ii+1])-Decimal(t_signal[ii]))/Decimal(2.0))

    # Interface-pairing signal-processing scheme
    # Shen et al. (2005)
    # Get the column number of the main sensor 0, just in case the order gets
    # messed up along the process
    id0 = np.where(sensor_ids == 0)[0][0]
    # Indices of the rising IFD signals for main sensor 0
    signal_ifd_rise_0 = np.where(signal_ifd[:,id0] > 0.0)[0]
    # Initialize lists to store bubbles
    print('\nChecking for complete bubbles....\n')
    bubbles = Parallel(n_jobs=int(args.nthreads),backend='threading')(delayed(run_interface_pairing)(idx_rise_0, id0, max_t_k, signal[idx_rise_0,:], signal_ifd, sensor_ids, aux_sensor_ids, t_signal_ifd_list, args) for idx_rise_0 in signal_ifd_rise_0)

    return bubbles


def run_sig_proc_dual_ED(args, bubble_props, sensors, cos_eta_0k):
    """
        This is signal processing algorithm for dual-tip probes based on
        Kramer et el. (2020).

        Kramer, M., Hohermuth, B., Valero, D., & Felder, S. (2020). Leveraging 
        event detection techniques and cross-correlation analysis for phase-
        detection probe measurements in turbulent air-water flows.
        https://doi.org/10.14264/uql.2020.591

        args:           the arguments from the CL-parser
        bubble_props:   bubble properties
        sensors:        the sensor locations
                        and leading tips
        cos_eta_0k:     unit distance vectors
    """

    if (len(bubble_props['aux_sensors_complete']) == 1):
        # get the interface detection times
        ifd_times = bubble_props['ifd_times']

        # calculate central times of the dispersed phase particles
        # leading sensor (0)
        t_p_0 = (Decimal(ifd_times['t_2h+1'][0]) + Decimal(ifd_times['t_2h'][0])) \
                    / Decimal(2.0)
        # trailing sensor (1)
        t_p_1 = (Decimal(ifd_times['t_2h+1'][1]) + Decimal(ifd_times['t_2h'][1])) \
                    / Decimal(2.0)

        # calcualte the travel time between the leading and trailing sensor
        delta_t_p = t_p_1 - t_p_0
        delta_x = Decimal(abs(sensors[1]['relative_location'][0] - \
                    sensors[0]['relative_location'][0]))

        # calculate the instantaneous velocity magnitude
        u_inst_mag = delta_x / delta_t_p

        # calculate the instantaneous velocity vector
        u_inst = np.array([u_inst_mag, 0.0, 0.0])

        # set the particle properties
        bubble_props['velocity'] = u_inst
        bubble_props['diameter'] = np.array([0.0, 0.0])
        bubble_props['if_norm_unit_vecs'] = [np.array([0.0, 0.0, 0.0]), \
                                             np.array([0.0, 0.0, 0.0])]
    else:
        # not enough data, set the particle properties to NaN
        bubble_props['velocity'] =  np.array([np.nan, np.nan, np.nan])
        bubble_props['diameter'] = np.array([np.nan, np.nan])
        bubble_props['if_norm_unit_vecs'] = [np.array([np.nan, np.nan, np.nan]), \
                                             np.array([np.nan, np.nan, np.nan])]


def run_sig_proc_shen(args, aux_sensor_ids, bubble_props, S_0k_mag, cos_eta_0k):
    """
        This is signal processing algorithm of Shen & Nakamura (2014).

        Shen, X., Nakamura, H. (2014). Spherical-bubble-based four-sensor probe
        signal processing algorithm for two-phase flow measurement. 
        International journal of multiphase flow, 60, 11-29.
        https://doi.org/10.1016/j.ijmultiphaseflow.2013.11.010

        args:           the arguments from the CL-parser
        aux_sensor_ids: auxillary (or trailing) sensor ids
        bubble_props:   bubble properties
        S_0k_mag:       magnitudes of the distance vectors between trailing
                        and leading tips
        cos_eta_0k:     unit distance vectors
    """

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
    V_bh_x = V_bh_mag * V_bh_mag * A_01_h_det * inverse_den(A_0_det)
    V_bh_y = V_bh_mag * V_bh_mag * A_02_h_det * inverse_den(A_0_det)
    V_bh_z = V_bh_mag * V_bh_mag * A_03_h_det * inverse_den(A_0_det)
    velocity = np.array([V_bh_x, V_bh_y, V_bh_z])

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
    diameter = np.array([D_h_2h, D_h_2hp1])

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
    if_norm_unit_vecs = [cos_eta_i_2h, cos_eta_i_2hp1]

    # Calculate the local Interfacial Area Concentration (IAC)
    # Eq. (54) in Shen and Nakamura (2014)
    dummy_A = A_01_h_det*A_01_h_det + A_02_h_det*A_02_h_det + A_03_h_det*A_03_h_det
    dummy_B = (B_01_2h_det*B_01_2h_det \
                + B_02_2h_det*B_02_2h_det \
                + B_03_2h_det*B_03_2h_det).sqrt()
    dummy_AB = abs(A_0_det * (A_01_h_det*B_01_2h_det \
                            + A_02_h_det*B_02_2h_det \
                            + A_03_h_det*B_03_2h_det))
    iac = dummy_A * dummy_B * inverse_den(dummy_AB)

    return velocity, diameter, if_norm_unit_vecs, iac

def run_sig_proc_tian(args, aux_sensor_ids, bubble_props, S_0k_mag, S_0k):
    """
        This is signal processing algorithm of Tian et al. (2015).

        Tian, D., Yan, C., & Sun, L. (2015). Model of bubble velocity vector
        measurement in upward and downward bubbly two-phase flows using a four-
        sensor optical probe. Progress in nuclear energy, 78, 110-120.
        https://doi.org/10.1016/j.pnucene.2014.08.005

        args:           the arguments from the CL-parser
        aux_sensor_ids: auxillary (or trailing) sensor ids
        bubble_props:   bubble properties
        S_0k_mag:       magnitudes of the distance vectors between trailing
                        and leading tips
        S_0k:           distance vectors
    """

    ifd_times = bubble_props['ifd_times']

    # Time differences between sensors 0 and k for interfaces 2h (front
    # hemisphere) and 2h+1 (back hemisphere)
    delta_t_0k_2h = {}
    delta_t_0k_2hp1 = {}
    delta_t_0k_mean = {}
    # Time differences k=1,2,3,4 for interfaces 2h and 2h+1
    delta_t_k_h = {}
    delta_t_k_h[0] = Decimal(ifd_times['t_2h+1'][0]) - Decimal(ifd_times['t_2h'][0])

    # 1. Reconstruct instantaneous local velocity
    D0 = []
    D1 = []
    D2 = []
    D3 = []
    for k in aux_sensor_ids:
        # Time differences
        delta_t_0k_2h[k] = Decimal(ifd_times['t_2h'][k]) \
                         - Decimal(ifd_times['t_2h'][0])
        delta_t_0k_2hp1[k] = Decimal(ifd_times['t_2h+1'][k]) \
                           - Decimal(ifd_times['t_2h+1'][0])
        delta_t_k_h[k] = Decimal(ifd_times['t_2h+1'][k]) \
                       - Decimal(ifd_times['t_2h'][k])
        # Mean time difference
        delta_t_0k_mean[k] = (delta_t_0k_2h[k] + delta_t_0k_2hp1[k])/Decimal(2.0)
        # Fill the matrices given in Eq. (17) in Tian et al. (2015)
        D0.append(np.array([Decimal(S_0k[k][0]), Decimal(S_0k[k][1]), Decimal(S_0k[k][2])]))
        D1.append(np.array([delta_t_0k_mean[k], Decimal(S_0k[k][1]), Decimal(S_0k[k][2])]))
        D2.append(np.array([Decimal(S_0k[k][0]), delta_t_0k_mean[k], Decimal(S_0k[k][2])]))
        D3.append(np.array([Decimal(S_0k[k][0]), Decimal(S_0k[k][1]), delta_t_0k_mean[k]]))

    # Calculate matrix determinants
    # Eq. (17) in Tian et al. (2015)
    D0_det = calc_det(D0)
    D1_det = calc_det(D1)
    D2_det = calc_det(D2)
    D3_det = calc_det(D3)

    # Calculate instantaneous velocity magnitude and components
    # Eq. (16) from Tian et al. (2015)
    V =  np.sqrt((D0_det**2)/(D1_det**2 + D2_det**2 + D3_det**2))

    # Eq. (15) from Tian et al. (2015)
    V_bh_x = V * V * D1_det / D0_det
    V_bh_y = V * V * D2_det / D0_det
    V_bh_z = V * V * D3_det / D0_det

    velocity = np.array([V_bh_x, V_bh_y, V_bh_z])
    diameter = np.array([math.nan, math.nan])
    if_norm_unit_vecs = [np.array([math.nan, math.nan, math.nan]),\
                        np.array([math.nan, math.nan, math.nan])]
    iac = math.nan

    return velocity, diameter, if_norm_unit_vecs, iac

def multi_tip_signal_processing(ii, bubble_props, S_0k, S_0k_mag, cos_eta_0k, nb, ra_type, args):

    # Check how many sensors recorded a valid signal
    sensors_valid = bubble_props['aux_sensors_complete']
    n_sensors_valid = len(sensors_valid)
    # can only do reconstruction with 3 or more sensors
    if (n_sensors_valid >= 3):
        # get possible
        sensor_combs = combinations(sensors_valid, 3)
        # Initialize the variables to store the reconstructed properties
        vel = np.ones((n_sensors_valid,3))*math.nan
        diam = np.ones((n_sensors_valid,2))*math.nan
        iac = np.ones(n_sensors_valid)*math.nan
        for jj,sensor_comb in enumerate(list(sensor_combs)):
            if ra_type == "Shen_Nakamura_2014":
                # Reconstruction algorithm of Shen and Nakamura (2014)
                # https://doi.org/10.1016/j.ijmultiphaseflow.2013.11.010
                vel[jj,:],diam[jj,:],_,iac[jj] =\
                        run_sig_proc_shen(args,
                                        sensor_comb,
                                        bubble_props,
                                        S_0k_mag,
                                        cos_eta_0k)
            elif ra_type == "Tian_et_al_2015":
                # Reconstruction algorithm of Tian et al. (2015)
                # https://doi.org/10.1016/j.pnucene.2014.08.005
                vel[jj,:],diam[jj,:],_,iac[jj] =\
                        run_sig_proc_tian(args,
                                        sensor_comb,
                                        bubble_props,
                                        S_0k_mag,
                                        S_0k)
            else:
                PRINTERRORANDEXIT(f'Reconstruction algorithm '+\
                    '<{ra_type}> not valid for 4-tip probes.')
        # Store the computed bubble properties
        bubble_props['velocity'] = np.nanmean(vel, axis=0)
        bubble_props['diameter'] = np.nanmean(diam, axis=0)
        # bubble_props['if_norm_unit_vecs'] =\
        #                            np.nanmean(if_norm_unit_vec, axis=0)
        IAC = Decimal(np.nanmean(iac))
        # Display progress
        if args.progress:
            printProgressBar(ii, nb, prefix = 'Progress:', \
                suffix = 'Complete', length = 50)
        return [bubble_props, IAC]
    else:
        return ['nan', 'nan']

def get_awcc_properties(path, args, config, sensor_ids, t_signal, signal):
    # get the distance vector between leading and trailing tips
    aux_sensor_ids = []
    S_k = {}
    S_0k = {}
    S_0k_mag = {}
    sensors = config['PROBE']['sensors']
    for sensor in sensors:
        # Get the sensor ID
        s_id = sensor['id']
        # Get relative location vectors
        S_k[s_id] = np.asarray(sensor['relative_location'],dtype='str')
    for sensor in sensors:
        # Get the sensor ID
        s_id = sensor['id']
        if s_id != 0:
            aux_sensor_ids.append(s_id)
            # Set the distance vectors
            S_0k[s_id] = np.array([Decimal(S_k[s_id][0])-Decimal(S_k[0][0]),
                                   Decimal(S_k[s_id][1])-Decimal(S_k[0][1]),
                                   Decimal(S_k[s_id][2])-Decimal(S_k[0][2])],
                                  dtype='str')
            # Calculate magnitudes
            S_0k_mag[s_id] = (Decimal(S_0k[s_id][0]) * Decimal(S_0k[s_id][0])
                            + Decimal(S_0k[s_id][1]) * Decimal(S_0k[s_id][1])
                            + Decimal(S_0k[s_id][2]) * Decimal(S_0k[s_id][2])).sqrt()

    # get ID of sensor with smallest distance to leading sensor for AWCC
    min_dist_sensor = min(S_0k_mag, key=S_0k_mag.get)

    # run AWCC for two tips to get first estimate of mean velocity
    particles = run_sig_proc_awcc(path, args,
                                    config,
                                    sensor_ids,
                                    t_signal,
                                    signal[:,[0, min_dist_sensor]])

    # convert to arrays
    velocity_awcc = np.empty((len(particles),3))
    time_awcc = np.empty((len(particles),2))
    for ii,particle in enumerate(particles):
        velocity_awcc[ii,:] = np.array([particle['velocity'][0], \
                                            particle['velocity'][1], \
                                            particle['velocity'][2] \
                                            ])
        time_awcc[ii,0] = np.nanmin(particle['ifd_times'].to_numpy().astype('float64'))
        time_awcc[ii,1] = np.nanmax(particle['ifd_times'].to_numpy().astype('float64'))
    # run ROC for velocity estimate
    while sum(sum(np.isnan(velocity_awcc))) < sum(sum(np.isnan(roc(velocity_awcc)))):
        velocity_awcc = roc(velocity_awcc)
    # get mean velocity estimate
    weighted_mean_velocity_awcc = np.empty(3)
    weighted_mean_velocity_awcc[0] = np.nansum(velocity_awcc[:,0] \
                                * (time_awcc[:,1]-time_awcc[:,0]),axis=0) \
                                / np.nansum((time_awcc[:,1]-time_awcc[:,0])*(velocity_awcc[:,0]/velocity_awcc[:,0]),axis=0)
    weighted_mean_velocity_awcc[1] = np.nansum(velocity_awcc[:,1] \
                                * (time_awcc[:,1]-time_awcc[:,0]),axis=0) \
                                / np.nansum((time_awcc[:,1]-time_awcc[:,0])*(velocity_awcc[:,1]/velocity_awcc[:,1]),axis=0)
    weighted_mean_velocity_awcc[2] = np.nansum(velocity_awcc[:,2] \
                                * (time_awcc[:,1]-time_awcc[:,0]),axis=0) \
                                / np.nansum((time_awcc[:,1]-time_awcc[:,0])*(velocity_awcc[:,2]/velocity_awcc[:,2]),axis=0)        # Eq. (43) in Shen et al. (2005): estimated gas velocity
    # Initialize the reynolds stress tensors time series
    reynolds_stress_awcc = np.empty((len(velocity_awcc),3,3))
    for ii in range(0,len(velocity_awcc)):
        # Calculate velocity fluctuations
        velocity_fluct_reconst_awcc = velocity_awcc[ii,:] - weighted_mean_velocity_awcc
        # Reynolds stresses as outer product of fluctuations
        reynolds_stress_awcc[ii,:,:] = np.outer(velocity_fluct_reconst_awcc, \
                                    velocity_fluct_reconst_awcc)
    # Calculate mean Reynolds stresses
    mean_reynolds_stress_awcc = np.nanmean(reynolds_stress_awcc,axis=0)

    return weighted_mean_velocity_awcc, mean_reynolds_stress_awcc


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
    parser.add_argument('-roc', '--ROC', default='False',
        help='Perform robust outlier cutoff (ROC) based on the maximum' + 
                'absolute deviation and the universal threshold (True/False).')
    parser.add_argument(
        "-n",
        "--nthreads",
        metavar="NTHREADS",
        default=1,
        help="set the number of threads for parallel execution",
    )
    parser.add_argument('-p', '--progress', action='store_true',
        help='Show progress bar.')
    args = parser.parse_args()

    global V_GAS

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

    # Create a H5-file reader
    reader = H5Reader(path / 'binary_signal.h5')
    # Read the time vector
    t_signal = (reader.getDataSet('time')[:]).astype(str)
    # Read the signal time series
    ds_signal = reader.getDataSet('signal')
    # Get the signal
    signal = np.array(ds_signal, dtype='int8')
    # Get the corresponding sensor ids
    sensor_ids = ds_signal.attrs['sensor_id']
    reader.close()

    n_sensors = len(sensor_ids)
    # Check the reconstruction algorithm type
    ra_type = config['RECONSTRUCTION']['type']
    bubbles_complete = []
    if (n_sensors == 2):
        if (ra_type == "dual_tip_AWCC"):
            # AWCC for 2 tips
            bubbles_complete = run_sig_proc_awcc(path, args,
                                                config,
                                                sensor_ids,
                                                t_signal,
                                                signal)
            # interfacial area concentration cannot be estimated with AWCC
            IAC = math.nan
            print(f"\nDetected {len(bubbles_complete)} averaging windows.")
        elif (ra_type == "dual_tip_ED"):
            # Event-detection (ED) for 2 tips
            print('Running AWCC to get initial velocity estimate for iterface pairing.\n')
            # run AWCC for two tips to get first estimate of mean velocity
            particles = run_sig_proc_awcc(path, args,
                                            config,
                                            sensor_ids,
                                            t_signal,
                                            signal[:,0:2])

            velocity_estimate = np.empty((len(particles),3))
            for ii,particle in enumerate(particles):
                velocity_estimate[ii,:] = np.array([particle['velocity'][0], \
                                                    particle['velocity'][1], \
                                                    particle['velocity'][2]
                                                    ])
            # run ROC for velocity estimate
            while sum(sum(np.isnan(velocity_estimate))) < sum(sum(np.isnan(roc(velocity_estimate)))):
                velocity_estimate = roc(velocity_estimate)
            # get mean velocity estimate
            mean_velocity_estimate = np.nanmean(velocity_estimate, axis=0)
            # Eq. (43) in Shen et al. (2005): estimated gas velocity
            V_GAS = Decimal(mean_velocity_estimate[0])
            print(f'Initial velocity estimate from AWCC: {V_GAS} m/s')

            # get the distance vector between leading and trailing tips
            aux_sensor_ids, max_t_k,S_0k_mag,S_0k,cos_eta_0k = get_sensor_distance_vectors(config)
            sensors = config['PROBE']['sensors']
            # run event detection algorithm
            bubbles = run_event_detection(args,
                                    aux_sensor_ids,
                                    max_t_k,
                                    sensor_ids,
                                    t_signal,
                                    signal)
            # Loop over bubbles
            for ii,bubble_props in enumerate(bubbles):
                run_sig_proc_dual_ED(args,
                                    bubble_props,
                                    sensors,
                                    cos_eta_0k)
                # append to complete bubbles
                bubbles_complete.append(bubble_props)
            # interfacial area concentration cannot be estimated with AWCC
            IAC = math.nan
        else:
            PRINTERRORANDEXIT(f'Reconstruction algorithm <{ra_type}> not valid for 2-tip probes.')

    elif (n_sensors >= 4):
        # Reconstruction algorithm for 4 or more sensors
        print('Running AWCC to get initial velocity estimate for iterface pairing.\n')

        weighted_mean_velocity_awcc,mean_reynolds_stress_awcc = get_awcc_properties(path,
                                                                args,
                                                                config,
                                                                sensor_ids,
                                                                t_signal,
                                                                signal)

        V_GAS = Decimal(weighted_mean_velocity_awcc[0])

        print(f'Mean velocity from AWCC: {V_GAS:.2f} m/s')

        # get the distance vector between leading and trailing tips
        aux_sensor_ids, max_t_k,S_0k_mag,S_0k,cos_eta_0k = get_sensor_distance_vectors(config)
        # run event detection algorithm
        bubbles = run_event_detection(args,
                                    aux_sensor_ids,
                                    max_t_k,
                                    sensor_ids,
                                    t_signal,
                                    signal)

        print('\nStarting signal processing...\n')
        # Initialize the Interfacial Area Concentration (IAC)
        IAC = Decimal(0.0)
        bubbles_complete = []
        results = []
        for ii,bubble_props in enumerate(bubbles):
            results.append(multi_tip_signal_processing(ii, bubble_props, S_0k, S_0k_mag, cos_eta_0k, len(bubbles), ra_type, args))
        
        for bubble in results:
            if bubble[0] == 'nan':
                continue
            else:
                bubbles_complete.append(bubble[0])
                IAC = IAC + bubble[1]

        IAC = IAC * 2 / (Decimal(t_signal[-1])-Decimal(t_signal[0]))
        # end (n_sensors >= 4)
        print(f"\nDetected {len(bubbles)} bubble signals.")
        print(f"\nDetected {len(bubbles_complete)} complete bubble signals.")

    print('\nProcessing results....\n')
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
        time_reconst[ii,0] = np.nanmin(bubble_props['ifd_times'].to_numpy().astype('float64'))
        time_reconst[ii,1] = np.nanmax(bubble_props['ifd_times'].to_numpy().astype('float64'))
        bubble_diam_reconst[ii,:] = bubble_props['diameter']

    # data filtering
    # ROC filtering, R12 and SPR filtering implemented in previous loop
    if args.ROC == 'True':
        print('Performing robust outlier cutoff.\n')
        if ((n_sensors >= 4) | (ra_type == "dual_tip_ED")):
            velocity_reconst = roc_mod(velocity_reconst, float(V_GAS))
        while sum(sum(np.isnan(velocity_reconst))) < sum(sum(np.isnan(roc(velocity_reconst)))):
            velocity_reconst = roc(velocity_reconst)
        discarded=(sum(np.isnan(velocity_reconst)[:,0])/len(velocity_reconst))*100
        print(f'Discarded data: {discarded:.2f} %\n')

    weighted_mean_velocity_reconst[0] = np.nansum(velocity_reconst[:,0] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.nansum((time_reconst[:,1]-time_reconst[:,0])*(velocity_reconst[:,0]/velocity_reconst[:,0]),axis=0)
    weighted_mean_velocity_reconst[1] = np.nansum(velocity_reconst[:,1] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.nansum((time_reconst[:,1]-time_reconst[:,0])*(velocity_reconst[:,1]/velocity_reconst[:,1]),axis=0)
    weighted_mean_velocity_reconst[2] = np.nansum(velocity_reconst[:,2] \
                                    * (time_reconst[:,1]-time_reconst[:,0]),axis=0) \
                                    / np.nansum((time_reconst[:,1]-time_reconst[:,0])*(velocity_reconst[:,2]/velocity_reconst[:,2]),axis=0)
    # Calculate mean velocity
    mean_velocity_reconst = np.nanmean(velocity_reconst, axis=0)
    # Initialize the reynolds stress tensors time series
    reynolds_stress = np.empty((len(velocity_reconst),3,3))
    for ii in range(0,len(velocity_reconst)):
        # Calculate velocity fluctuations
        velocity_fluct_reconst = velocity_reconst[ii,:] - weighted_mean_velocity_reconst
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
            ])) / np.sqrt(weighted_mean_velocity_reconst.dot(weighted_mean_velocity_reconst))
    print('\nSaving results....\n')
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
    if ((n_sensors >= 4) | (ra_type == "dual_tip_ED")):
        writer.writeDataSet('bubbles/mean_velocity_awcc', \
            weighted_mean_velocity_awcc, 'float64')
        writer.writeDataSet('bubbles/reynold_stresses_awcc', \
        mean_reynolds_stress_awcc, 'float64')
    ds_vel = writer.getDataSet('bubbles/velocity')
    # Add the attributes
    ds_vel.attrs['labels'] = ['Ux','Uy','Uz']
    # Create the dataset for the bubble diameter
    writer.writeDataSet('bubbles/diameters', bubble_diam_reconst, 'float64')
    ds_d = writer.getDataSet('bubbles/diameters')
    # Add the attributes
    ds_d.attrs['labels'] = ['D_h_2h', 'D_h_2hp1']
    # Create the IAC and void_fraction datasets
    writer.writeDataSet('IAC', np.array([IAC],dtype='float64'), 'float64')
    writer.writeDataSet('voidFraction', np.array([np.mean(signal[:,0])]), 'float64')
    writer.close()
    time2 = time.time()
    print(f'Successfully run the reconstruction algorithm')
    print(f'Finished in {time2-time1:.2f} seconds\n')

if __name__ == "__main__":
    main()