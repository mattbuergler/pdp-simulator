#!/usr/bin/env python3
import os
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
from scipy.stats import chi2
from joblib import Parallel, delayed
import warnings
import gc

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
try:
    from dataio.H5Writer import H5Writer
    from dataio.H5Reader import H5Reader
    from globals import *
except ImportError:
    print("Error while importing modules")
    raise

def run_roc_filter(values: np.ndarray = None):
    """
    Iteratively running the robust outlier cutoff filter based on the
    standard deviation and the universal threshold.

    Parameters
    ----------
    values : np.ndarray
        The values to be filtered.

    Returns
    -------
    np.ndarray
        The ROC filtered values.

    """
    val = np.copy(values)
    if len(val[~np.isnan(val)]) > 0:
        N = len(val)
        while len(val[np.isnan(val)]) < \
                            len(val[np.isnan(roc_filter_1d(val))]):
            val = roc_filter_1d(val)
    else:
        print('100% invalid data. Not performing ROC.\n')
    return val

def nan_weighted_average(A: np.ndarray = None,
            weights: np.ndarray = None,
            axis: np.ndarray = None):
    """
    Return the weighted average and standard deviation for
    arrays containing NaN values.

    Parameters
    ----------
    values : np.ndarray
        The time series values.
    weights : np.ndarray
        The weights.

    Returns
    -------
    np.ndarray
        The weighted averaged of the time series, ignoring NaN entries.

    """
    return np.nansum(A*weights,axis=axis) \
                / ((~np.isnan(A))*weights).sum(axis=axis)


def weighted_moments(values: np.ndarray = None,
                weights: np.ndarray = None):
    """
    Return the weighted average and standard deviation.

    Parameters
    ----------
    values : np.ndarray
        The time series values.
    weights : np.ndarray
        The weights.

    Returns
    -------
    average : np.ndarray
        The first order moment of the time series.
    std_dev : np.ndarray
        The second order moment of the time series.
    """

    average = nan_weighted_average(values, weights=weights, axis=0)
    std_dev = np.sqrt(nan_weighted_average((values-average)**2,
                                    weights=weights, axis=0))
    return average, std_dev

"""
    Evaluate the reconstructed data
"""

def get_input_parameters(config_file):
    # Read the configuration file
    config = json.loads(config_file.read_bytes())
    # Get flow properties
    mean_velocity = config['FLOW_PROPERTIES']['mean_velocity']
    turbulent_intensity = config['FLOW_PROPERTIES']['turbulent_intensity']
    integral_time_scale = config['FLOW_PROPERTIES']['integral_timescale']
    # Get aerated flow properties
    void_fraction = config['FLOW_PROPERTIES']['void_fraction']
    if (config['FLOW_PROPERTIES']['bubbles']['shape'] == 'sphere') & \
        (config['FLOW_PROPERTIES']['bubbles']['size_distribution'] == 'constant'):
        bubble_diameter = np.ones(3)*config['FLOW_PROPERTIES']['bubbles']['diameter']
    else:
        print('ERROR: Currently only implemented for spherical bubbles and constant diameters.')
    duration = config['FLOW_PROPERTIES']['duration']
    # Get sampling and probe characteristics
    sampling_frequency = config['PROBE']['sampling_frequency']
    n_sensors = len(config['PROBE']['sensors'])
    sensor_delta = {}
    for sensor in config['PROBE']['sensors']:
        # fill dictionary:  id -> relative_location
        sensor_delta[sensor['id']] = np.asarray(sensor['relative_location'])
    # Get an estimate of the maximum probe dimension
    sensors = config['PROBE']['sensors']
    min_range = np.array([LARGENUMBER,LARGENUMBER,LARGENUMBER])
    max_range = np.array([LARGENEGNUMBER,LARGENEGNUMBER,LARGENEGNUMBER])
    for sensor in sensors:
        for ii in range(0,3):
            min_range[ii] = min(min_range[ii], sensor['relative_location'][ii])
            max_range[ii] = max(max_range[ii], sensor['relative_location'][ii])
    max_probe_size = max_range - min_range
    if (config['RECONSTRUCTION']['type'] == 'dual_tip_AWCC'):
        n_p = config['RECONSTRUCTION']['n_particles']
    else:
        print('ERROR: Currently only implemented for AWCC.')
        sys.exit()
    df = pd.DataFrame([[
                        mean_velocity[0],
                        mean_velocity[1],
                        mean_velocity[2],
                        turbulent_intensity[0],
                        turbulent_intensity[1],
                        turbulent_intensity[2],
                        integral_time_scale[0],
                        integral_time_scale[1],
                        integral_time_scale[2],
                        void_fraction,
                        bubble_diameter[0],
                        bubble_diameter[1],
                        bubble_diameter[2],
                        duration,
                        sampling_frequency,
                        max_probe_size[0],
                        max_probe_size[1],
                        max_probe_size[2],
                        n_p
                        ]],
                        columns=
                        [
                        'u_x_real','u_y_real','u_z_real',
                        'T_ux_real','T_uy_real','T_uz_real',
                        'T_x_real','T_y_real','T_z_real',
                        'c_real','d_bx_real','d_by_real','d_bz_real',
                        'T','f_s','delta_x','delta_y','delta_z','N_p'
                        ]
                        )
    return df

def get_flow_data(flow_data_file,awcc_windows):
    # Parse velocity time series
    # Create a H5-file reader
    reader = H5Reader(flow_data_file)
    # Read the time vector
    t_fluid = reader.getDataSet('fluid/time')[:]
    # Read the velocity vector
    u_fluid = reader.getDataSet('fluid/velocity')[:]
    # Read the mean velocity
    u_mean_fluid = reader.getDataSet('fluid/mean_velocity')[:]
    # Read the turbulent intensity
    turbulent_intensity_fluid = reader.getDataSet('fluid/turbulent_intensity')[:]
    # Read the bubble velocity vector
    u_bubbles = reader.getDataSet('bubbles/mean_velocity')[:]
    try:
        pierced_index = reader.getDataSet('bubbles/pierced_bubbles')[:]
    except:
        print(f'Could no read dataset <bubbles/pierced_bubbles> from {flow_data_file.name}')
        pierced_index = np.ones((len(u_bubbles),2))
    reader.close()
    gc.collect()
    u_fluid_awcc = window_average_fluid_velocity(t_fluid,u_fluid,awcc_windows)

    pierced_index = pierced_index.astype('float')
    pierced_index[pierced_index == 0] = np.nan
    n_fluid = len(u_fluid)
    u_rms_fluid = np.nanstd(u_fluid,axis=0)
    n_fluid = sum(~np.isnan(u_fluid[:,0]))
    # degrees of freedom
    dof = n_fluid - 1
    # find critical value Chi-square values
    critical_chi_sqr_low = chi2.ppf(1-signifcance_level/2.0, dof)
    critical_chi_sqr_high = chi2.ppf(signifcance_level/2.0, dof)
    ci_u_rms_fluid = np.array([
                math.sqrt(dof*u_rms_fluid[0]**2/critical_chi_sqr_low),
                math.sqrt(dof*u_rms_fluid[0]**2/critical_chi_sqr_high)
                    ])
    ci_turb_int_fluid = ci_u_rms_fluid/u_mean_fluid[0]
    # find critical t-distribution values
    critical_t = stats.t.ppf(1-signifcance_level/2.0,dof)
    ci_u_mean_fluid = np.array([
                critical_t*u_rms_fluid[0]/math.sqrt(n_fluid),
                critical_t*u_rms_fluid[0]/math.sqrt(n_fluid)
                    ])

    u_mean_fluid_awcc = np.nanmean(u_fluid_awcc,axis=0)
    u_rms_fluid_awcc = np.nanstd(u_fluid_awcc,axis=0)

    if len(u_fluid_awcc) > 0:
        turbulent_intensity_fluid_awcc = u_rms_fluid_awcc/u_mean_fluid_awcc[0]
        n_fluid_awcc = sum(~np.isnan(u_fluid_awcc[:,0]))
        # degrees of freedom
        dof = n_fluid_awcc - 1
        # find critical value Chi-square values
        critical_chi_sqr_low = chi2.ppf(1-signifcance_level/2.0, dof)
        critical_chi_sqr_high = chi2.ppf(signifcance_level/2.0, dof)
        ci_u_rms_fluid_awcc = np.array([
                    math.sqrt(dof*u_rms_fluid_awcc[0]**2/critical_chi_sqr_low),
                    math.sqrt(dof*u_rms_fluid_awcc[0]**2/critical_chi_sqr_high)
                        ])
        ci_turb_int_fluid_awcc = ci_u_rms_fluid_awcc/u_mean_fluid_awcc[0]
        # find critical t-distribution values
        critical_t = stats.t.ppf(1-signifcance_level/2.0,dof)
        ci_u_mean_fluid_awcc = np.array([
                    critical_t*u_rms_fluid_awcc[0]/math.sqrt(n_fluid_awcc),
                    critical_t*u_rms_fluid_awcc[0]/math.sqrt(n_fluid_awcc)
                        ])
    else:
        turbulent_intensity_fluid_awcc = np.array([np.nan,np.nan,np.nan])
        ci_turb_int_fluid_awcc = np.array([np.nan,np.nan])
        ci_u_mean_fluid_awcc = np.array([np.nan,np.nan])
        ci_u_rms_fluid_awcc = np.array([np.nan,np.nan])

    n_bubbles = len(u_bubbles[~np.isnan(u_bubbles)])
    u_mean_bubbles = np.nanmean(u_bubbles,axis=0)
    u_rms_bubbles = np.nanstd(u_bubbles,axis=0)
    turbulent_intensity_bubbles = u_rms_bubbles/u_mean_bubbles[0]

    n_bubbles = sum(~np.isnan(u_bubbles[:,0]))
    # degrees of freedom
    dof = n_bubbles - 1
    # find critical value Chi-square values
    critical_chi_sqr_low = chi2.ppf(1-signifcance_level/2.0, dof)
    critical_chi_sqr_high = chi2.ppf(signifcance_level/2.0, dof)
    ci_u_rms_bubbles = np.array([
                math.sqrt(dof*u_rms_bubbles[0]**2/critical_chi_sqr_low),
                math.sqrt(dof*u_rms_bubbles[0]**2/critical_chi_sqr_high)
                    ])
    ci_turb_int_bubbles = ci_u_rms_bubbles/u_mean_bubbles[0]
    # find critical t-distribution values
    critical_t = stats.t.ppf(1-signifcance_level/2.0,dof)
    ci_u_mean_bubbles = np.array([
                critical_t*u_rms_bubbles[0]/math.sqrt(n_bubbles),
                critical_t*u_rms_bubbles[0]/math.sqrt(n_bubbles)
                    ])

    u_pierced_bubbles = ((u_bubbles.T * pierced_index[:,0])* pierced_index[:,1]).T
    n_pierced_bubbles = len(u_pierced_bubbles[~np.isnan(u_pierced_bubbles)])
    u_mean_pierced_bubbles = np.nanmean(u_pierced_bubbles,axis=0)
    u_rms_pierced_bubbles = np.nanstd(u_pierced_bubbles,axis=0)
    turbulent_intensity_pierced_bubbles = u_rms_pierced_bubbles/u_mean_pierced_bubbles[0]

    n_pierced_bubbles = sum(~np.isnan(u_pierced_bubbles[:,0]))
    # degrees of freedom
    dof = n_pierced_bubbles - 1
    # find critical value Chi-square values
    critical_chi_sqr_low = chi2.ppf(1-signifcance_level/2.0, dof)
    critical_chi_sqr_high = chi2.ppf(signifcance_level/2.0, dof)
    ci_u_rms_pierced_bubbles = np.array([
                math.sqrt(dof*u_rms_pierced_bubbles[0]**2/critical_chi_sqr_low),
                math.sqrt(dof*u_rms_pierced_bubbles[0]**2/critical_chi_sqr_high)
                    ])
    ci_turb_int_pierced_bubbles = ci_u_rms_pierced_bubbles/u_mean_pierced_bubbles[0]
    # find critical t-distribution values
    critical_t = stats.t.ppf(1-signifcance_level/2.0,dof)
    ci_u_mean_pierced_bubbles = np.array([
                critical_t*u_rms_pierced_bubbles[0]/math.sqrt(n_pierced_bubbles),
                critical_t*u_rms_pierced_bubbles[0]/math.sqrt(n_pierced_bubbles)
                    ])

    df = pd.DataFrame([[
                        n_fluid,
                        n_bubbles,
                        n_pierced_bubbles,
                        u_mean_fluid[0],
                        u_mean_fluid[1],
                        u_mean_fluid[2],
                        u_mean_fluid[0]-ci_u_mean_fluid[0],
                        u_mean_fluid[0]+ci_u_mean_fluid[1],
                        u_rms_fluid[0],
                        u_rms_fluid[1],
                        u_rms_fluid[2],
                        ci_u_rms_fluid[0],
                        ci_u_rms_fluid[1],
                        turbulent_intensity_fluid[0],
                        turbulent_intensity_fluid[1],
                        turbulent_intensity_fluid[2],
                        ci_turb_int_fluid[0],
                        ci_turb_int_fluid[1],
                        u_mean_fluid_awcc[0],
                        u_mean_fluid_awcc[1],
                        u_mean_fluid_awcc[2],
                        u_mean_fluid_awcc[0]-ci_u_mean_fluid_awcc[0],
                        u_mean_fluid_awcc[0]+ci_u_mean_fluid_awcc[1],
                        u_rms_fluid_awcc[0],
                        u_rms_fluid_awcc[1],
                        u_rms_fluid_awcc[2],
                        ci_u_rms_fluid_awcc[0],
                        ci_u_rms_fluid_awcc[1],
                        turbulent_intensity_fluid_awcc[0],
                        turbulent_intensity_fluid_awcc[1],
                        turbulent_intensity_fluid_awcc[2],
                        ci_turb_int_fluid_awcc[0],
                        ci_turb_int_fluid_awcc[1],
                        u_mean_bubbles[0],
                        u_mean_bubbles[1],
                        u_mean_bubbles[2],
                        u_mean_bubbles[0]-ci_u_mean_bubbles[0],
                        u_mean_bubbles[0]+ci_u_mean_bubbles[1],
                        u_rms_bubbles[0],
                        u_rms_bubbles[1],
                        u_rms_bubbles[2],
                        ci_u_rms_bubbles[0],
                        ci_u_rms_bubbles[1],
                        turbulent_intensity_bubbles[0],
                        turbulent_intensity_bubbles[1],
                        turbulent_intensity_bubbles[2],
                        ci_turb_int_bubbles[0],
                        ci_turb_int_bubbles[1],
                        u_mean_pierced_bubbles[0],
                        u_mean_pierced_bubbles[1],
                        u_mean_pierced_bubbles[2],
                        u_mean_pierced_bubbles[0]-ci_u_mean_pierced_bubbles[0],
                        u_mean_pierced_bubbles[0]+ci_u_mean_pierced_bubbles[1],
                        u_rms_pierced_bubbles[0],
                        u_rms_pierced_bubbles[1],
                        u_rms_pierced_bubbles[2],
                        ci_u_rms_pierced_bubbles[0],
                        ci_u_rms_pierced_bubbles[1],
                        turbulent_intensity_pierced_bubbles[0],
                        turbulent_intensity_pierced_bubbles[1],
                        turbulent_intensity_pierced_bubbles[2],
                        ci_turb_int_pierced_bubbles[0],
                        ci_turb_int_pierced_bubbles[1]
                        ]],
                        columns=
                        [
                        'n_fluid','n_bubbles','n_pierced_bubbles',
                        'u_x_fluid','u_y_fluid','u_z_fluid',
                        'u_x_fluid_2.5%CI','u_x_fluid_97.5%CI',
                        'u_rms_x_fluid','u_rms_y_fluid','u_rms_z_fluid',
                        'u_rms_x_fluid_2.5%CI','u_rms_x_fluid_97.5%CI',
                        'T_ux_fluid','T_uy_fluid','T_uz_fluid',
                        'T_ux_fluid_2.5%CI','T_ux_fluid_97.5%CI',
                        'u_x_fluid_awcc','u_y_fluid_awcc','u_z_fluid_awcc',
                        'u_x_fluid_awcc_2.5%CI','u_x_fluid_awcc_97.5%CI',
                        'u_rms_x_fluid_awcc','u_rms_y_fluid_awcc','u_rms_z_fluid_awcc',
                        'u_rms_x_fluid_awcc_2.5%CI','u_rms_x_fluid_awcc_97.5%CI',
                        'T_ux_fluid_awcc','T_uy_fluid_awcc','T_uz_fluid_awcc',
                        'T_ux_fluid_awcc_2.5%CI','T_ux_fluid_awcc_97.5%CI',
                        'u_x_bubbles','u_y_bubbles','u_z_bubbles',
                        'u_x_bubbles_2.5%CI','u_x_bubbles_97.5%CI',
                        'u_rms_x_bubbles','u_rms_y_bubbles','u_rms_z_bubbles',
                        'u_rms_x_bubbles_2.5%CI','u_rms_x_bubbles_97.5%CI',
                        'T_ux_bubbles','T_uy_bubbles','T_uz_bubbles',
                        'T_ux_bubbles_2.5%CI','T_ux_bubbles_97.5%CI',
                        'u_x_pierced_bubbles','u_y_pierced_bubbles','u_z_pierced_bubbles',
                        'u_x_pierced_bubbles_2.5%CI','u_x_pierced_bubbles_97.5%CI',
                        'u_rms_x_pierced_bubbles','u_rms_y_pierced_bubbles','u_rms_z_pierced_bubbles',
                        'u_rms_x_pierced_bubbles_2.5%CI','u_rms_x_pierced_bubbles_97.5%CI',
                        'T_ux_pierced_bubbles','T_uy_pierced_bubbles','T_uz_pierced_bubbles',
                        'T_ux_pierced_bubbles_2.5%CI','T_ux_pierced_bubbles_97.5%CI'
                        ]
                        )
    return df


def get_signal(signal_file):
    # Create a H5-file reader
    reader = H5Reader(signal_file)
    # Read the time vector of the signal time series
    t_signal = reader.getDataSet('time')[:]
    # Read the signal time series
    signal = reader.getDataSet('signal')[:]
    reader.close()
    gc.collect()
    return t_signal,signal


def get_awcc_data(awcc_file):
    # Create a H5-file reader
    reader = H5Reader(awcc_file)
    # Read the reconstructed time bubbles
    t_awcc = reader.getDataSet('bubbles/interaction_times')[:]
    # Read the reconstructed velocity time series
    u_awcc = reader.getDataSet('bubbles/velocity')[:]
    # Read the bubble size
    b_size_rec = reader.getDataSet('bubbles/diameters')[:]
    # Read the bubble frequency
    F_awcc = reader.getDataSet('bubbles/bubble_frequency')[:][0]
    c_awcc = reader.getDataSet('voidFraction')[:][0]
    reader.close()
    # remove values < 0
    u_awcc[u_awcc < 0] = np.nan
    u_awcc[np.isinf(u_awcc)] = np.nan
    # run ROC filtering
    u_awcc[:,0] = run_roc_filter(u_awcc[:,0])
    # Calculate weighted moments of reconstructed time series
    # weights = (t_awcc[:,1]-t_awcc[:,0])
    weights = np.ones(len(u_awcc))
    # get mean and std_dev
    u_mean_awcc,u_rms_awcc = weighted_moments(u_awcc[:,0],weights)
    # calculate turbulent intensity
    turb_int_awcc = u_rms_awcc / u_mean_awcc
    # calculate confidence intervals
    n_vel_samples = sum(~np.isnan(u_awcc[:,0]))
    # degrees of freedom
    dof = n_vel_samples - 1
    # find critical value Chi-square values
    critical_chi_sqr_low = chi2.ppf(1-signifcance_level/2.0, dof)
    critical_chi_sqr_high = chi2.ppf(signifcance_level/2.0, dof)
    ci_std_dev_velocity_rec = np.array([
                math.sqrt(dof*u_rms_awcc**2/critical_chi_sqr_low),
                math.sqrt(dof*u_rms_awcc**2/critical_chi_sqr_high)
                    ])
    ci_turb_int_rec = ci_std_dev_velocity_rec/u_mean_awcc
    # find critical t-distribution values
    critical_t = stats.t.ppf(1-signifcance_level/2.0,dof)
    ci_mean_velocity_awcc = np.array([
                critical_t*u_rms_awcc/math.sqrt(n_vel_samples),
                critical_t*u_rms_awcc/math.sqrt(n_vel_samples)
                    ])
    d_32_awcc = 1.5*u_mean_awcc*c_awcc/F_awcc
    df = pd.DataFrame([[
                        n_vel_samples,
                        u_mean_awcc,
                        u_mean_awcc-ci_mean_velocity_awcc[0],
                        u_mean_awcc+ci_mean_velocity_awcc[1],
                        u_rms_awcc,
                        ci_std_dev_velocity_rec[0],
                        ci_std_dev_velocity_rec[1],
                        turb_int_awcc,
                        ci_turb_int_rec[0],
                        ci_turb_int_rec[1],
                        c_awcc,F_awcc,d_32_awcc
                        ]],
                        columns=
                        [
                        'n_awcc',
                        'u_x_awcc','u_x_awcc_2.5%CI','u_x_awcc_97.5%CI',
                        'u_rms_x_awcc','u_rms_x_awcc_2.5%CI','u_rms_x_awcc_97.5%CI',
                        'T_ux_awcc','T_ux_awcc_2.5%CI','T_ux_awcc_97.5%CI',
                        'c_awcc','F_awcc','d_32_awcc'
                        ]
                        )
    return df,t_awcc.mean(axis=1),u_awcc[:,0],t_awcc

def get_awcc_raw_data(path):
    # Read raw awcc data without SPR and R12_max filtering
    u_awcc_raw = np.genfromtxt(path / 'U.csv')
    SPR = np.genfromtxt(path / 'SPR.csv')
    R12max = np.genfromtxt(path / 'R12max.csv')
    # remove inf values
    u_awcc_raw[np.isinf(u_awcc_raw)] = np.nan
    SPR[np.isinf(SPR)] = np.nan
    R12max[np.isinf(R12max)] = np.nan
    return u_awcc_raw,SPR,R12max

def simulation_exists(simulation_results,simulation_id):
    idx = np.where(simulation_results['id'].values == simulation_id)[0]
    return (len(idx) > 0)

def remove_from_results(simulation_results,simulation_id):
    simulation_results.drop(simulation_results.index[np.where(simulation_results['id'].values == simulation_id)[0]],inplace=True)

def window_average_fluid_velocity(t_fluid,u_fluid,awcc_windows):
    u_fluid_awcc = np.empty((len(awcc_windows),u_fluid.shape[1]))
    for ii in range(0,len(awcc_windows)):
        idx = np.where((t_fluid >= awcc_windows[ii,0]) & (t_fluid < awcc_windows[ii,1]))[0]
        if len(idx) > 0:
            u_fluid_awcc[ii,:] = np.nanmean(u_fluid[idx,:],axis=0)
        else:
            u_fluid_awcc[ii,:] = np.array([np.nan,np.nan,np.nan])
    return u_fluid_awcc

def process_simulation(path,simulation_id,simulation_results_file,overwrite):
    # check if files exist
    config_file = path / 'config.json'
    flow_data_file = path / 'flow_data.h5'
    awcc_file = path / 'reconstructed.h5'
    if (config_file.is_file() & flow_data_file.is_file() & awcc_file.is_file()):
        # get input parameters
        df_input = get_input_parameters(config_file)
        df_input['id'] = simulation_id
        # get awcc processed data
        df_awcc,t_awcc,u_x_awcc,awcc_windows = get_awcc_data(awcc_file)
        # get flow data
        df_sbg = get_flow_data(flow_data_file,awcc_windows)
        # get awcc raw data
        u_awcc_raw,SPR,R12max = get_awcc_raw_data(path)

        u_real = np.array([df_input['u_x_real'][0],df_input['u_y_real'][0],df_input['u_z_real'][0]])
        u_x_real = df_input['u_x_real'][0]
        T_ux_real = df_input['T_ux_real'][0]
        T_x_real = df_input['T_x_real'][0]
        void_fraction_real = df_input['c_real'][0]
        duration = df_input['T'][0]
        result = pd.concat([df_input,df_sbg,df_awcc], axis=1, join='inner')
        simulation_results = pd.read_csv(simulation_results_file, index_col=None)
        if simulation_exists(simulation_results,simulation_id):
            # exists already
            if overwrite:
                # overwrite anyway
                print('save data.')
                # delete row
                remove_from_results(simulation_results,simulation_id)
                simulation_results = simulation_results.reset_index(drop=True)
                simulation_results = pd.concat([simulation_results,result]).reset_index(drop=True)
                simulation_results = simulation_results.sort_values(['id'])
                simulation_results.to_csv(simulation_results_file, index=False,
                            na_rep='nan')
            else:
                print('Data already exists and is not overwritten.')
        else:
            print('save data.')
            simulation_results = pd.concat([simulation_results,result]).reset_index(drop=True)
            simulation_results = simulation_results.sort_values(['id'])
            simulation_results.to_csv(simulation_results_file, index=False,
                        na_rep='nan')
        print('Finished.')


def main():
    """
        Main function of the the evaluation
    """
    # Create parser to read in the configuration JSON-file to read from
    # the command line interface (CLI)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--id', '-i', type=str, default='',
        help='The simulation id.')
    parser.add_argument(
        '--simulation_results_file', '-r', type=str, default='',
        help='Path to the simulation result table file.')
    parser.add_argument(
        "-o", "--overwrite", action='store_true', help="overwrite existing data.",
    )
    parser.add_argument('path', type=str,
        help="The path to the scenario directory.")
    args = parser.parse_args()

    global signifcance_level
    signifcance_level = 0.05


    # Create Posix path for OS indepency
    path = pathlib.Path(args.path)
    simulation_results_file = pathlib.Path(args.simulation_results_file)
    process_simulation(path,int(args.id),simulation_results_file,args.overwrite)

if __name__ == "__main__":
    main()