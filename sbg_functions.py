#!/usr/bin/env python3

import numpy as np
import pandas as pd
import bisect
import random
import time
import math
from joblib import Parallel, delayed
from pathlib import Path
import shutil
import scipy
from matplotlib import pyplot as plt

try:
    from dataio.H5Writer import H5Writer
    from dataio.H5Reader import H5Reader
    from tools.globals import *
except ImportError:
    print("Error while importing modules")
    raise

def SBG_dt_corr(Uprev, Um, sigma, Ti, dt, R):
    """Auto-correlation between previous and current realization by Langevin
    equation

        Parameters
        ----------
        Uprev (float): Previous velocity realization
        Um    (float): Mean velocity component
        sigma (float): Std. deviation
        Ti    (float): Integral time scale component
        dt    (float): Realization timestep
        R     (float): random cross-correlation component

        Returns
        ----------
        U     (float): The instantaneous velocity realization
    """

    # Langevin equation (Eq. 12.90, Pope, 2000)
    # R cross-correlated random number
    unew = (Uprev-Um)*(1.0 - dt/Ti) + sigma*np.sqrt(2.0*dt/Ti)*R;
    # velocity = mean + fluctuation
    U = Um + unew;
    return U

def create_bubbles(flow_properties, F):
    """Produces a vector with bubble sizes

        Parameters
        ----------
        flow_properties   (dict): A dictionary containing the flow properties
        F                 (dict): The bubble frequency

        Returns
        ----------
        nb        (float): The number of bubbles
        b_size (np.array): A vector with bubble size
    """

    # Define some variables for easier use
    tau = flow_properties['duration'];
    C = flow_properties['void_fraction'];
    Um = flow_properties['mean_velocity'];
    # Get bubble properties
    bubbles = flow_properties['bubbles']
    if bubbles['shape'] == 'sphere':
        if bubbles['size_distribution'] == 'constant':
            # Number of simulated bubbles
            nb = round(F * tau)
            # Vector with bubble size (A,B,C), V = 1/6*Pi*A*B*C
            b_size = np.ones((nb,3))*bubbles['diameter']
        elif bubbles['size_distribution'] == 'lognormal':
            # Number of simulated bubbles
            nb = round(F * tau)
            # Sample lognormal distribution
            D = np.random.lognormal(bubbles['mean_ln_diameter'],
                                     bubbles['sd_ln_diameter'],
                                     size=nb)
            # Vector with bubble size (A,B,C), V = 1/6*Pi*A*B*C
            b_size = np.array([D,D,D]).T
    elif bubbles['shape'] == 'ellipsoid':
        if bubbles['size_distribution'] == 'constant':
            # The shorter axis (streamwise)
            A = np.nan
            # The longer axes (perpendicular to streamwise direction)
            B = np.nan
            # The aspect ratio
            E = np.nan
            # Sphere-volume-equivalent bubble diameter
            D = bubbles['diameter']
            if bubbles['aspect_ratio_type'] == 'constant':
                E = bubbles['aspect_ratio_value']
            elif bubbles['aspect_ratio_type'] == 'Aoyama':
                # Bubble Reynolds number
                Re = (1000 * bubbles['slip_velocity_ratio'] \
                    * np.linalg.norm(Um) * D) / (10**(-3))
                # Bubble Eotvos number
                Eo = (999 * 9.81 * D**2) / 0.07
                # The aspect ratio (Aoyama et al. 2016)
                E = 1.0/(1.0 + (0.016 * Eo**1.12 * Re))**0.388
            elif bubbles['aspect_ratio_type'] == 'Vakhrushev':
                # Bubble Reynolds number
                Re = (1000 * bubbles['slip_velocity_ratio'] \
                    * np.linalg.norm(Um) * D) / (10**(-3))
                # Morton number
                Mo = (9.81*999*(0.001**4)) / ((1000**2)*(0.07**3))
                # Tadaki number
                Ta = Re*Mo**0.23
                # The aspect ratio (Vakhrushev & Efremov 1970)
                if (Ta < 1.0):
                    E = 1.0
                elif (Ta >= 1.0) & (Ta <= 39.0):
                    E = (0.81 + 0.206*math.tanh(2.0*0.8-math.log10(Ta)))**3
                elif (Ta >= 39.8):
                    E  = 0.24
            B = (D**3 / E)**(1.0/3.0)
            A = E * B
            # Mean bubble frequency
            F = 1.5 * C * np.linalg.norm(Um) / A
            # Number of bubbles
            nb = round(F * tau)
            A = np.ones(nb)*A
            B = np.ones(nb)*B
            # Vector with bubble size (A,B,B), V = 1/6*Pi*A*B*B
            b_size = np.array([A,B,B]).T
        elif bubbles['size_distribution'] == 'lognormal':
            # Mean bubble frequency
            F = 1.5 * C * np.linalg.norm(Um) \
                / math.exp(bubbles['mean_ln_diameter']+0.5*bubbles['sd_ln_diameter']**2)
            # Number of simulated bubbles
            nb = round(F*tau)
            # Sphere-volume-equivalent bubble diameter
            D = np.random.lognormal(bubbles['mean_ln_diameter'],
                                     bubbles['sd_ln_diameter'],
                                     size=nb)
            if bubbles['aspect_ratio_type'] == 'constant':
                E = bubbles['aspect_ratio_value']
                B = (D**3 / E)**(1.0/3.0)
                A = E * B
                b_size = np.array([A,B,B]).T
            elif bubbles['aspect_ratio_type'] == 'Aoyama':
                # Loop over sphere-volume-equivalent bubble diameters
                A = np.ones((nb))*np.nan
                B = np.ones((nb))*np.nan
                for ii,d in enumerate(D):
                    # Bubble Reynolds number
                    Re = (1000 * bubbles['slip_velocity_ratio'] \
                        * np.linalg.norm(Um) * d) / (10**(-3))
                    # Bubble Eotvos number
                    Eo = (999 * 9.81 * d**2) / 0.07
                    # The aspect ratio (Aoyama et al. 2016)
                    E = 1.0/(1.0 + (0.016 * Eo**1.12 * Re))**0.388
                    B[ii] = (d**3 / E)**(1.0/3.0)
                    A[ii] = E * B[ii]
                b_size = np.array([A,B,B]).T
            elif bubbles['aspect_ratio_type'] == 'Vakhrushev':
                # Loop over sphere-volume-equivalent bubble diameters
                A = np.ones((nb))*np.nan
                B = np.ones((nb))*np.nan
                for ii,d in enumerate(D):
                    # Bubble Reynolds number
                    Re = (1000 * bubbles['slip_velocity_ratio'] \
                        * np.linalg.norm(Um) * d) / (10**(-3))
                    # Morton number
                    Mo = (9.81*999*(0.001**4)) / ((1000**2)*(0.07**3))
                    # Tadaki number
                    Ta = Re*Mo**0.23
                    # The aspect ratio (Vakhrushev & Efremov 1970)
                    if (Ta < 1.0):
                        E = 1.0
                    elif (Ta >= 1.0) & (Ta <= 39.0):
                        E = (0.81 + 0.206*math.tanh(2.0*0.8-math.log10(Ta)))**3
                    elif (Ta >= 39.8):
                        E  = 0.24
                    B[ii] = (d**3 / E)**(1.0/3.0)
                    A[ii] = E * B[ii]
                b_size = np.array([A,B,B]).T
    return nb, b_size

def get_mean_bubble_sve_size(flow_properties):
    """Calculate mean sphere-volume-equivalent bubble size D_sve

        Parameters
        ----------
        flow_properties   (dict): A dictionary containing the flow properties

        Returns
        ----------
        d_sve     (float): Mean sphere-volume-equivalent bubble size D_sve
    """

    # Get bubble properties
    bubbles = flow_properties['bubbles']
    if bubbles['size_distribution'] == 'constant':
        d_sve = bubbles['diameter']
    elif bubbles['size_distribution'] == 'lognormal':
        d_sve = math.exp(bubbles['mean_ln_diameter']+0.5*bubbles['sd_ln_diameter']**2)
    return d_sve

def get_max_bubble_sve_size(flow_properties):
    """Calculate the 99.5-percentile sphere-volume-equivalent bubble size d_max

        Parameters
        ----------
        flow_properties   (dict): A dictionary containing the flow properties

        Returns
        ----------
        d_max     (float): 99.5-percentile sphere-volume-equivalent bubble size
    """
    # Get bubble properties
    bubbles = flow_properties['bubbles']
    d_max = math.nan
    if bubbles['size_distribution'] == 'constant':
        d_max = bubbles['diameter']
    elif bubbles['size_distribution'] == 'lognormal':
        # Get 99.5-percentile from log-normal distribution
        d_max = scipy.stats.lognorm.ppf(0.995, s=bubbles['sd_ln_diameter'],scale=math.exp(bubbles['mean_ln_diameter']))
    return d_max

def get_perc_bubble_sve_size(flow_properties, percentile):
    """Calculate the sphere-volume-equivalent bubble size d_percentile for given
    percentile.

        Parameters
        ----------
        flow_properties    (dict): A dictionary containing the flow properties
        percentile        (float): percentile of the bubble size

        Returns
        ----------
        d_percentile      (float): sphere-volume-equivalent bubble size for given percentile
    """
    # Get bubble properties
    bubbles = flow_properties['bubbles']
    d_percentile = math.nan
    if bubbles['size_distribution'] == 'constant':
        d_percentile = bubbles['diameter']
    elif bubbles['size_distribution'] == 'lognormal':
        # Get 99.5-percentile from log-normal distribution
        d_percentile = scipy.stats.lognorm.ppf(percentile, s=bubbles['sd_ln_diameter'],scale=math.exp(bubbles['mean_ln_diameter']))
    return d_percentile

def SBG_fluid_velocity(path, flow_properties, reproducible, progress):
    """Generate stochastic fluid velocity time series including correlation
    between u' and v' as well as autocorrelation in u(t) depending
    on integral time scale.

        Parameters
        ----------
        path      (pathlib.Path): The directory
        flow_properties   (dict): A dictionary containing the flow properties
        reproducible    (string): A string defining the reproducibility
        progress          (bool): A flag to print the progress

        Returns
        ----------
        -
    """
    time1 = time.time()
    # Determine the reproducability of the generated time series
    # If reproducible, then fix seed of random generator
    if reproducible == 'yes':
        np.random.seed(42)

    # Define the time average flow velocity [m/s]
    um = np.asarray(flow_properties['mean_velocity'])
    # Optional factor to use only time average flow velocity without any
    # stochasticity (mainly for testing purposes)
    if "ti_factor" in flow_properties:
        ti_f = flow_properties['ti_factor']
    else:
        ti_f = 1.0
    # Define the standard deviations of the RMS velocities [-]
    sigma = np.empty(3)
    # Calculate the magnitude of the input velocity
    um_mag = np.sqrt(um.dot(um))
    sigma = um_mag * np.asarray(flow_properties['turbulent_intensity']) * ti_f
    # Define the integral time scales of the flow velocity [s]
    T = np.asarray(flow_properties['integral_timescale'])

    # Simulate time series with duration [s]
    duration = flow_properties['duration'];
    # Number of time step for simulating velocity realizations
    dt = 1.0 / flow_properties['realization_frequency']
    # Number of simulated velocity realizations
    n = round(duration / dt)+1;
    # Time vector
    t = np.linspace(0,duration,n);

    # Define the correlation parameters with standardized multivariate Gaussian
    # Zero mean
    mean = [0.0, 0.0, 0.0]
    # Zero Z-Components of covariance in a statistically 2D flow (Pope, 2000)
    rho_shear = flow_properties['shear_stress_corr_coeff']
    # Covariance with diagonal components = 1
    cov = [[1.0, rho_shear[0], rho_shear[1]], \
           [rho_shear[0], 1.0, rho_shear[2]], \
           [rho_shear[1], rho_shear[2], 1.0]]
    
    print('\nGenerating velocity and trajectory time series of the fluid\n')
    # Time series is written in chunks of size n_chunk_max in order to avoid
    # memory overload
    n_chunk_max = 10000
    # Create the H5-file writer
    writer = H5Writer(path / 'flow_data.h5', 'w')
    # Create the velocity data set for the entire time series of length n
    writer.createDataSet('fluid/velocity', (n,3), 'float64')
    writer.createDataSet('fluid/trajectory', (n,3), 'float64')
    writer.writeDataSet('fluid/time', t, 'float64')
    # Initialize with mean flow velocity and write to file
    u_f_old = np.empty((1,3)) * np.NaN
    u_f_old[0,:] = um
    writer.write2DataSet('fluid/velocity', u_f_old, row=0, col=0)
    # Initialize trajectory
    x_f_old = np.zeros((1,3))
    writer.write2DataSet('fluid/trajectory', x_f_old, row=0, col=0)
    kk = 1
    while (kk < n):
        # Define the actual chunk size
        n_chunk = min(n_chunk_max, n-kk)
        # Create multivariate normal random variables for chunk
        R = np.random.multivariate_normal(mean, cov, n_chunk)
        # Initialize the velocity and trajectory arrays
        u_f = np.empty((n_chunk,3)) * np.NaN
        x_f = np.empty((n_chunk,3)) * np.NaN
        # First velocity vector depends on old chunk
        u_f[0,:] = SBG_dt_corr(u_f_old[0,:], um, sigma, T, dt, R[0,:])
        # First trajectory vector depends on old chunk
        x_f[0,:] = x_f_old[0,:] + u_f_old[0,:]*dt
        # Calculate rest of the chunk
        for ii in range(1,n_chunk):
            # Calculate velocity
            u_f[ii,:] = SBG_dt_corr(u_f[ii-1,:], um, sigma, T, dt, R[ii,:])
            # Calculate trajectory
            x_f[ii,:] = x_f[ii-1,:] + u_f[ii-1,:]*dt
        # Write chunk to .h5 file
        writer.write2DataSet('fluid/velocity', u_f, row=kk, col=0)
        writer.write2DataSet('fluid/trajectory', x_f, row=kk, col=0)
        # Save last velocity and trajectory
        u_f_old[0,:] = u_f[-1,:]
        x_f_old[0,:] = x_f[-1,:]
        # Write chunk to .h5 file
        kk+=n_chunk
        # Display progress
        if progress:
            printProgressBar(kk, n, prefix = 'Progress:', suffix = 'Complete', length = 50)

    writer.close()
    # Calculate statistics
    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the fluid velocity
    u_f = reader.getDataSet('fluid/velocity')[:]
    reader.close()

    # Calculate the mean velocity
    u_f_m = u_f.mean(axis=0)
    # Initialize the Reynolds stress tensor
    reynolds_stress = np.empty((len(u_f),3,3))
    for ii in range(0,len(u_f)):
        # Calculate the velocity fluctuation
        u_f_prime = u_f[ii,:]-u_f_m
        # Reynolds stresses as outer product of fluctuations
        reynolds_stress[ii,:,:] = np.outer(u_f_prime, u_f_prime)
    # Average the Reynolds stress tensor
    mean_reynolds_stress = reynolds_stress.mean(axis=0)
    # Calculate turbulent intensity with regard to mean x-velocity
    turbulent_intensity = np.sqrt(np.array([
            mean_reynolds_stress[0,0], \
            mean_reynolds_stress[1,1], \
            mean_reynolds_stress[2,2], \
            ])) / np.sqrt(u_f_m.dot(u_f_m))
    # Write the statistics
    writer = H5Writer(path / 'flow_data.h5', 'a')
    writer.writeDataSet('fluid/mean_velocity', \
        u_f_m, 'float64')
    writer.writeDataSet('fluid/reynold_stresses', \
        mean_reynolds_stress, 'float64')
    writer.writeDataSet('fluid/turbulent_intensity', \
        turbulent_intensity, 'float64')
    flow_props = str(flow_properties)
    writer.writeStringDataSet('.flow_properties', flow_props)
    writer.close()
    time2 = time.time()
    print(f'Successfully written fluid velocity time series and trajectory')
    print(f'Finished in {time2-time1:.2f} seconds\n')

def interpolate_time_series(t_old, value_old, t_new):
    result = np.empty((t_new.shape[0], value_old.shape[1]))
    
    for i in range(value_old.shape[1]):
        result[:, i] = np.interp(t_new, t_old, value_old[:, i])
    
    return result

def get_bubble_properties(at,ax,t_f,x_f,u_f,control_volume_size):
    """ Returns the bubble properties.

        Parameters
        ----------
        at                  (np.array): The bubble arrival time
        ax                  (np.array): The bubble arrival location
        t_f                 (np.array): Fluid time vector
        x_f                 (np.array): Fluid trajectory vector
        u_f                 (np.array): Fluid velocity vector
        control_volume_size (np.array): The control volume size (x,y,z)

        Returns
        ----------
        bubble_props (dict): A dictionary with bubble properties
    """
    # Find the time index
    idxs = np.where(t_f <= at)
    at_idx = idxs[0][np.abs(t_f[idxs] - at).argmin()]

    # Compute the array slice indexes for subsequent operations
    slice_from_at_idx = x_f[at_idx:, 0] - x_f[at_idx, 0]
    condition = slice_from_at_idx <= control_volume_size[0]

    et_idx = at_idx + np.sum(condition) + 2
    if et_idx >= len(t_f):
        et_idx = len(t_f) - 1

    # Define time series slice for bubble in control volume
    t_p = t_f[at_idx:(et_idx + 1)]

    # Interpolate arrival location
    x_p_at = interpolate_time_series(t_p, x_f[at_idx:(et_idx + 1)], np.array([at]))

    # Calculate locations
    x_at = np.array([-control_volume_size[0] / 2.0, ax[1], ax[2]])
    x_p = x_at - (x_p_at - x_f[at_idx]) + (x_f[at_idx:(et_idx + 1)] - x_f[at_idx])

    # Exit location and time
    x_et = interpolate_time_series(x_p[:, 0], x_p, np.array([control_volume_size[0] / 2.0]))
    et = np.interp(np.array([control_volume_size[0] / 2.0]), x_p[:, 0], t_p)

    # Get bubble velocity (assuming no-slip)
    u_p = u_f[at_idx:(et_idx + 1)]
    # Return the bubble properties
    bubble_props = {
        "arrival_location":     x_at,
        "exit_location":        x_et,
        "arrival_time":         at,
        "exit_time":            et,
        "time":                 t_p,
        "trajectory":           x_p,
        "velocity":             u_p,
        "mean_velocity":        np.ones((1, 3)) * np.nan,
        "mean_reynolds_stress": np.ones((1, 6)) * np.nan,
        "turbulent_intensity":  np.ones((1, 3)) * np.nan
    }

    return bubble_props

def SBG_simulate_bubble_piercing(
        bubble_id,
        t_f,
        x_f,
        u_f,
        control_volume_size,
        at,
        ax,
        b_size,
        dt_probe,
        sensor_delta,
        start_idx_probe,
        c_probe=np.array([])):

    """ Simulate bubble trajectories, track bubble trajectories with respect to
        to sensor locations and store during which time indices the bubble is 
        pierced by the probe.

        Parameters
        ----------
        bubble_id                (int): The bubble ID
        t_f                 (np.array): The fluid time
        x_f                 (np.array): The fluid trajectory
        u_f                 (np.array): The fluid velocity
        control_volume_size (np.array): Control volume dimensions
        at                  (np.array): Arrival time of the bubble
        ax                  (np.array): Random arrival location of bubbles
        b_size              (np.array): The bubble size
        dt_probe            (np.array): Sampling time step of the probe
        sensor_delta            (list): Locations of each individual sensor
        start_idx_probe          (int): Start index of probe time vector
        c_probe             (np.array): center location of the probe

        Returns
        ----------
        results (list): [0]: time indices of the signal where bubble is pierced
                        [1]: the bubble properties (time,trajectory,etc.)
    """

    # get bubble properties
    bubble_props = get_bubble_properties(at, ax, t_f, x_f, u_f, control_volume_size)
    bubble_props['id'] = bubble_id
    t_p = bubble_props['time']
    x_p = bubble_props['trajectory']
    # Define the estimated time frame
    t_min, t_max = t_p[0], t_p[-1]
    # Get probe sampling times that lie within the estimated timeframe
    t_probe_kk_start = round_up(t_min, dt_probe)
    idx_start = round(t_probe_kk_start/dt_probe)
    t_probe_kk_end = round_down(t_max, dt_probe)
    t_probe_kk = np.linspace(t_probe_kk_start, t_probe_kk_end, round((t_probe_kk_end-t_probe_kk_start)/dt_probe)+1)
    signal_indices = {}
    if len(t_probe_kk) > 0:
        # Resample the trajectory to the sampling times of the probe
        x_p_resampled = interpolate_time_series(t_p, x_p, t_probe_kk)
        # Get probe locations that lie within the estimated timeframe
        c_probe_kk = c_probe[(idx_start-start_idx_probe):(idx_start-start_idx_probe+len(t_probe_kk)),:] if len(c_probe) > 0 else np.zeros((len(t_probe_kk), 3))
        abc = b_size / 2.0

        min_idx, max_idx = len(t_probe_kk)+1, -1
        for idx, delta in sensor_delta.items():
            # Check if ellipsoid is pierced by sensor idx
            # Vectorized calculation for standard equation: (x/a)2 + (y/b)2 + (z/c)2 = 1
            c_probe_shifted = c_probe_kk + delta  
            diff = c_probe_shifted - x_p_resampled
            normalized_diff = diff / abc
            radius = np.sum(normalized_diff**2, axis=1)
            idxs = np.where(radius <= 1)
            signal_indices[idx] = idxs
            if idxs[0].size:
                min_idx, max_idx = min(min_idx, idxs[0].min()), max(max_idx, idxs[0].max())

        if (max_idx - min_idx) >= 0:
            # Resample the velocity to the sampling times of the probe
            u_p_resampled = interpolate_time_series(t_p, bubble_props['velocity'], t_probe_kk)
            u_p_interaction = u_p_resampled[min_idx:(max_idx+1)]

            # Calculate the statistics
            # Calculate mean velocity
            u_mean = u_p_interaction.mean(axis=0)

            # Vectorized Reynolds stress tensor calculation
            u_prime = u_p_interaction - u_mean
            reynolds_stress_tensor = np.einsum('ij,ik->ijk', u_prime, u_prime) / len(u_p_interaction)

            # Calculate the mean Reynolds stress tensor across all time steps
            mean_reynolds_stress_tensor = reynolds_stress_tensor.mean(axis=0)

            # Extract the desired elements
            mean_reynolds_stress = [mean_reynolds_stress_tensor[0,0], 
                                    mean_reynolds_stress_tensor[1,0], 
                                    mean_reynolds_stress_tensor[1,1], 
                                    mean_reynolds_stress_tensor[2,0], 
                                    mean_reynolds_stress_tensor[2,1], 
                                    mean_reynolds_stress_tensor[2,2]]

            # Calculate turbulent intensity with regard to mean x-velocity
            turbulent_intensity = np.sqrt([mean_reynolds_stress[0], mean_reynolds_stress[2], mean_reynolds_stress[5]]) / np.linalg.norm(u_mean)

            bubble_props.update({"mean_velocity": u_mean, "mean_reynolds_stress": mean_reynolds_stress, "turbulent_intensity": turbulent_intensity})

        return [idx_start, signal_indices, bubble_props]

    else:
        for idx in sensor_delta.keys():
            signal_indices[idx] = np.array([])

        return [idx_start, signal_indices, bubble_props]


def get_virtual_distance(pos1, pos2, control_volume_size):
    y_min, y_max = np.min([pos1[1], pos2[1]]), np.max([pos1[1], pos2[1]])
    z_min, z_max = np.min([pos1[2], pos2[2]]), np.max([pos1[2], pos2[2]])

    dx = abs(pos2[0] - pos1[0])
    dy = (control_volume_size[1]/2.0 - y_max) + (y_min + control_volume_size[1]/2.0)
    dz = (control_volume_size[2]/2.0 - z_max) + (z_min + control_volume_size[2]/2.0)

    return np.sqrt(dx**2 + dy**2 + dz**2)

def bubbles_overlap(pos1, pos2, d1, d2, control_volume_size):
    actual_distance = min(np.linalg.norm(pos1 - pos2), get_virtual_distance(pos1, pos2, control_volume_size))
    necessary_distance = (d1 + d2) / 2.0
    return actual_distance < necessary_distance

def place_bubbles_without_overlap(
        random_arrival_locations,
        arrival_times,
        b_size,
        control_volume_size,
        nb,
        nb_check_max,
        velocity
        ):

    """
        Place random bubble arrival locations in a way that there is no overlap

        Parameters
        ----------
        random_arrival_locations (list): Random arrival locations of bubbles
        arrival_times        (np.array): Arrival times of the bubbles
        b_size               (np.array): Bubble sizes
        control_volume_size  (np.array): Control volume dimensions
        nb                        (int): Number of bubbles
        nb_check_max              (int): The fluid velocity
        velocity                (float): The fluid velocity

        Returns
        ----------
        results (np.array): Random arrival locations of bubbles without overlap
    """

    arrival_locations = np.zeros((nb, 2))
    arrival_locations[0, :] = random_arrival_locations.pop(0)
    touch_cnt = 0
    still_touch_cnt = 0

    for ii in range(1, nb):
        nb_check = min(ii, nb_check_max)

        n_iter = 0
        max_iter = len(random_arrival_locations) - 1

        while n_iter <= max_iter:
            overlap = False
            random_pos = random_arrival_locations[n_iter]
            pos2 = np.array([0.0, random_pos[0], random_pos[1]])

            for jj in range(nb_check):
                idx = ii - jj - 1
                pos_jj = np.array([velocity * (arrival_times[ii] - arrival_times[idx]),
                                   arrival_locations[idx, 0],
                                   arrival_locations[idx, 1]])

                if bubbles_overlap(pos_jj, pos2, b_size[idx, 0], b_size[ii, 0], control_volume_size):
                    overlap = True
                    touch_cnt += 1
                    break

            if not overlap or n_iter == max_iter:
                if overlap:
                    still_touch_cnt += 1
                random_arrival_locations.pop(n_iter)
                arrival_locations[ii, :] = pos2[1:]
                break
            n_iter += 1

    return arrival_locations


def down_sample_fluid_time_series(t_f,u_f,x_f,f_u):
    n = round((t_f[-1] - t_f[0]) * f_u)+1
    # Time vector
    t_new = np.linspace(t_f[0],t_f[-1],n)
    t_new = np.sort(np.unique(np.append(t_new,t_f)))
    u_f = interpolate_time_series(t_f,u_f,t_new)
    x_f = interpolate_time_series(t_f,x_f,t_new)
    t_f = t_new
    return t_f,u_f,x_f

def get_chunk_times(T,T_chunk,n_chunk):
    chunk_times = np.zeros((n_chunk,2))
    for ii in range(n_chunk):
        t_start = max(0.0, ii*T_chunk)
        t_end = min(T, (ii+1)*T_chunk)
        chunk_times[ii,:] = np.array([t_start,t_end])
    return chunk_times

def get_chunk_times_buffered(T,T_chunk,n_chunk,T_buffer):
    chunk_times_buffered = np.zeros((n_chunk,2))
    for ii in range(n_chunk):
        t_start_buffer = max(0.0, ii*T_chunk-T_buffer)
        t_end_buffer = min(T, (ii+1)*T_chunk+T_buffer)
        chunk_times_buffered[ii,:] = np.array([t_start_buffer,t_end_buffer])
    return chunk_times_buffered

def get_chunk_indices(n_chunk,chunk_times,t):
    chunk_idx = np.zeros((n_chunk,2)).astype('int')
    for ii in range(n_chunk):
        idx = np.array([])
        if ii < (n_chunk-1):
            # get indices
            idx = np.where((chunk_times[ii,0] <= t) & (t < chunk_times[ii,1]))[0]
        else:
            # get indices
            idx = np.where((chunk_times[ii,0] <= t) & (t <= chunk_times[ii,1]))[0]
        chunk_idx[ii,:] = np.array([int(np.min(idx)),int(np.max(idx))]).astype('int')
    return chunk_idx

def SBG_signal(
    path,
    flow_properties,
    probe,
    reproducible,
    uncertainty,
    progress,
    nthreads):

    """ Generate the bubble field, simulate bubble movement with respect to 
        probe and record the signal.

        Parameters
        ----------
        path      (pathlib.Path): The directory
        flow_properties   (dict): A dictionary containing the flow properties
        probe             (dict): A dictionary containing the probe properties
        reproducible    (string): A string defining the reproducibility
        progress          (bool): A flag to print the progress
        nthreads           (int): Number of jobs to start for parallel runs

        Returns
        ----------
        -
    """

    time1 = time.time()
    # Determine the reproducability of the generated time series
    # If reproducible, the fix seed of random generator
    if reproducible == 'yes':
        np.random.seed(42)
        random.seed(42)

    # Define some variables for easier use
    T = float(flow_properties['duration'])
    T_chunk = 2.0
    T_buffer = 0.5
    n_chunk = math.ceil(T/T_chunk)
    um = np.asarray(flow_properties['mean_velocity'])
    C = flow_properties['void_fraction']

    # Gather the probe information (id and relative location [=delta])
    n_sensors = len(probe['sensors'])
    # Initialize an empty dictionary
    sensor_delta = {}
    for sensor in probe['sensors']:
        # fill dictionary:  id -> relative_location
        sensor_delta[sensor['id']] = np.asarray(sensor['relative_location'])
    # Get an estimate of the maximum probe dimension
    sensors = probe['sensors']
    min_range = np.array([LARGENUMBER,LARGENUMBER,LARGENUMBER])
    max_range = np.array([LARGENEGNUMBER,LARGENEGNUMBER,LARGENEGNUMBER])

    for sensor in sensors:
        for ii in range(0,3):
            min_range[ii] = min(min_range[ii], sensor['relative_location'][ii])
            max_range[ii] = max(max_range[ii], sensor['relative_location'][ii])
    
    max_probe_size = max_range - min_range


    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the time vector
    t_f = reader.getDataSet('fluid/time')[:]
    # Read the fluid velocity
    u_f = reader.getDataSet('fluid/velocity')[:]
    # Read the fluid velocity
    x_f = reader.getDataSet('fluid/trajectory')[:]
    reader.close()

    # if necessary, downsample velocity and trajectory
    # minimum frequency to have at approx. five trajectory values
    # while bubble travels between tips
    travel_time = max_probe_size[0]/um[0]
    f_u = 1.0/travel_time*5.0
    if f_u >= flow_properties['realization_frequency']:
        t_f,u_f,x_f = down_sample_fluid_time_series(t_f,u_f,x_f,f_u)

    # Determine control volume (CV) size
    # get the mean sphere-volume-equivalent bubble diameter
    d = get_mean_bubble_sve_size(flow_properties)
    V_b = math.pi/6.0*d**3
    # get max bubble size
    d_max = get_max_bubble_sve_size(flow_properties)
    control_volume_size = np.zeros(3)
    # length
    control_volume_size[0] = 2*d_max + max_probe_size[0]
    # width
    control_volume_size[1] = 2*d + max(max_probe_size[1],max_probe_size[2])
    control_volume_size[2] = 2*d + max(max_probe_size[1],max_probe_size[2])

    # cross-sectional CV area at inflow
    control_volume_area = control_volume_size[1]*control_volume_size[2]

    # Calculate bubble frequency
    # F = C*U*CV_a/V_b
    F = C*np.linalg.norm(um)*(control_volume_area)/V_b
    # Create the bubbles (number of bubbles and array with bubble size)
    nb, b_size = create_bubbles(flow_properties, F)
    # correct number of bubbles to get correct void fraction in case of lognormal distributions
    chord_times = np.zeros(nb)
    if flow_properties['bubbles']['size_distribution'] == 'lognormal':
        for ii in range(0,nb):
            r = math.sqrt(random.uniform(-control_volume_size[1]/2.0,control_volume_size[1]/2.0)**2+
                random.uniform(-control_volume_size[2]/2.0,control_volume_size[2]/2.0)**2)
            R = b_size[ii,0]/2
            if r < R:
                chord_times[ii] = math.sqrt(R**2-r**2)*2/np.linalg.norm(um)
        cumulative_chord_times = np.cumsum(chord_times)
        C_eff = cumulative_chord_times/T
        nb = len(C_eff[C_eff <= C])
        F = nb/T
        b_size = b_size[0:nb,:]

    # Inter-arrival time (time between two bubbles)
    # ASSUMPTION: equally spaced (in time) bubble distribution
    # iat = 1.0/F - db/np.linalg.norm(um)
    inter_arrival_time = 1.0/F
    # Initialize arrival time (at) vector
    arrival_times = np.linspace(0,(nb-1)*inter_arrival_time,nb)
    arrival_times = arrival_times[arrival_times <= t_f[-2]]
    # delete not needed large arrays
    del t_f,u_f,x_f
    nb = len(arrival_times)
    # create random bubble arrival locations
    low = [-control_volume_size[1] / 2.0, -control_volume_size[2] / 2.0]
    high = [control_volume_size[1] / 2.0, control_volume_size[2] / 2.0]
    random_arrival_locations = np.random.uniform(low=low, high=high, size=(nb, 2))
    print('\nRandomly placing bubbles.\n')
    n_split = max(int(nb/10000),1)
    nb_check_max = math.ceil(d_max / (inter_arrival_time * um[0]))
    random_arrival_locations = np.array_split(random_arrival_locations,n_split)
    arrival_times_split = np.array_split(arrival_times,n_split)
    b_size_split = np.array_split(b_size,n_split)
    arrival_locations_parallel = Parallel(n_jobs=nthreads,backend='multiprocessing') \
        (delayed(place_bubbles_without_overlap)(
                                        list(random_arrival_locations[kk]),
                                        arrival_times_split[kk],
                                        b_size_split[kk],
                                        control_volume_size,
                                        len(arrival_times_split[kk]),
                                        nb_check_max,
                                        um[0]) for kk in range(0,n_split))
    arrival_locations = arrival_locations_parallel[0]
    for kk in range(1,n_split):
        arrival_locations = np.concatenate((arrival_locations,arrival_locations_parallel[kk]), axis=0)
    # set x-value of arrival locations to start of control volume
    arrival_locations = np.concatenate((-control_volume_size[0]*np.ones((nb,1)),arrival_locations), axis=1)
    arrival_times = arrival_times.reshape((nb,1))
    # Create the H5-file writer
    writer = H5Writer(path / 'flow_data.h5', 'a')
    # Write the bubble properties
    writer.writeDataSet('bubbles/arrival_times', arrival_times, 'float64')
    writer.writeDataSet('bubbles/arrival_locations', arrival_locations, 'float64')
    writer.writeDataSet('bubbles/size', b_size, 'float64')
    writer.close()
    del arrival_locations,random_arrival_locations
    time2 = time.time()
    print(f'Finished bubble placement. Current runtime: {time2-time1:.2f} seconds\n')

    print('\nInitialize the signal.\n')
    # The sampling frequency
    f_probe = probe['sampling_frequency']
    # Sampling time step
    dt_probe = 1.0 / f_probe
    # Number sampling time steps
    n_probe = math.ceil(T / dt_probe);
    # Time vector
    t_probe = np.linspace(0, n_probe*dt_probe, n_probe+1);
    # Initialize signals with zeros
    signal = np.zeros((n_probe+1, n_sensors)).astype('uint8')
    # Create the H5-file writer
    writer = H5Writer(path / 'binary_signal.h5', 'w')
    # Write the time vector
    writer.writeDataSet('time', t_probe, 'float64')
    writer.writeDataSet('signal', signal, 'u1')
    ds_sig = writer.getDataSet('signal')
    ds_sig.attrs['sensor_id'] = list(sensor_delta.keys())
    writer.close()
    time2 = time.time()
    print(f'Finished signal initialization. Current runtime: {time2-time1:.2f} seconds\n')

    print('\nTracking bubbles with respect to probe.\n')

    if 'VIBRATION' in uncertainty:
        # Initialize the probe location
        c_probe = np.zeros((n_probe+1,3))
        # amplitudes of vibrations in all directions
        vib_amps = uncertainty['VIBRATION']['amplitudes']
        # frequencies of vibrations in all directions
        vib_freqs = uncertainty['VIBRATION']['frequencies']
        # probe location as function of time: f(t) = A*sin(2*Pi*f*t)
        for ii in range(0,3):
            c_probe[:,ii] = c_probe[:,ii] + vib_amps[ii] * np.sin(2.0 * math.pi * vib_freqs[ii] * t_probe)

        # Write probe location over time
        writer = H5Writer(path / 'flow_data.h5', 'a')
        # Create the velocity data set for the entire time series of length n
        writer.writeDataSet('probe/time', t_probe, 'float64')
        writer.writeDataSet('probe/location', c_probe, 'float64')
        writer.close()

    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the time vector
    t_f = reader.getDataSet('fluid/time')[:]
    reader.close()

    # Get start and end indices all chunks
    chunk_times = get_chunk_times(T,T_chunk,n_chunk)
    chunk_times_buffered = get_chunk_times_buffered(T,T_chunk,n_chunk,T_buffer)
    chunk_indice_probe = get_chunk_indices(n_chunk,chunk_times_buffered,t_probe)
    chunk_indice_fluid = get_chunk_indices(n_chunk,chunk_times_buffered,t_f)
    chunk_indice_bubbles = get_chunk_indices(n_chunk,chunk_times,arrival_times)
    del signal, t_f, arrival_times, t_probe
    # Initialize some datasets
    # Create the H5-file writer
    writer = H5Writer(path / 'flow_data.h5', 'a')
    # Write the bubble properties
    writer.createDataSet('bubbles/exit_times', (nb,1), 'float64')
    writer.createDataSet('bubbles/exit_locations', (nb,3), 'float64')
    writer.createDataSet('bubbles/mean_velocity', (nb,3), 'float64')
    writer.createDataSet('bubbles/reynold_stresses', (nb,6), 'float64')
    writer.createDataSet('bubbles/turbulent_intensity', (nb,3), 'float64')
    writer.createDataSet('bubbles/chord_times', (nb,len(sensors)), 'float64')
    writer.createDataSet('bubbles/chord_lengths', (nb,len(sensors)), 'float64')
    writer.createDataSet('bubbles/pierced_bubbles', (nb,len(sensors)), 'u1')
    writer.close()

    # Track bubbles in chunks to reduce memory use
    n_bubbles_pierced = 0
    cumulative_chord_times = 0.0
    cumulative_chord_times_overlap = 0.0
    n_tot = 0
    for ii in range(n_chunk):
        # Create a H5-file reader
        start_idx_fluid = chunk_indice_fluid[ii,0]
        end_idx_fluid = chunk_indice_fluid[ii,1]
        reader = H5Reader(path / 'flow_data.h5')
        # Read the time vector
        t_f_chunk = reader.getRowsFromDataSet('fluid/time', start_idx_fluid, end_idx_fluid)[:]
        # Read the fluid velocity
        u_f_chunk = reader.getRowsFromDataSet('fluid/velocity', start_idx_fluid, end_idx_fluid)[:]
        # Read the fluid trajectory
        x_f_chunk = reader.getRowsFromDataSet('fluid/trajectory', start_idx_fluid, end_idx_fluid)[:]
        # if necessary, downsample velocity and trajectory
        # minimum frequency to have at approx. five trajectory values
        # while bubble travels between tips
        travel_time = max_probe_size[0]/um[0]
        f_u = 1.0/travel_time*5.0
        if f_u >= flow_properties['realization_frequency']:
            t_f_chunk,u_f_chunk,x_f_chunk = down_sample_fluid_time_series(t_f_chunk,u_f_chunk,x_f_chunk,f_u)
        start_idx_bubbles = chunk_indice_bubbles[ii,0]
        end_idx_bubbles = chunk_indice_bubbles[ii,1]
        arrival_times_chunk = reader.getRowsFromDataSet('bubbles/arrival_times', start_idx_bubbles, end_idx_bubbles)[:]
        arrival_locations_chunk = reader.getRowsFromDataSet('bubbles/arrival_locations', start_idx_bubbles, end_idx_bubbles)[:]
        b_size_chunk = reader.getRowsFromDataSet('bubbles/size', start_idx_bubbles, end_idx_bubbles)[:]
        n_tot += len(arrival_locations_chunk)
        reader.close()
        nb_chunk = len(arrival_times_chunk)
        bubble_id_chunk = np.linspace(0,nb_chunk-1,nb_chunk).astype('int')
        start_idx_probe = chunk_indice_probe[ii,0]
        end_idx_probe = chunk_indice_probe[ii,1]
        if 'VIBRATION' in uncertainty:
            reader = H5Reader(path / 'flow_data.h5')
            c_probe_chunk = reader.getRowsFromDataSet('probe/location', start_idx_probe, end_idx_probe)[:]
            reader.close()
            results = Parallel(n_jobs=nthreads,backend='multiprocessing') \
                (delayed(SBG_simulate_bubble_piercing)(bubble_id_chunk[kk], 
                                                       t_f_chunk,
                                                       x_f_chunk,
                                                       u_f_chunk,
                                                       control_volume_size,
                                                       arrival_times_chunk[kk],
                                                       arrival_locations_chunk[kk],
                                                       b_size_chunk[kk,:],
                                                       dt_probe,
                                                       sensor_delta,
                                                       start_idx_probe,
                                                       c_probe_chunk
                                                       ) for kk in range(0,nb_chunk))
        else:
            results = Parallel(n_jobs=nthreads,backend='multiprocessing') \
                (delayed(SBG_simulate_bubble_piercing)(bubble_id_chunk[kk],
                                                       t_f_chunk,
                                                       x_f_chunk,
                                                       u_f_chunk,
                                                       control_volume_size,
                                                       arrival_times_chunk[kk],
                                                       arrival_locations_chunk[kk],
                                                       b_size_chunk[kk,:],
                                                       dt_probe,
                                                       sensor_delta,
                                                       start_idx_probe
                                                       ) for kk in range(0,nb_chunk))
        time2 = time.time()
        print(f'Finished bubble tracking for chunk {ii+1} out of {n_chunk}. Current runtime: {time2-time1:.2f} seconds\n')
        print('\nWriting signal and results.\n')
        # Initialize bubble properties
        arrival_times_bubble = np.zeros((nb_chunk,1))*np.nan
        arrival_locations_bubble = np.zeros((nb_chunk,3))*np.nan
        exit_times_bubble = np.zeros((nb_chunk,1))*np.nan
        exit_locations_bubble = np.zeros((nb_chunk,3))*np.nan
        mean_velocities_bubble = np.zeros((nb_chunk,3))*np.nan
        mean_reynolds_stresses_bubble = np.zeros((nb_chunk,6))*np.nan
        turbulent_intensities_bubble = np.zeros((nb_chunk,3))*np.nan
        chord_times_bubble = np.zeros((nb_chunk,len(sensors)))*np.nan
        chord_lengths_bubble = np.zeros((nb_chunk,len(sensors)))*np.nan
        pierced_bubbles = np.zeros((nb_chunk,len(sensors)),dtype='uint8')
        # Create reader for signal
        reader = H5Reader(path / 'binary_signal.h5')
        # Read the signal
        signal = reader.getRowsFromDataSet('signal', start_idx_probe, end_idx_probe)[:]
        reader.close()


        for jj,bubble in enumerate(results):
            # get signal start time (to get proper indices)
            idx_start = bubble[0]
            # get indices of signal for which the signal must be changed to 1
            signal_indices = bubble[1]
            # get bubble properties
            bubble_props = bubble[2]
            bid = bubble_props["id"]
            # Determine the row in the signal time series where to write the signal
            # write the signal for each bubble based on time indices when
            # when the bubble was in contact with a sensor
            for sensor_idx in signal_indices:
                # set those indices to 1
                if len(signal_indices[sensor_idx][0]) > 0:
                    if sensor_idx == 0:
                        cumulative_chord_times += float(len(signal_indices[sensor_idx][0]))/float(f_probe)
                        overlap = np.sum(signal[(signal_indices[sensor_idx][0]+idx_start-start_idx_probe),sensor_idx] == 1)
                        cumulative_chord_times_overlap += float(overlap)/float(f_probe)
                        n_bubbles_pierced+=1
                    signal[(signal_indices[sensor_idx][0]+idx_start-start_idx_probe),sensor_idx] = 1
                    pierced_bubbles[bid,sensor_idx] = 1
                    chord_times_bubble[bid,sensor_idx] = float(len(signal_indices[sensor_idx][0]))/float(f_probe)
                    chord_lengths_bubble[bid,sensor_idx] = float(len(signal_indices[sensor_idx][0]))/float(f_probe)*bubble_props["mean_velocity"][0]
            arrival_times_bubble[bid] = bubble_props["arrival_time"]
            arrival_locations_bubble[bid,:] = bubble_props["arrival_location"]
            exit_times_bubble[bid] = bubble_props["exit_time"]
            exit_locations_bubble[bid,:] = bubble_props["exit_location"]
            mean_velocities_bubble[bid,:] = bubble_props["mean_velocity"]
            mean_reynolds_stresses_bubble[bid,:] = bubble_props["mean_reynolds_stress"]
            turbulent_intensities_bubble[bid,:] = bubble_props["turbulent_intensity"]

        # Create the H5-file writer
        writer = H5Writer(path / 'binary_signal.h5', 'a')
        # Write the time vector
        writer.write2DataSet('signal', signal, row=start_idx_probe, col=0)
        writer.close()
        del signal
        # Create the H5-file writer
        writer = H5Writer(path / 'flow_data.h5', 'a')
        # Write the bubble properties
        writer.write2DataSet('bubbles/arrival_times', arrival_times_bubble, row=start_idx_bubbles, col=0)
        writer.write2DataSet('bubbles/arrival_locations', arrival_locations_bubble, row=start_idx_bubbles, col=0)
        writer.write2DataSet('bubbles/exit_times', exit_times_bubble, row=start_idx_bubbles, col=0)
        writer.write2DataSet('bubbles/exit_locations', exit_locations_bubble, row=start_idx_bubbles, col=0)
        writer.write2DataSet('bubbles/mean_velocity', mean_velocities_bubble, row=start_idx_bubbles, col=0)
        writer.write2DataSet('bubbles/reynold_stresses', mean_reynolds_stresses_bubble, row=start_idx_bubbles, col=0)
        writer.write2DataSet('bubbles/turbulent_intensity', turbulent_intensities_bubble, row=start_idx_bubbles, col=0)
        writer.write2DataSet('bubbles/chord_times', chord_times_bubble, row=start_idx_bubbles, col=0)
        writer.write2DataSet('bubbles/chord_lengths', chord_lengths_bubble, row=start_idx_bubbles, col=0)
        writer.close()

    time2 = time.time()
    print(f'Finished processing all chunks in {time2-time1:.2f} seconds\n')
    print(f'Running final calulations')
    # Create reader for signal
    reader = H5Reader(path / 'binary_signal.h5')
    # Read the signal
    signal = reader.getDataSet('signal')[:]
    reader.close()
    C_bubbles_overlapped = cumulative_chord_times_overlap/T
    C_bubbles_pierced = cumulative_chord_times/T
    F_bubbles_pierced = n_bubbles_pierced/T
    C_measured = np.mean(signal[:,0])
    # Create the H5-file writer
    writer = H5Writer(path / 'flow_data.h5', 'a')
    # Write the bubble properties
    writer.writeDataSet('bubbles/pierced_bubbles', pierced_bubbles, 'u1')
    writer.writeDataSet('bubbles/pierced_bubble_frequency', np.array([F_bubbles_pierced]), 'float64')
    writer.writeDataSet('bubbles/pierced_bubble_void_fraction', np.array([C_bubbles_pierced]), 'float64')
    writer.writeDataSet('bubbles/pierced_overlapped_bubble_void_fraction', np.array([C_bubbles_overlapped]), 'float64')
    writer.writeDataSet('bubbles/measured_void_fraction', np.array([C_measured]), 'float64')
    writer.close()
    time2 = time.time()
    print(f'Successfully generated the signal')
    print(f'Finished in {time2-time1:.2f} seconds\n')