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
        x_f[0,:] = x_f_old[0,:] + (u_f[0,:]+u_f_old[0,:])/2.0*dt
        # Calculate rest of the chunk
        for ii in range(1,n_chunk):
            # Calculate velocity
            u_f[ii,:] = SBG_dt_corr(u_f[ii-1,:], um, sigma, T, dt, R[ii,:])
            # Calculate trajectory
            x_f[ii,:] = x_f[ii-1,:] + (u_f[ii,:]+u_f[ii-1,:])/2.0*dt;
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

def interpolate_time_series(t_old,value_old,t_new):
    x_interp = np.interp(t_new, t_old, value_old[:,0])
    y_interp = np.interp(t_new, t_old, value_old[:,1])
    z_interp = np.interp(t_new, t_old, value_old[:,2])
    return np.array([x_interp,y_interp,z_interp]).transpose()

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

    # time index
    at_idx = find_nearest_smaller_idx(t_f,at)
    # get exit time index of bubble
    et_idx = at_idx + len(x_f[at_idx::,0][(x_f[at_idx::,0]-x_f[at_idx,0]) <= control_volume_size[0]]) + 2
    if et_idx >= len(t_f):
        et_idx = len(t_f)-1
    # bubble time in control volume
    t_p = t_f[at_idx:(et_idx+1)]
    # interpolate location at arrival time
    x_p_at = interpolate_time_series(t_f[at_idx:(et_idx+1)],x_f[at_idx:(et_idx+1)],np.array([at]))
    # arrival location
    x_at = np.array([-control_volume_size[0]/2.0,ax[0],ax[1]])
    x_p_0 = x_f[at_idx:(et_idx+1)] - x_f[at_idx]
    x_p = (x_at - (x_p_at - x_f[at_idx])) + x_p_0
    # exit location
    x_et = interpolate_time_series(x_p[:,0],x_p,np.array([control_volume_size[0]/2.0]))
    et = np.interp(np.array([control_volume_size[0]/2.0]), x_p[:,0], t_p)
    # get bubble velocity
    # assuming no-slip:
    # bubble velocity = fluid velocity after arrival time
    u_p = u_f[at_idx:(et_idx+1)]
    # Store bubble properties
    bubble_props = {
        "arrival_location":     x_at,
        "exit_location":        x_et,
        "arrival_time":         at,
        "exit_time":            et,
        "time":                 t_p,
        "trajectory":           x_p,
        "velocity":             u_p,
        "mean_velocity":        np.ones((1,3))*np.nan,
        "mean_reynolds_stress": np.ones((1,6))*np.nan,
        "turbulent_intensity":  np.ones((1,3))*np.nan
    }
    return bubble_props

def SBG_simulate_bubble_piercing(kk,
        t_f,
        x_f,
        u_f,
        control_volume_size,
        arrival_times,
        random_arrival_loc,
        b_size,
        dt_probe,
        sensor_delta,
        progress,
        nb,
        c_probe=np.array([])):

    """ Simulate bubble trajectories, track bubble trajectories with respect to
        to sensor locations and store during which time indices the bubble is 
        pierced by the probe.

        Parameters
        ----------
        t_f                 (np.array): The directory
        x_f                 (np.array): A dictionary containing the flow properties
        u_f                 (np.array): A dictionary containing the probe properties
        control_volume_size (np.array): A string defining the reproducibility
        arrival_times       (np.array): A flag to print the progress
        random_arrival_loc  (np.array): Random arrival locations of bubbles
        b_size              (np.array): The bubble size
        dt_probe            (np.array): Sampling time step of the probe
        c_probe             (np.array): center location of the probe
        sensor_delta            (list): Locations of each individual sensor
        progress                (bool): A flag to print the progress
        nb                       (int): Number of bubbles

        Returns
        ----------
        results (list): [0]: time indices of the signal where bubble is pierced
                        [1]: the bubble properties (time,trajectory,etc.)
    """

    # get bubble arrival time
    at = arrival_times[kk]
    # arrival location
    ax = random_arrival_loc[kk,:]
    # get bubble properties
    bubble_props = get_bubble_properties(at,ax,t_f,x_f,u_f,control_volume_size)
    t_p = bubble_props['time']
    x_p = bubble_props['trajectory']
    t_min = t_p[0]
    t_max = t_p[-1]
    # Get probe sampling times that lie within the estimated timeframe
    t_probe_kk_start = round_up(t_min,dt_probe)
    idx_start = round(t_probe_kk_start/dt_probe)
    t_probe_kk_end = round_down(t_max,dt_probe)
    t_probe_kk = np.linspace(t_probe_kk_start,t_probe_kk_end,round((t_probe_kk_end-t_probe_kk_start)/dt_probe)+1)
    signal_indices = {}
    # Check if number of samples lies inside the timeframe is larger than 0
    if len(t_probe_kk) > 0:
        # Resample the trajectory to the sampling times of the probe
        x_p_resampled = interpolate_time_series(t_p, \
                            x_p, t_probe_kk)
        if len(c_probe) > 0:
            # Get probe locations that lie within the estimated timeframe
            c_probe_kk = c_probe[round(t_probe_kk_start/dt_probe):round(t_probe_kk_end/dt_probe)+1,:]
        else:
            c_probe_kk = np.zeros((len(t_probe_kk),3))
        abc = b_size[kk,:] / 2.0
        # Loop over each sensor and check if it is inside the bubble
        min_idx = len(t_probe_kk)+1
        max_idx = -1
        for idx,delta in sensor_delta.items():
            # Check if ellipsoid is pierced by sensor idx
            # Standard euqation: (x/a)2 + (y/b)2 + (z/c)2 = 1
            # with x = (cx+delta - x_bubble)
            radius = \
                (((c_probe_kk[:,0]+delta[0])-x_p_resampled[:,0])/abc[0])**2 \
              + (((c_probe_kk[:,1]+delta[1])-x_p_resampled[:,1])/abc[1])**2 \
              + (((c_probe_kk[:,2]+delta[2])-x_p_resampled[:,2])/abc[2])**2
            # Check for which time steps the bubble is pierced
            idxs = np.where(radius <= 1)
            # pierced, set signal to 1
            signal_indices[idx] = idxs
            if len(idxs[0]) > 0:
                min_idx = min(min_idx,np.nanmin(idxs[0]))
                max_idx = max(max_idx,np.nanmax(idxs[0]))
        if (max_idx - min_idx) >= 0:
            u_p_resampled = interpolate_time_series(t_p, \
                            bubble_props['velocity'], t_probe_kk)
            u_p_interaction = u_p_resampled[min_idx:(max_idx+1)]
            # Calculate the statistics
            # Calculate mean velocity
            u_mean = u_p_interaction.mean(axis=0)
            # Initialize the reynolds stress tensors time series
            reynolds_stress = np.empty((len(u_p_interaction),3,3))
            for ii in range(0,len(u_p_interaction)):
                # Calculate velocity fluctuations
                u_prime = u_p_interaction[ii,:]-u_mean
                # Reynolds stresses as outer product of fluctuations
                reynolds_stress[ii,:,:] = np.outer(u_prime, u_prime)
            # Calculate mean Reynolds stresses
            reynolds_stress = reynolds_stress.mean(axis=0)
            mean_reynolds_stress = [reynolds_stress[0,0],
                                    reynolds_stress[1,0],
                                    reynolds_stress[1,1],
                                    reynolds_stress[2,0],
                                    reynolds_stress[2,1],
                                    reynolds_stress[2,2]]
            # Calculate turbulent intensity with regard to mean x-velocity
            turbulent_intensity = np.sqrt(np.array([
                    mean_reynolds_stress[0], \
                    mean_reynolds_stress[2], \
                    mean_reynolds_stress[5], \
                    ])) / np.sqrt(u_mean.dot(u_mean))
            bubble_props["mean_velocity"] = u_mean
            bubble_props["mean_reynolds_stress"] = mean_reynolds_stress
            bubble_props["turbulent_intensity"] = turbulent_intensity
        # Display progress
        if progress:
            printProgressBar(kk + 1, nb, prefix = 'Progress:', suffix = 'Complete', length = 50)
        return [idx_start,signal_indices,bubble_props]
    else:
        for idx,delta in sensor_delta.items():
            signal_indices[idx] = np.array([])
        return [idx_start,signal_indices,bubble_props]

def get_virtual_distance(pos1,pos2,d1,d2,control_volume_size):
    y_min = min(pos1[1],pos2[1])
    y_max = min(pos1[1],pos2[1])
    z_min = min(pos1[2],pos2[2])
    z_max = min(pos1[2],pos2[2])
    dx = abs(pos2[0] - pos1[1])
    dy = (control_volume_size[1]/2.0-y_max) + (y_min - (-control_volume_size[1]/2.0))
    dz = (control_volume_size[2]/2.0-z_max) + (z_min - (-control_volume_size[2]/2.0))
    virtual_distance = math.sqrt(dx**2 + dy**2 + dz**2) 
    return virtual_distance

def get_bubble_overlap_distance(pos1,pos2,d1,d2,control_volume_size):
    actual_distance = min(math.sqrt(sum(((pos1-pos2)**2))),get_virtual_distance(pos1,pos2,d1,d2,control_volume_size))
    necessary_distance = d1/2.0 + d2/2.0
    bubbles_overlap = actual_distance < necessary_distance
    return max(necessary_distance - actual_distance,0.0)

def bubbles_overlap(pos1,pos2,d1,d2,control_volume_size):
    actual_distance = min(math.sqrt(sum(((pos1-pos2)**2))),get_virtual_distance(pos1,pos2,d1,d2,control_volume_size))
    necessary_distance = d1/2.0 + d2/2.0
    bubbles_overlap = actual_distance < necessary_distance
    return bubbles_overlap

def move_bubble(pos1,pos2,overlap_distance,control_volume_size):
    unit_vec = calculate_unit_vector(pos1,pos2)

    # determine new position
    y_new = pos2[1] + unit_vec[1]*overlap_distance*1.1
    z_new = pos2[2] + unit_vec[2]*overlap_distance*1.1

    # mirror at control volume boundary of y-axis if necessary
    if y_new > control_volume_size[1]/2.0:
        y_new = -control_volume_size[1]/2.0 + (y_new - control_volume_size[1]/2.0)
    elif y_new < -control_volume_size[1]/2.0:
        y_new = control_volume_size[1]/2.0 - (-control_volume_size[1]/2.0 - y_new)

    # mirror at control volume boundary of z-axis if necessary
    if z_new > control_volume_size[2]/2.0:
        z_new = -control_volume_size[2]/2.0 + (z_new - control_volume_size[2]/2.0)
    elif z_new < -control_volume_size[2]/2.0:
        z_new = control_volume_size[2]/2.0 - (-control_volume_size[2]/2.0 - z_new)

    # move to new position
    pos2_new = np.array([pos2[0],y_new,z_new])
    return pos2_new

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

    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the time vector
    t_f = reader.getDataSet('fluid/time')[:]
    # Read the fluid velocity
    u_f = reader.getDataSet('fluid/velocity')[:]
    # Read the fluid velocity
    x_f = reader.getDataSet('fluid/trajectory')[:]
    reader.close()

    # Define some variables for easier use
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

    # if necessary, downsample velocity and trajectory
    # minimum frequency to have at approx. five trajectory values
    # while bubble travels between tips
    travel_time = max_probe_size[0]/um[0]
    f_u = 1.0/travel_time*5.0
    if f_u >= flow_properties['realization_frequency']:
        n = round(flow_properties['duration'] * f_u)+1
        # Time vector
        t_new = np.linspace(0,flow_properties['duration'],n)
        t_new = np.sort(np.unique(np.append(t_new,t_f)))
        u_f = interpolate_time_series(t_f,u_f,t_new)
        x_f = interpolate_time_series(t_f,x_f,t_new)
        t_f = t_new

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
        C_eff = cumulative_chord_times/flow_properties['duration']
        nb = len(C_eff[C_eff <= C])
        F = nb/flow_properties['duration']
        b_size = b_size[0:nb,:]

    # Inter-arrival time (time between two bubbles)
    # ASSUMPTION: equally spaced (in time) bubble distribution
    # iat = 1.0/F - db/np.linalg.norm(um)
    inter_arrival_time = 1.0/F
    # Initialize arrival time (at) vector
    arrival_times = np.linspace(0,(nb-1)*inter_arrival_time,nb)
    arrival_times = arrival_times[arrival_times <= t_f[-2]]
    nb = len(arrival_times)
    # create random bubble arrival locations
    low  = [-control_volume_size[1]/2.0,-control_volume_size[2]/2.0]
    high = [ control_volume_size[1]/2.0, control_volume_size[2]/2.0]
    random_arrival_locations = np.random.uniform(low=low,high=high,size=(nb,2))

    arrival_locations = np.zeros((nb,2))
    arrival_locations[0,:] = random_arrival_locations[0,:]
    random_arrival_locations = np.delete(random_arrival_locations, 0, 0)
    touch_cnt = 0
    still_touch_cnt = 0

    print('\nRandomly placing bubbles.\n')

    for ii in range(1,nb):
        # Option 2:
        nb_check = min(ii,math.ceil(d_max/(inter_arrival_time*um[0])))

        overlap = False
        # approx. position of previous bubble, when bubble ii enters
        # entering position of bubble ii
        random_pos = random_arrival_locations[0,:]
        pos2 = np.array([0.0,
                        random_pos[0],
                        random_pos[1]])
        for jj in range(0,nb_check):
            idx = ii-jj-1
            pos_jj = np.array([um[0]*(arrival_times[ii]-arrival_times[idx]),
                                         arrival_locations[idx,0],
                                         arrival_locations[idx,1]])

            if bubbles_overlap(pos_jj,pos2,b_size[idx,0],b_size[ii,0],control_volume_size):
                overlap = True
        if overlap:
            touch_cnt += 1
        max_iter = len(random_arrival_locations)-2
        n_iter = 0
        while overlap & (n_iter <= max_iter):
            n_iter += 1
            overlap = False
            # create now position and hope they don't overlap
            random_pos = random_arrival_locations[n_iter,:]
            pos2 = np.array([0.0,
                        random_pos[0],
                        random_pos[1]])
            for jj in range(0,nb_check):
                idx = ii-jj-1
                pos_jj = np.array([um[0]*(arrival_times[ii]-arrival_times[idx]),
                                             arrival_locations[idx,0],
                                             arrival_locations[idx,1]])
                if bubbles_overlap(pos_jj,pos2,b_size[idx,0],b_size[ii,0],control_volume_size):
                    overlap = True
        if overlap:
            still_touch_cnt += 1
        random_arrival_locations = np.delete(random_arrival_locations, n_iter, 0)
        arrival_locations[ii,:] = pos2[1::]
    time2 = time.time()
    print(f'Finished bubble placement. Current runtime: {time2-time1:.2f} seconds\n')
    print('\nTracking bubbles with respect to probe.\n')
    # The duration
    duration = flow_properties['duration'];
    # The sampling frequency
    f_probe = probe['sampling_frequency']
    # Sampling time step
    dt_probe = 1.0 / f_probe
    # Number sampling time steps
    n_probe = round(duration / dt_probe) + 1;
    # Time vector
    t_probe = np.linspace(0, duration, n_probe);
    # Initialize the probe location
    c_probe = np.zeros((n_probe,3))
    if 'VIBRATION' in uncertainty:
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


    # get the indices of the signal where piercing happened
    if 'VIBRATION' in uncertainty:
        results = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(SBG_simulate_bubble_piercing)(kk, t_f, x_f, u_f, control_volume_size, arrival_times, arrival_locations, b_size, dt_probe, sensor_delta, progress, nb, c_probe) for kk in range(0,nb))
    else:
        results = Parallel(n_jobs=nthreads,backend='multiprocessing')(delayed(SBG_simulate_bubble_piercing)(kk, t_f, x_f, u_f, control_volume_size, arrival_times, arrival_locations, b_size, dt_probe, sensor_delta, progress, nb) for kk in range(0,nb))
    # Initialize signals to zero
    signal = np.zeros((n_probe, n_sensors)).astype('uint8')
    time2 = time.time()
    print(f'Finished bubble tracking. Current runtime: {time2-time1:.2f} seconds\n')
    print('\nWriting signal and results.\n')
    # Initialize bubble properties
    arrival_times_bubble = np.zeros(nb)*np.nan
    arrival_locations_bubble = np.zeros((nb,3))*np.nan
    exit_times_bubble = np.zeros(nb)*np.nan
    exit_locations_bubble = np.zeros((nb,3))*np.nan
    mean_velocities_bubble = np.zeros((nb,3))*np.nan
    mean_reynolds_stresses_bubble = np.zeros((nb,6))*np.nan
    turbulent_intensities_bubble = np.zeros((nb,3))*np.nan
    chord_times_bubble = np.zeros((nb,len(sensors)))*np.nan
    chord_lengths_bubble = np.zeros((nb,len(sensors)))*np.nan
    pierced_bubbles = np.zeros((nb,len(sensors)),dtype='uint8')
    n_bubbles_pierced = 0
    cumulative_chord_times = 0.0
    cumulative_chord_times_overlap = 0.0
    for ii,bubble in enumerate(results):
        # get signal start time (to get proper indices)
        idx_start = bubble[0]
        # get indices of signal for which the signal must be changed to 1
        signal_indices = bubble[1]
        # get bubble properties
        bubble_props = bubble[2]
        # Determine the row in the signal time series where to write the signal
        # write the signal for each bubble based on time indices when
        # when the bubble was in contact with a sensor
        for sensor_idx in signal_indices:
            # set those indices to 1
            if len(signal_indices[sensor_idx][0]) > 0:
                if sensor_idx == 0:
                    cumulative_chord_times += float(len(signal_indices[sensor_idx][0]))/float(f_probe)
                    overlap = np.sum(signal[(signal_indices[sensor_idx][0]+idx_start),sensor_idx] == 1)
                    cumulative_chord_times_overlap += float(overlap)/float(f_probe)
                    n_bubbles_pierced+=1
                signal[(signal_indices[sensor_idx][0]+idx_start),sensor_idx] = 1
                pierced_bubbles[ii,sensor_idx] = 1
                chord_times_bubble[ii,sensor_idx] = float(len(signal_indices[sensor_idx][0]))/float(f_probe)
                chord_lengths_bubble[ii,sensor_idx] = float(len(signal_indices[sensor_idx][0]))/float(f_probe)*bubble_props["mean_velocity"][0]
        arrival_times_bubble[ii] = bubble_props["arrival_time"]
        arrival_locations_bubble[ii,:] = bubble_props["arrival_location"]
        exit_times_bubble[ii] = bubble_props["exit_time"]
        exit_locations_bubble[ii,:] = bubble_props["exit_location"]
        mean_velocities_bubble[ii,:] = bubble_props["mean_velocity"]
        mean_reynolds_stresses_bubble[ii,:] = bubble_props["mean_reynolds_stress"]
        turbulent_intensities_bubble[ii,:] = bubble_props["turbulent_intensity"]
    C_bubbles_overlapped = cumulative_chord_times_overlap/float(duration)
    C_bubbles_pierced = cumulative_chord_times/float(duration)
    F_bubbles_pierced = n_bubbles_pierced/duration
    C_measured = np.mean(signal[:,0])
    # Create the H5-file writer
    writer = H5Writer(path / 'binary_signal.h5', 'w')
    # Write the time vector
    writer.writeDataSet('time', t_probe, 'float64')
    writer.writeDataSet('signal', signal, 'u1')
    ds_sig = writer.getDataSet('signal')
    ds_sig.attrs['sensor_id'] = list(sensor_delta.keys())
    writer.close()
    time2 = time.time()

    # Create the H5-file writer
    writer = H5Writer(path / 'flow_data.h5', 'a')
    # Write the bubble properties
    writer.writeDataSet('bubbles/arrival_times', arrival_times_bubble, 'float64')
    writer.writeDataSet('bubbles/arrival_locations', arrival_locations_bubble, 'float64')
    writer.writeDataSet('bubbles/exit_times', exit_times_bubble, 'float64')
    writer.writeDataSet('bubbles/exit_locations', exit_locations_bubble, 'float64')
    writer.writeDataSet('bubbles/size', b_size, 'float64')
    writer.writeDataSet('bubbles/mean_velocity', mean_velocities_bubble, 'float64')
    writer.writeDataSet('bubbles/reynold_stresses', mean_reynolds_stresses_bubble, 'float64')
    writer.writeDataSet('bubbles/turbulent_intensity', turbulent_intensities_bubble, 'float64')
    writer.writeDataSet('bubbles/chord_times', chord_times_bubble, 'float64')
    writer.writeDataSet('bubbles/chord_lengths', chord_lengths_bubble, 'float64')
    writer.writeDataSet('bubbles/pierced_bubbles', pierced_bubbles, 'u1')
    writer.writeDataSet('bubbles/pierced_bubble_frequency', np.array([F_bubbles_pierced]), 'float64')
    writer.writeDataSet('bubbles/pierced_bubble_void_fraction', np.array([C_bubbles_pierced]), 'float64')
    writer.writeDataSet('bubbles/pierced_overlapped_bubble_void_fraction', np.array([C_bubbles_overlapped]), 'float64')
    writer.writeDataSet('bubbles/measured_void_fraction', np.array([C_measured]), 'float64')
    writer.close()

    print(f'Successfully generated the signal')
    print(f'Finished in {time2-time1:.2f} seconds\n')