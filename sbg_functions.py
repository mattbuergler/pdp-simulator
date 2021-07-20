#!/usr/bin/env python3

import numpy as np
import pandas as pd
import bisect
import random
import time
import math

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

def get_bubble_size(flow_properties):
    # Define some variables for easier use
    Um = np.asarray(flow_properties['mean_velocity'])
    C = flow_properties['void_fraction']
    tau = flow_properties['duration'];
    # Get bubble properties
    bubbles = flow_properties['bubbles']
    if bubbles['shape'] == 'sphere':
        if bubbles['size_distribution'] == 'constant':
            # Mean bubble frequency
            F = 1.5 * C * np.linalg.norm(Um) / (bubbles['diameter'])
            # Number of simulated bubbles
            nb = round(F * tau)
            # Vector with bubble size (A,B,C), V = 1/6*Pi*A*B*C
            b_size = np.ones((nb,3))*bubbles['diameter']
        elif bubbles['size_distribution'] == 'lognormal':
            # Mean bubble frequency
            F = 1.5 * C * np.linalg.norm(Um)  \
                / math.exp(bubbles['mean_ln_diameter'])
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
                / math.exp(bubbles['mean_ln_diameter'])
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
    return nb, F, b_size

def get_mean_bubble_sve_size(flow_properties):
    # Returns mean sphere-volume-equivalent bubble size D_sve
    # Get bubble properties
    bubbles = flow_properties['bubbles']
    if bubbles['size_distribution'] == 'constant':
        D_sve = bubbles['diameter']
    elif bubbles['size_distribution'] == 'lognormal':
        D_sve = math.exp(bubbles['mean_ln_diameter'])
    return D_sve

def SBG_fluid_velocity(path, flow_properties, reproducible, progress):
    """SBG including correlation between u' and v' as well as autocorrelation
    in u(t) depending on integral time scale.

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
    pXZ = -0.45
    pYZ = 0.0
    # rho_{uv}=-0.45 at y+~98 (Table 7.2 Pope, 2000)
    pXY = 0.0
    # Covariance with diagonal components = 1
    cov = [[1.0, pXY, pXZ], \
           [pXY, 1.0, pYZ], \
           [pXZ, pYZ, 1.0]]

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
        u_f[0,0] = SBG_dt_corr(u_f_old[0,0], um[0], sigma[0], T[0], dt, R[0,0]);
        u_f[0,1] = SBG_dt_corr(u_f_old[0,1], um[1], sigma[1], T[1], dt, R[0,1]);
        u_f[0,2] = SBG_dt_corr(u_f_old[0,2], um[2], sigma[2], T[2], dt, R[0,2]);
        # First trajectory vector depends on old chunk
        x_f[0,0] = x_f_old[0,0] + (u_f[0,0]+u_f_old[0,0])/2.0*dt;
        x_f[0,1] = x_f_old[0,1] + (u_f[0,1]+u_f_old[0,1])/2.0*dt;
        x_f[0,2] = x_f_old[0,2] + (u_f[0,2]+u_f_old[0,2])/2.0*dt;
        # Calculate rest of the chunk
        for ii in range(1,n_chunk):
            # Calculate velocity
            u_f[ii,0] = SBG_dt_corr(u_f[ii-1,0], um[0], sigma[0], T[0], dt, R[ii,0]);
            u_f[ii,1] = SBG_dt_corr(u_f[ii-1,1], um[1], sigma[1], T[1], dt, R[ii,1]);
            u_f[ii,2] = SBG_dt_corr(u_f[ii-1,2], um[2], sigma[2], T[2], dt, R[ii,2]);
            # Calculate trajectory
            x_f[ii,0] = x_f[ii-1,0] + (u_f[ii,0]+u_f[ii-1,0])/2.0*dt;
            x_f[ii,1] = x_f[ii-1,1] + (u_f[ii,1]+u_f[ii-1,1])/2.0*dt;
            x_f[ii,2] = x_f[ii-1,2] + (u_f[ii,2]+u_f[ii-1,2])/2.0*dt;
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
    print(f'Finished in {time2-time1} seconds\n')

def SBG_bubble_velocity(path, flow_properties, progress):
    """Calculate the bubble velocity based on a force balance equation between drag and virtual mass acting on the bubble by the fluid.

        Parameters
        ----------
        path      (pathlib.Path): The directory
        flow_properties   (dict): A dictionary containing the flow properties
        progress          (bool): A flag to print the progress

        Returns
        ----------
        -
    """
    time1 = time.time()

    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the time vector
    t_f = reader.getDataSet('fluid/time')[:]
    # Read the fluid velocity
    u_f = reader.getDataSet('fluid/velocity')[:]
    reader.close()

    print('\nGenerating velocity and trajectory time series of the bubbles\n')

    # Initialize velocity time series and trajectory of the bubble
    u_p = np.empty((len(u_f),3)) * np.nan
    x_p = np.empty((len(u_f),3)) * np.NaN

    # Set initial conditions: u_p(t=0) = u_f(t=0) 
    u_p[0,:] = u_f[0,:]
    x_p[0,:] = 0.0
    # Solve momentum equation for bubble velocity with a first order Euler
    # scheme. The forces acting on the bubble are given by Eq. 5 in 
    # Balachandar, S., & Eaton, J. K. (2010). Turbulent dispersed multiphase
    # flow. Annual review of fluid mechanics, 42, 111-133.
    # https://doi.org/10.1146/annurev.fluid.010908.165243
    # Some parameters
    C_M = 0.5                   # virtual mass coefficient (Rushe, 2002)
    rho_f = 1000.0              # density of the fluid [kg/m3]
    rho_p = 1.0                 # density of the bubble [kg/m3]
    mu_f = 0.001                # dyn. viscosity of the fluid [Pa*s]
    piTimes3 = 3.0 * math.pi    # 3 * Pi
    # get the mean sphere-volume-equivalent bubble diameter
    d = get_mean_bubble_sve_size(flow_properties)
    V_p = math.pi/6.0*d**3.     # sphere volume
    for ii in range(1, len(t_f)):
        dt = t_f[ii] - t_f[ii-1]            # Integration time step
        u_r = u_f[ii-1,:] - u_p[ii-1,:]     # Rel. velocity of prev. timestep
        Re = rho_p*abs(u_r)*d/mu_f          # Bubble Reynolds number
        phi = 1.0 + 0.15*(Re**0.687)        # Drag coefficient * Re / 24
        du_dt = 1.0/dt * (u_f[ii,:]
                        -u_f[ii-1,:])       # Time derivative of fluid velocity
        u_p[ii,:] = u_p[ii-1,:] + dt/(V_p*(rho_p + C_M*rho_f)) * ( \
                            # Drag force
                            piTimes3 * mu_f * d * u_r * phi \
        #                     # Acceleration force of fluid
        #                     + rho_f * du_dt \
        #                     # Added mass force
        #                     + (C_M*rho_f) * du_dt \
                        )
        # Calculate trajectory
        x_p[ii,:] = x_p[ii-1,:] + (u_p[ii,:]+u_p[ii-1,:])/2.0*dt;

    # Calculate the statistics
    # Calculate mean velocity
    u_p_m = u_p.mean(axis=0)
    # Initialize the reynolds stress tensors time series
    reynolds_stress = np.empty((len(u_p),3,3))
    for ii in range(0,len(u_p)):
        # Calculate velocity fluctuations
        u_p_prime = u_p[ii,:]-u_p_m
        # Reynolds stresses as outer product of fluctuations
        reynolds_stress[ii,:,:] = np.outer(u_p_prime, u_p_prime)
    # Calculate mean Reynolds stresses
    mean_reynolds_stress = reynolds_stress.mean(axis=0)
    # Calculate turbulent intensity with regard to mean x-velocity
    turbulent_intensity = np.sqrt(np.array([
            mean_reynolds_stress[0,0], \
            mean_reynolds_stress[1,1], \
            mean_reynolds_stress[2,2], \
            ])) / np.sqrt(u_p_m.dot(u_p_m))
    # Create the H5-file writer
    writer = H5Writer(path / 'flow_data.h5', 'a')
    # Write the time vector
    writer.writeDataSet('bubbles/time', t_f, 'float64')
    # Write the velocity time series
    writer.writeDataSet('bubbles/velocity', u_p, 'float64')
    # Write the trajectory
    writer.writeDataSet('bubbles/trajectory', x_p, 'float64')
    writer.writeDataSet('bubbles/mean_velocity', \
        u_p_m, 'float64')
    writer.writeDataSet('bubbles/reynold_stresses', \
        mean_reynolds_stress, 'float64')
    writer.writeDataSet('bubbles/turbulent_intensity', \
        turbulent_intensity, 'float64')
    writer.close()
    time2 = time.time()
    print(f'Successfully written bubble velocity time series and trajectory')
    print(f'Finished in {time2-time1} seconds\n')

def SBG_interp_trajectory(
    t_traj,
    X,
    t):
    idx = bisect.bisect_right(t_traj, t)
    m = (X[idx] - X[idx-1]) / (t_traj[idx] - t_traj[idx-1])
    Xinterp = X[idx] + m*(t - t_traj[idx])
    return Xinterp

def SBG_get_Signal_traj(
    t_traj,
    X,
    t_probe):
    t_probe_unique = np.unique(np.hstack((t_traj, t_probe[t_probe<=max(t_traj)])))
    t_traj_del = t_traj[np.invert(np.isin(t_traj, t_probe))]
    Xinterp = pd.DataFrame(X, index=t_traj, columns=['x','y','z'])
    Xinterp = Xinterp.reindex(t_probe_unique).interpolate(method='index')
    Xinterp = Xinterp.drop(index=t_traj_del)
    return Xinterp.index, Xinterp.values

def SBG_signal(
    path,
    flow_properties,
    probe,
    reproducible,
    progress):
    """Generate the bubble field, place probe and collect signal.

        Parameters
        ----------
        path      (pathlib.Path): The directory
        flow_properties   (dict): A dictionary containing the flow properties
        probe             (dict): A dictionary containing the probe properties
        reproducible    (string): A string defining the reproducibility
        progress          (bool): A flag to print the progress

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
    t_traj = reader.getDataSet('bubbles/time')[:]
    # Read the trajectory
    X = reader.getDataSet('bubbles/trajectory')[:]
    reader.close()
    # Duration of the time series
    duration = t_traj[-1] - t_traj[0]
    # Number of velocity realizations
    n = len(t_traj)

    # Define some variables for easier use
    um = np.asarray(flow_properties['mean_velocity'])
    C = flow_properties['void_fraction']

    # Get number of bubbles, bubble frequency and array with bubble size
    nb, F, b_size = get_bubble_size(flow_properties)

    # Inter-arrival time (time between two bubbles)
    # ASSUMPTION: equally spaced (in time) bubble distribution
    # iat = 1.0/F - db/np.linalg.norm(um)
    iat = np.ones(nb)*1.0/F
    # Initialize arrival time (AT) vector
    AT = np.zeros(len(iat))
    for kk in range(1,len(iat)):
        AT[kk] = AT[kk-1] + iat[kk];

    # Create the H5-file writer
    writer = H5Writer(path / 'flow_data.h5', 'a')
    # Create the velocity data set for the entire time series of length n
    writer.writeDataSet('bubbles/arrival_time', AT, 'float64')
    writer.writeDataSet('bubbles/size', b_size, 'float64')
    writer.close()
    # Initialize bubble center location to zero
    cx = np.zeros(nb)
    cy = np.zeros(nb)
    cz = np.zeros(nb);

    # Gather the probe information (id and relative location [=delta])
    n_sensors = len(probe['sensors'])
    # Initialize an empty dictionary
    sensor_delta = {}
    for sensor in probe['sensors']:
        # fill dictionary:  id -> relative_location
        sensor_delta[sensor['id']] = np.asarray(sensor['relative_location'])
    # Get an estimate of the maximum probe dimension
    sensors = probe['sensors']
    minmax = np.array([[LARGENUMBER,LARGENEGNUMBER],
                       [LARGENUMBER,LARGENEGNUMBER],
                       [LARGENUMBER,LARGENEGNUMBER]])
    for sensor in sensors:
        for ii in range(0,3):
            minmax[ii,0] = min(minmax[ii,0], sensor['relative_location'][ii])
            minmax[ii,1] = max(minmax[ii,1], sensor['relative_location'][ii])
    max_probe_size = max(minmax[:,1]-minmax[:,0])

    # The sampling frequency
    f_probe = probe['sampling_frequency']
    # Sampling time step
    dt_probe = 1.0 / f_probe
    # Number sampling time steps
    n_probe = round(duration / dt_probe) + 1;
    # Time vector
    t_probe = np.linspace(0, duration, n_probe);
    # Initialize signals to zero
    signal = np.zeros((n_probe, n_sensors)).astype('int')
    # Loop over all bubbles
    print('\nSampling the sensor signal')
    for kk in range(0,nb):
        # Interpolate the location of the bubble at t = AT
        X_b_AT = SBG_interp_trajectory(t_traj, X, AT[kk])
        # Set x-coordinate of probe to x-coordinate ob bubble center at AT
        cx[kk] = X_b_AT[0]
        # Set y-coordinate of probe to y-coordinate ob bubble center + random
        # shift with ~Uniform[-B/2,B/2]
        cy[kk] = X_b_AT[1] + random.uniform(-b_size[kk,1]/2.0,b_size[kk,1]/2.0)
        # Set z-coordinate of probe to z-coordinate ob bubble center + random
        # shift with ~Uniform[-B/2,B/2]
        cz[kk] = X_b_AT[2] + random.uniform(-b_size[kk,2]/2.0,b_size[kk,2]/2.0)
        # Estimate timeframe for tracking the movement of bubble kk
        critical_time_pre = 3.0*np.linalg.norm(b_size[kk,:])/np.linalg.norm(um)
        critical_time_post = 3.0*(np.linalg.norm(b_size[kk,:]) + \
                             max_probe_size)/np.linalg.norm(um)
        t_min = AT[kk] - critical_time_pre
        t_max = AT[kk] + critical_time_post
        # Get probe sampling times that lie within the estimated timeframe
        t_probe_kk = t_probe[(t_probe >= t_min) & (t_probe <= t_max)]
        # Check if number of samples lies inside the timeframe is larger than 0
        if len(t_probe_kk) > 0:
            # Get the range of the trajectory that encloses this timeframe
            id_t_min_traj = max(bisect.bisect_left(t_traj, min(t_probe_kk))-1,
                0)
            id_t_max_traj = min(bisect.bisect_right(t_traj, max(t_probe_kk))+1,len(t_traj))
            t_traj_kk = t_traj[id_t_min_traj:id_t_max_traj]
            X_traj_kk = X[id_t_min_traj:id_t_max_traj]
            # Resample the trajectory to the sampling times of the probe
            t_resampled, X_resampled = SBG_get_Signal_traj(t_traj_kk, \
                                X_traj_kk, t_probe_kk)
            # Loop over all time steps within the timeframe
            abc = b_size[kk,:] / 2.0
            # Determine the row in the signal time series where to write the signal
            row = np.where(t_probe == min(t_probe_kk))[0][0]
            # Loop over each sensor and check if it is inside the bubble
            for idx,delta in sensor_delta.items():
                # Check if ellipsoid is pierced by sensor idx
                # Standard euqation: (x/a)2 + (y/b)2 + (z/c)2 = 1
                # with x = (cx+delta - x_bubble)
                radius = \
                        (((cx[kk]+delta[0])-X_resampled[:,0])/abc[0])**2 \
                      + (((cy[kk]+delta[1])-X_resampled[:,1])/abc[1])**2 \
                      + (((cz[kk]+delta[2])-X_resampled[:,2])/abc[2])**2
                # Check for which time steps the bubble is pierced
                idxs = np.where(radius <= 1) + row
                # pierced, set signal to 1
                signal[idxs,idx] = 1;
            # Display progress
            if progress:
                printProgressBar(kk + 1, nb, prefix = 'Progress:', suffix = 'Complete', length = 50)
    # Create the H5-file writer
    writer = H5Writer(path / 'binary_signal.h5', 'w')
    # Write the time vector
    writer.writeDataSet('time', t_probe, 'float64')
    writer.writeDataSet('signal', signal, 'u4')
    ds_sig = writer.getDataSet('signal')
    ds_sig.attrs['sensor_id'] = list(sensor_delta.keys())
    writer.close()
    time2 = time.time()
    print(f'Successfully generated the signal')
    print(f'Finished in {time2-time1} seconds\n')