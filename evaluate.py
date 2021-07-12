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
    reader.close()
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
    # Initialize arrays for errors
    RMSD_vel_reco = np.empty((len(vel_rec),3))
    RMSD_vel_true = np.empty((len(vel_rec),3))
    E_vel = np.empty((len(vel_rec),3))
    RE_vel = np.empty((len(vel_rec),3))
    AE_vel = np.empty((len(vel_rec),3))
    ARE_vel = np.empty((len(vel_rec),3))
    # Initialize array for true mean bubble velocity
    vel_true_bubble = np.empty((len(vel_rec),3))
    for ii in range(0,len(t_rec)):
        # Get the range of the velocity timeseries that encloses the timeframe
        # of bubble-probe interaction (tfbpi)
        id_t_min = max(bisect.bisect_right(t_true, t_rec[ii,0])-1, 0)
        id_t_max = min(bisect.bisect_left(t_true, t_rec[ii,1]),len(t_true))
        # Calculate true mean bubble velocity (mean over tfbpi)
        vel_true_bubble[ii,:] = np.mean(vel_true[id_t_min:(id_t_max+1),:],axis=0)
        # The mean squared deviation (MSD) of the reconstructed and true
        # velocity
        MSD_vel_rec = np.mean((vel_rec[ii,:]
                -vel_true[id_t_min:(id_t_max+1),:])**2, axis=0)
        MSD_vel_true = np.mean((vel_true[id_t_min:(id_t_max+1),:] \
                - vel_true_bubble[ii,:])**2,axis=0)
        # The root of the mean squared deviations (RMSD)
        RMSD_vel_reco[ii,:] = np.sqrt(MSD_vel_rec / (id_t_max-id_t_min))
        RMSD_vel_true[ii,:] = np.sqrt(MSD_vel_true / (id_t_max-id_t_min))
        # Get the absolute error (AE) and the absolute relative error (ARE)
        E_vel[ii,:] = vel_rec[ii,:] - vel_true_bubble[ii,:]
        RE_vel[ii,:] = (vel_rec[ii,:] - vel_true_bubble[ii,:]) / vel_true_bubble[ii,:]
        AE_vel[ii,:] = abs(vel_rec[ii,:] - vel_true_bubble[ii,:])
        ARE_vel[ii,:] = abs((vel_rec[ii,:] - vel_true_bubble[ii,:]) / vel_true_bubble[ii,:])

    # Convert to DataFrames
    RMSD_vel_reco = pd.DataFrame(RMSD_vel_reco, index=np.mean(t_rec,axis=1),
        columns=['Ux','Uy','Uz'])
    RMSD_vel_true = pd.DataFrame(RMSD_vel_true, index=np.mean(t_rec,axis=1),
        columns=['Ux','Uy','Uz'])
    E_vel = pd.DataFrame(E_vel, index=np.mean(t_rec,axis=1),
        columns=['Ux','Uy','Uz'])
    RE_vel = pd.DataFrame(RE_vel, index=np.mean(t_rec,axis=1),
        columns=['Ux','Uy','Uz'])
    AE_vel = pd.DataFrame(AE_vel, index=np.mean(t_rec,axis=1),
        columns=['Ux','Uy','Uz'])
    ARE_vel = pd.DataFrame(ARE_vel, index=np.mean(t_rec,axis=1),
        columns=['Ux','Uy','Uz'])

    # Calculate mean RMSD for the true and reconstructed velocity and their    # ratio epsilon
    summary_vel = pd.DataFrame(np.nan, \
            index=['MRMSD_true','MRMSD_rec','mean_epsilon'], \
            columns=['Ux','Uy','Uz'])
    summary_vel.loc['MRMSD_true',:] = RMSD_vel_true.mean()
    summary_vel.loc['MRMSD_rec',:] = RMSD_vel_reco.mean()
    summary_vel.loc['mean_epsilon',:] = ((RMSD_vel_reco/RMSD_vel_true)).mean()
    summary_vel.loc['ME_rec',:] = E_vel.mean()
    summary_vel.loc['MRE_rec',:] = RE_vel.mean()
    summary_vel.loc['MAE_rec',:] = AE_vel.mean()
    summary_vel.loc['MARE_rec',:] = ARE_vel.mean()

    cumulative_mean_AE = np.empty((len(ARE_vel),3))
    # Convergence of mean error with number of bubbles
    for ii in range(0,len(AE_vel)):
        cumulative_mean_AE[ii,:] = AE_vel.iloc[:ii].mean()

    # Calculate error for bubble size
    b_size = pd.DataFrame(b_size,index=t_b_size, columns=['A','B','C'])
    # Calculate volumen equivalent diameter
    b_size['D'] = (b_size['A']*b_size['B']*b_size['C'])**(1.0/3.0)
    # Convert to pandas array
    b_size_rec = pd.DataFrame(b_size_rec,index=np.mean(t_rec,axis=1), \
        columns=['D_h_2h', 'D_h_2hp1'])
    b_size_rec['D'] = (b_size_rec['D_h_2h'] + b_size_rec['D_h_2hp1'])/2.0
    # Get timesteps of the reconstructed velocities that are NOT in present in
    # the original time series
    b_size_rec_unique = b_size_rec.index[np.invert(np.isin(b_size_rec.index, \
        b_size.index))]
    # Add times of reconstructed velocity field to original DataFrame with
    # NaN-values and then interpolate
    empty = pd.DataFrame(np.nan, index=b_size_rec_unique, columns=['D'])
    b_size = b_size.append(empty).sort_index().interpolate()

    # Caluculate actual and relative errors
    error_D = b_size_rec['D'] - b_size['D'][b_size_rec.index]
    rel_error_D = error_D / b_size['D'][b_size_rec.index]

    # Convert to DataFrame
    errors_D = pd.DataFrame(error_D.values, index=np.mean(t_rec,axis=1),
        columns=['E'])
    errors_D['RE'] = rel_error_D
    # Calculate squared error (SE) and squared relative errors (SRE)
    errors_D['SE'] = errors_D['E']**2
    errors_D['SRE'] = errors_D['RE']**2

    # Calculate mean average error (MAE) and root mean square error (RMSE)
    # for the bubble diameter
    summary_b_size = pd.DataFrame(np.nan, index=['ME','MRE','MAE','RMSE','MARE','RMSRE'],\
                            columns=['D'])

    # Store summary of errors
    summary_b_size['D']['ME'] = errors_D['E'].mean()
    summary_b_size['D']['MRE'] = errors_D['RE'].mean()
    summary_b_size['D']['MAE'] = errors_D['E'].abs().mean()
    summary_b_size['D']['MARE'] = errors_D['RE'].abs().mean()
    summary_b_size['D']['RMSE'] = math.sqrt(errors_D['SE'].mean())
    summary_b_size['D']['RMSRE'] = math.sqrt(errors_D['SRE'].abs().mean())

    # Calculate true mean velocity and Reynolds stress tensor based on true
    # bubble velocity
    # Calculate mean true bubble velocity
    mean_vel_true_bubble = vel_true_bubble.mean(axis=0)
    # Initialize array for reynolds stress based on true mean bubble velocity
    rs_true_bubble = np.empty((len(vel_rec),3,3))
    for ii in range(0,len(t_rec)):
        u_x = vel_true_bubble[ii,0] - mean_vel_true_bubble[0]
        u_y = vel_true_bubble[ii,1] - mean_vel_true_bubble[1]
        u_z = vel_true_bubble[ii,2] - mean_vel_true_bubble[2]
        rs_true_bubble[ii,0,0] = u_x*u_x
        rs_true_bubble[ii,0,1] = u_x*u_y
        rs_true_bubble[ii,0,2] = u_x*u_z
        rs_true_bubble[ii,1,0] = u_y*u_x
        rs_true_bubble[ii,1,1] = u_y*u_y
        rs_true_bubble[ii,1,2] = u_y*u_z
        rs_true_bubble[ii,2,0] = u_z*u_x
        rs_true_bubble[ii,2,1] = u_z*u_y
        rs_true_bubble[ii,2,2] = u_z*u_z
    mean_rs_true_bubble = rs_true_bubble.mean(axis=0)
    turbulent_intensity_bubble = np.sqrt(np.array([
            mean_rs_true_bubble[0,0], \
            mean_rs_true_bubble[1,1], \
            mean_rs_true_bubble[2,2], \
            ])) / mean_vel_true_bubble[0]
    # Calculate errors in mean velocity and Reynolds stresses
    # Relative error of mean velocity with regard to mean of entire bubble velocity time series 
    rel_error_mean_vel = pd.DataFrame([((vel_mean_rec-vel_mean_true) \
        / vel_mean_true)],index=[0],columns=['ux','uy','uz'])
    rel_error_weighted_mean_vel = pd.DataFrame([((vel_weighted_mean_rec-vel_mean_true) \
        / vel_mean_true)],index=[0],columns=['ux','uy','uz'])
    # Relative error of mean velocity with regard to mean bubble velocity during tfbpi
    rel_error_mean_vel_bubble = pd.DataFrame([((vel_mean_rec-mean_vel_true_bubble) \
        / mean_vel_true_bubble)],index=[0],columns=['ux','uy','uz'])
    # Error of RST with regard to mean of entire bubble velocity time series 
    rel_error_rs = pd.DataFrame((rs_rec-rs_true)/rs_true, \
        index=['x','y','z'],columns=['x','y','z'])
    # Relative error of RST with regard to mean bubble velocity during tfbpi
    rel_error_rs_bubble = pd.DataFrame((rs_rec-mean_rs_true_bubble)/mean_rs_true_bubble, \
        index=['x','y','z'],columns=['x','y','z'])
    # Error of mean velocity with regard to mean of entire bubble velocity time series 
    error_mean_vel = pd.DataFrame([(vel_mean_rec-vel_mean_true)],index=[0],columns=['ux','uy','uz'])
    error_weighted_mean_vel = pd.DataFrame([(vel_weighted_mean_rec-vel_mean_true)],index=[0],columns=['ux','uy','uz'])
    # Error of mean velocity with regard to mean bubble velocity during tfbpi
    error_mean_vel_bubble = pd.DataFrame([(vel_mean_rec-mean_vel_true_bubble)],index=[0],columns=['ux','uy','uz'])
    # Error of RST with regard to mean of entire bubble velocity time series 
    error_rs = pd.DataFrame(rs_rec-mean_rs_true_bubble, \
        index=['x','y','z'],columns=['x','y','z'])
    error_rs_bubble = pd.DataFrame(rs_rec-rs_true, \
        index=['x','y','z'],columns=['x','y','z'])
    rel_error_Ti = pd.DataFrame([((Ti_rec-Ti_true)/Ti_true)],index=[0],columns=['Tix','Tiy','Tiz'])
    rel_error_Ti_bubble = pd.DataFrame([((Ti_rec-turbulent_intensity_bubble)/turbulent_intensity_bubble)],index=[0],columns=['Tix','Tiy','Tiz'])
    error_Ti = pd.DataFrame([(Ti_rec-Ti_true)],index=[0],columns=['Tix','Tiy','Tiz'])
    error_Ti_bubble = pd.DataFrame([(Ti_rec-turbulent_intensity_bubble)],index=[0],columns=['Tix','Tiy','Tiz'])
    # Write output
    errors_D.to_csv(path / 'errors_D.csv', index=True,index_label='t')
    RMSD_vel_reco.to_csv(path / 'RMSD_vel_reco.csv', index=True,index_label='t')
    RMSD_vel_true.to_csv(path / 'RMSD_vel_true.csv', index=True,index_label='t')
    E_vel.to_csv(path / 'E_vel.csv', index=True,index_label='t')
    RE_vel.to_csv(path / 'RE_vel.csv', index=True,index_label='t')
    AE_vel.to_csv(path / 'AE_vel.csv', index=True,index_label='t')
    ARE_vel.to_csv(path / 'ARE_vel.csv', index=True,index_label='t')
    summary_b_size.to_csv(path / 'error_summary_bubble_size.csv', index=True)
    summary_vel.to_csv(path / 'error_summary_velocity.csv', index=True)
    error_mean_vel.to_csv(path / 'error_mean_velocity.csv', index=True)
    error_weighted_mean_vel.to_csv(path / 'error_weighted_mean_velocity.csv', index=True)
    error_mean_vel_bubble.to_csv(path / 'error_mean_velocity_bubble.csv', index=True)
    error_rs.to_csv(path / 'error_reynolds_stresses.csv', index=True)
    error_rs_bubble.to_csv(path / 'error_reynolds_stresses_bubble.csv', index=True)
    error_Ti.to_csv(path / 'error_turbulent_intensity.csv', index=True)
    error_Ti_bubble.to_csv(path / 'error_turbulent_intensity_bubble.csv', index=True)
    rel_error_mean_vel.to_csv(path / 'rel_error_mean_velocity.csv', index=True)
    rel_error_weighted_mean_vel.to_csv(path / 'rel_error_weighted_mean_velocity.csv', index=True)
    rel_error_mean_vel_bubble.to_csv(path / 'rel_error_mean_velocity_bubble.csv', index=True)
    rel_error_rs.to_csv(path / 'rel_error_reynolds_stresses.csv', index=True)
    rel_error_rs_bubble.to_csv(path / 'rel_error_reynolds_stresses_bubble.csv', index=True)
    rel_error_Ti.to_csv(path / 'rel_error_turbulent_intensity.csv', index=True)
    rel_error_Ti_bubble.to_csv(path / 'rel_error_turbulent_intensity_bubble.csv', index=True)

    # Check convergence of Reynolds
    # Calculate mean velocity
    mean_vel_rec = vel_rec.mean(axis=0)
    mean_reynolds_stress = np.empty((len(vel_rec),3,3))
    for ii in range(0,len(vel_rec)):
        reynolds_stress = np.empty((ii+1,3,3))
        for jj in range(0,ii+1):
            # Calculate velocity fluctuations
            vel_rec_fluct = vel_rec[jj,:] - mean_vel_rec
            # Reynolds stresses as outer product of fluctuations
            reynolds_stress[jj,:,:] = np.outer(vel_rec_fluct, \
                                    vel_rec_fluct)
        # Calculate mean Reynolds stresses
        mean_reynolds_stress[ii,:,:] = reynolds_stress.mean(axis=0)

    # Calculate mean velocity
    mean_vel_true = vel_true_bubble.mean(axis=0)
    mean_reynolds_stress_true = np.empty((len(vel_true_bubble),3,3))
    for ii in range(0,len(vel_true_bubble)):
        reynolds_stress_true = np.empty((ii+1,3,3))
        for jj in range(0,ii+1):
            # Calculate velocity fluctuations
            vel_true_fluct = vel_true_bubble[jj,:] - mean_vel_true
            # Reynolds stresses as outer product of fluctuations
            reynolds_stress_true[jj,:,:] = np.outer(vel_true_fluct, \
                                    vel_true_fluct)
        # Calculate mean Reynolds stresses
        mean_reynolds_stress_true[ii,:,:] = reynolds_stress_true.mean(axis=0)

    # Compare input / output
    overview = pd.DataFrame(columns=['config','CP','DP','DP_bubble','DP_reconstructed'], \
                            index=['Ux','Uy','Uz','Tau_xx','Tau_yy','Tau_zz','Tau_xy','Tau_xz','Tau_yz','T_Ix','T_Iy','T_Iz'])
    # Configuration file
    fp_vel = np.asarray(fp_vel)
    fp_ti = np.asarray(fp_ti)
    overview['config']['Ux'] = fp_vel[0]
    overview['config']['Uy'] = fp_vel[1]
    overview['config']['Uz'] = fp_vel[2]
    overview['config']['Tau_xx'] = (fp_vel[0]*fp_ti[0])*(fp_vel[0]*fp_ti[0])
    overview['config']['Tau_yy'] = (fp_vel[0]*fp_ti[1])*(fp_vel[0]*fp_ti[1])
    overview['config']['Tau_zz'] = (fp_vel[0]*fp_ti[2])*(fp_vel[0]*fp_ti[2])
    overview['config']['Tau_xy'] = np.nan
    overview['config']['Tau_xz'] = np.nan
    overview['config']['Tau_yz'] = np.nan
    if fp_vel[1] == 0.0:
        overview['config']['Tau_xy'] = 0.0
        overview['config']['Tau_yz'] = 0.0
    if fp_vel[2] == 0.0:
        overview['config']['Tau_xz'] = 0.0
        overview['config']['Tau_yz'] = 0.0
    overview['config']['T_Ix'] = fp_ti[0]*math.sqrt(fp_vel.dot(fp_vel))
    overview['config']['T_Iy'] = fp_ti[1]*math.sqrt(fp_vel.dot(fp_vel))
    overview['config']['T_Iz'] = fp_ti[2]*math.sqrt(fp_vel.dot(fp_vel))
    # Continuous phase time series
    overview['CP']['Ux'] = vel_mean_true_fluid[0]
    overview['CP']['Uy'] = vel_mean_true_fluid[1]
    overview['CP']['Uz'] = vel_mean_true_fluid[2]
    overview['CP']['Tau_xx'] = rs_true_fluid[0,0]
    overview['CP']['Tau_yy'] = rs_true_fluid[1,1]
    overview['CP']['Tau_zz'] = rs_true_fluid[2,2]
    overview['CP']['Tau_xy'] = rs_true_fluid[0,1]
    overview['CP']['Tau_xz'] = rs_true_fluid[0,2]
    overview['CP']['Tau_yz'] = rs_true_fluid[1,2]
    overview['CP']['T_Ix'] = Ti_true_fluid[0]
    overview['CP']['T_Iy'] = Ti_true_fluid[1]
    overview['CP']['T_Iz'] = Ti_true_fluid[2]
    # Dispersed phase time series (full time series)
    overview['DP']['Ux'] = vel_mean_true[0]
    overview['DP']['Uy'] = vel_mean_true[1]
    overview['DP']['Uz'] = vel_mean_true[2]
    overview['DP']['Tau_xx'] = rs_true[0,0]
    overview['DP']['Tau_yy'] = rs_true[1,1]
    overview['DP']['Tau_zz'] = rs_true[2,2]
    overview['DP']['Tau_xy'] = rs_true[0,1]
    overview['DP']['Tau_xz'] = rs_true[0,2]
    overview['DP']['Tau_yz'] = rs_true[1,2]
    overview['DP']['T_Ix'] = Ti_true[0]
    overview['DP']['T_Iy'] = Ti_true[1]
    overview['DP']['T_Iz'] = Ti_true[2]
    # Dispersed phase time series (only bubbles-probe interations
    overview['DP_bubble']['Ux'] = mean_vel_true_bubble[0]
    overview['DP_bubble']['Uy'] = mean_vel_true_bubble[1]
    overview['DP_bubble']['Uz'] = mean_vel_true_bubble[2]
    overview['DP_bubble']['Tau_xx'] = mean_rs_true_bubble[0,0]
    overview['DP_bubble']['Tau_yy'] = mean_rs_true_bubble[1,1]
    overview['DP_bubble']['Tau_zz'] = mean_rs_true_bubble[2,2]
    overview['DP_bubble']['Tau_xy'] = mean_rs_true_bubble[0,1]
    overview['DP_bubble']['Tau_xz'] = mean_rs_true_bubble[0,2]
    overview['DP_bubble']['Tau_yz'] = mean_rs_true_bubble[1,2]
    overview['DP_bubble']['T_Ix'] = turbulent_intensity_bubble[0]
    overview['DP_bubble']['T_Iy'] = turbulent_intensity_bubble[1]
    overview['DP_bubble']['T_Iz'] = turbulent_intensity_bubble[2]
    # Dispersed phase time series (reconstructed)
    overview['DP_reconstructed']['Ux'] = vel_mean_rec[0]
    overview['DP_reconstructed']['Uy'] = vel_mean_rec[1]
    overview['DP_reconstructed']['Uz'] = vel_mean_rec[2]
    overview['DP_reconstructed']['Tau_xx'] = rs_rec[0,0]
    overview['DP_reconstructed']['Tau_yy'] = rs_rec[1,1]
    overview['DP_reconstructed']['Tau_zz'] = rs_rec[2,2]
    overview['DP_reconstructed']['Tau_xy'] = rs_rec[0,1]
    overview['DP_reconstructed']['Tau_xz'] = rs_rec[0,2]
    overview['DP_reconstructed']['Tau_yz'] = rs_rec[1,2]
    overview['DP_reconstructed']['T_Ix'] = Ti_rec[0]
    overview['DP_reconstructed']['T_Iy'] = Ti_rec[1]
    overview['DP_reconstructed']['T_Iz'] = Ti_rec[2]
    overview.to_csv(path / 'overview.csv')

    # Write number of bubbles
    file1 = open(path / "n_bubbles.txt","w")
    file1.write(f"Input: {len(t_b_size)}\n")
    file1.write(f"Reconstructed: {len(t_rec)}\n")
    file1.close()

    # Plot parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['xtick.major.pad']='6'
    plt.rcParams['ytick.major.pad']='6'
    plt.rcParams['mathtext.fontset'] = 'cm'

    # Plot velocity time series
    color = [0.2,0.2,0.2]
    lw = 0.2
    fig, axs = plt.subplots(3,figsize=(6,4))
    axs[0].plot(t_true,vel_true[:,0],color=color,lw=lw)
    axs[1].plot(t_true,vel_true[:,1],color=color,lw=lw)
    axs[2].plot(t_true,vel_true[:,2],color=color,lw=lw)
    axs[0].plot(np.mean(t_rec,axis=1), vel_rec[:,0], color='r', \
                lw=0,marker='o')
    axs[1].plot(np.mean(t_rec,axis=1), vel_rec[:,1], color='r', \
                lw=0,marker='o')
    axs[2].plot(np.mean(t_rec,axis=1), vel_rec[:,2], color='r', \
                lw=0,marker='o')
    axs[0].set_ylabel('$u_x$ [m/s]')
    axs[1].set_ylabel('$u_y$ [m/s]')
    axs[2].set_ylabel('$u_z$ [m/s]')
    axs[2].set_xlabel('$t$ [s]')
    lim = [[fp_vel[0] - 1.2*max(abs(vel_true[:,0]-fp_vel[0])),
            fp_vel[0] + 1.2*max(abs(vel_true[:,0]-fp_vel[0]))],
           [fp_vel[1] - 1.2*max(abs(vel_true[:,1]-fp_vel[1])),
            fp_vel[1] + 1.2*max(abs(vel_true[:,1]-fp_vel[1]))],
           [fp_vel[2] - 1.2*max(abs(vel_true[:,2]-fp_vel[2])),
            fp_vel[2] + 1.2*max(abs(vel_true[:,2]-fp_vel[2]))]]
    axs[0].set_ylim([lim[0][0],lim[0][1]])
    axs[1].set_ylim([lim[1][0],lim[1][1]])
    axs[2].set_ylim([lim[2][0],lim[2][1]])
    axs[0].set_xlim([fp_dur/2.0,fp_dur/2.0+0.1])
    axs[1].set_xlim([fp_dur/2.0,fp_dur/2.0+0.1])
    axs[2].set_xlim([fp_dur/2.0,fp_dur/2.0+0.1])
    plt.tight_layout()
    fig.savefig(path / 'velocity_05s.svg',dpi=300)


    fig, axs = plt.subplots(3,figsize=(6,4))
    axs[0].plot(np.linspace(1,len(cumulative_mean_AE),len(cumulative_mean_AE)),cumulative_mean_AE[:,0],color='k',lw=lw)
    axs[1].plot(np.linspace(1,len(cumulative_mean_AE),len(cumulative_mean_AE)),cumulative_mean_AE[:,1],color='k',lw=lw)
    axs[2].plot(np.linspace(1,len(cumulative_mean_AE),len(cumulative_mean_AE)),cumulative_mean_AE[:,2],color='k',lw=lw)
    axs[0].set_ylabel('MAE $u_x$ [m/s]')
    axs[1].set_ylabel('MAE $u_y$ [m/s]')
    axs[2].set_ylabel('MAE $u_z$ [m/s]')
    axs[2].set_xlabel('number of bubbles [-]')
    plt.tight_layout()
    fig.savefig(path / 'AE_convergence.svg',dpi=300)

    fig, axs = plt.subplots(6,figsize=(6,6))
    axs[0].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,0,0]),color='r',lw=1,label='mean')
    axs[1].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,1,1]),color='r',lw=1)
    axs[2].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,2,2]),color='r',lw=1)
    axs[3].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,0,1]),color='r',lw=1)
    axs[4].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,0,2]),color='r',lw=1)
    axs[5].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,1,2]),color='r',lw=1)
    axs[0].set_ylabel(r'$\tau_{xx}$ [m$^2$/s$^2$]')
    axs[1].set_ylabel(r'$\tau_{yy}$ [m$^2$/s$^2$]')
    axs[2].set_ylabel(r'$\tau_{zz}$ [m$^2$/s$^2$]')
    axs[3].set_ylabel(r'$\tau_{xy}$ [m$^2$/s$^2$]')
    axs[4].set_ylabel(r'$\tau_{xz}$ [m$^2$/s$^2$]')
    axs[5].set_ylabel(r'$\tau_{yz}$ [m$^2$/s$^2$]')
    axs[5].set_xlabel('number of bubbles [-]')
    axs[0].set_ylim([10**math.floor(np.log10(min(mean_reynolds_stress[:,0,0]))),10**math.ceil(np.log10(max(mean_reynolds_stress[:,0,0])))])
    axs[1].set_ylim([10**math.floor(np.log10(min(mean_reynolds_stress[:,1,1]))),10**math.ceil(np.log10(max(mean_reynolds_stress[:,1,1])))])
    axs[2].set_ylim([10**math.floor(np.log10(min(mean_reynolds_stress[:,2,2]))),10**math.ceil(np.log10(max(mean_reynolds_stress[:,2,2])))])
    axs[3].set_ylim([-max(abs(mean_reynolds_stress[:,0,1])),max(abs(mean_reynolds_stress[:,0,1]))])
    axs[4].set_ylim([-max(abs(mean_reynolds_stress[:,0,2])),max(abs(mean_reynolds_stress[:,0,2]))])
    axs[5].set_ylim([-max(abs(mean_reynolds_stress[:,1,2])),max(abs(mean_reynolds_stress[:,1,2]))])
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    axs[0].legend()
    plt.tight_layout()
    fig.savefig(path / 'RST_mean_convergence.svg',dpi=300)

    fig, axs = plt.subplots(6,figsize=(6,6))
    axs[0].plot(np.linspace(1,len(reynolds_stress),len(reynolds_stress)),(reynolds_stress[:,0,0]),color='k',lw=lw,label='inst.')
    axs[1].plot(np.linspace(1,len(reynolds_stress),len(reynolds_stress)),(reynolds_stress[:,1,1]),color='k',lw=lw)
    axs[2].plot(np.linspace(1,len(reynolds_stress),len(reynolds_stress)),(reynolds_stress[:,2,2]),color='k',lw=lw)
    axs[3].plot(np.linspace(1,len(reynolds_stress),len(reynolds_stress)),(reynolds_stress[:,0,1]),color='k',lw=lw)
    axs[4].plot(np.linspace(1,len(reynolds_stress),len(reynolds_stress)),(reynolds_stress[:,0,2]),color='k',lw=lw)
    axs[5].plot(np.linspace(1,len(reynolds_stress),len(reynolds_stress)),(reynolds_stress[:,1,2]),color='k',lw=lw)
    axs[0].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,0,0]),color='r',lw=1,label='mean')
    axs[1].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,1,1]),color='r',lw=1)
    axs[2].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,2,2]),color='r',lw=1)
    axs[3].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,0,1]),color='r',lw=1)
    axs[4].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,0,2]),color='r',lw=1)
    axs[5].plot(np.linspace(1,len(mean_reynolds_stress),len(mean_reynolds_stress)),(mean_reynolds_stress[:,1,2]),color='r',lw=1)
    axs[0].set_ylabel(r'$\tau_{xx}$ [m$^2$/s$^2$]')
    axs[1].set_ylabel(r'$\tau_{yy}$ [m$^2$/s$^2$]')
    axs[2].set_ylabel(r'$\tau_{zz}$ [m$^2$/s$^2$]')
    axs[3].set_ylabel(r'$\tau_{xy}$ [m$^2$/s$^2$]')
    axs[4].set_ylabel(r'$\tau_{xz}$ [m$^2$/s$^2$]')
    axs[5].set_ylabel(r'$\tau_{yz}$ [m$^2$/s$^2$]')
    axs[5].set_xlabel('number of bubbles [-]')
    axs[0].set_ylim([10**math.floor(np.log10(min(reynolds_stress[:,0,0]))),10**math.ceil(np.log10(max(reynolds_stress[:,0,0])))])
    axs[1].set_ylim([10**math.floor(np.log10(min(reynolds_stress[:,1,1]))),10**math.ceil(np.log10(max(reynolds_stress[:,1,1])))])
    axs[2].set_ylim([10**math.floor(np.log10(min(reynolds_stress[:,2,2]))),10**math.ceil(np.log10(max(reynolds_stress[:,2,2])))])
    axs[3].set_ylim([-max(abs(reynolds_stress_true[:,0,1])),max(abs(reynolds_stress_true[:,0,1]))])
    axs[4].set_ylim([-max(abs(reynolds_stress_true[:,0,2])),max(abs(reynolds_stress_true[:,0,2]))])
    axs[5].set_ylim([-max(abs(reynolds_stress_true[:,1,2])),max(abs(reynolds_stress_true[:,1,2]))])
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    axs[0].legend()
    plt.tight_layout()
    fig.savefig(path / 'RST_convergence_rec.svg',dpi=300)

    fig, axs = plt.subplots(6,figsize=(6,6))
    axs[0].plot(np.linspace(1,len(reynolds_stress_true),len(reynolds_stress_true)),(reynolds_stress_true[:,0,0]),color='k',lw=lw,label='inst.')
    axs[1].plot(np.linspace(1,len(reynolds_stress_true),len(reynolds_stress_true)),(reynolds_stress_true[:,1,1]),color='k',lw=lw)
    axs[2].plot(np.linspace(1,len(reynolds_stress_true),len(reynolds_stress_true)),(reynolds_stress_true[:,2,2]),color='k',lw=lw)
    axs[3].plot(np.linspace(1,len(reynolds_stress_true),len(reynolds_stress_true)),(reynolds_stress_true[:,0,1]),color='k',lw=lw)
    axs[4].plot(np.linspace(1,len(reynolds_stress_true),len(reynolds_stress_true)),(reynolds_stress_true[:,0,2]),color='k',lw=lw)
    axs[5].plot(np.linspace(1,len(reynolds_stress_true),len(reynolds_stress_true)),(reynolds_stress_true[:,1,2]),color='k',lw=lw)
    axs[0].plot(np.linspace(1,len(mean_reynolds_stress_true),len(mean_reynolds_stress_true)),(mean_reynolds_stress_true[:,0,0]),color='r',lw=1,label='mean')
    axs[1].plot(np.linspace(1,len(mean_reynolds_stress_true),len(mean_reynolds_stress_true)),(mean_reynolds_stress_true[:,1,1]),color='r',lw=1)
    axs[2].plot(np.linspace(1,len(mean_reynolds_stress_true),len(mean_reynolds_stress_true)),(mean_reynolds_stress_true[:,2,2]),color='r',lw=1)
    axs[3].plot(np.linspace(1,len(mean_reynolds_stress_true),len(mean_reynolds_stress_true)),(mean_reynolds_stress_true[:,0,1]),color='r',lw=1)
    axs[4].plot(np.linspace(1,len(mean_reynolds_stress_true),len(mean_reynolds_stress_true)),(mean_reynolds_stress_true[:,0,2]),color='r',lw=1)
    axs[5].plot(np.linspace(1,len(mean_reynolds_stress_true),len(mean_reynolds_stress_true)),(mean_reynolds_stress_true[:,1,2]),color='r',lw=1)
    axs[0].set_ylabel(r'$\tau_{xx}$ [m$^2$/s$^2$]')
    axs[1].set_ylabel(r'$\tau_{yy}$ [m$^2$/s$^2$]')
    axs[2].set_ylabel(r'$\tau_{zz}$ [m$^2$/s$^2$]')
    axs[3].set_ylabel(r'$\tau_{xy}$ [m$^2$/s$^2$]')
    axs[4].set_ylabel(r'$\tau_{xz}$ [m$^2$/s$^2$]')
    axs[5].set_ylabel(r'$\tau_{yz}$ [m$^2$/s$^2$]')
    axs[5].set_xlabel('number of bubbles [-]')
    axs[0].set_ylim([10**math.floor(np.log10(min(reynolds_stress[:,0,0]))),10**math.ceil(np.log10(max(reynolds_stress[:,0,0])))])
    axs[1].set_ylim([10**math.floor(np.log10(min(reynolds_stress[:,1,1]))),10**math.ceil(np.log10(max(reynolds_stress[:,1,1])))])
    axs[2].set_ylim([10**math.floor(np.log10(min(reynolds_stress[:,2,2]))),10**math.ceil(np.log10(max(reynolds_stress[:,2,2])))])
    axs[3].set_ylim([-max(abs(reynolds_stress_true[:,0,1])),max(abs(reynolds_stress_true[:,0,1]))])
    axs[4].set_ylim([-max(abs(reynolds_stress_true[:,0,2])),max(abs(reynolds_stress_true[:,0,2]))])
    axs[5].set_ylim([-max(abs(reynolds_stress_true[:,1,2])),max(abs(reynolds_stress_true[:,1,2]))])
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    axs[0].legend()
    plt.tight_layout()
    fig.savefig(path / 'RST_convergence_true.svg',dpi=300)

    lw=0.5
    fig, axs = plt.subplots(3,figsize=(6,4))
    axs[0].plot(AE_vel.index,AE_vel['Ux'],color='k',lw=lw,label='MAE $u_i$')
    axs[1].plot(AE_vel.index,AE_vel['Uy'],color='k',lw=lw,label='MAE $u_i$')
    axs[2].plot(AE_vel.index,AE_vel['Uz'],color='k',lw=lw,label='MAE $u_i$')
    axs[2].plot([],[],color='r',lw=lw,label='$u_i$')
    axs[0].set_ylabel('MAE $u_x$ [m/s]')
    axs[1].set_ylabel('MAE $u_y$ [m/s]')
    axs[2].set_ylabel('MAE $u_z$ [m/s]')
    axs[2].set_xlabel('time $t$ [s]')
    axs[0].set_ylim([0,1.2*max(AE_vel['Ux'])])
    axs[1].set_ylim([0,1.2*max(AE_vel['Uy'])])
    axs[2].set_ylim([0,1.2*max(AE_vel['Uz'])])
    axs[0].set_xlim([0,10])
    axs[1].set_xlim([0,10])
    axs[2].set_xlim([0,10])
    ax1 = axs[0].twinx()
    ax2 = axs[1].twinx()
    ax3 = axs[2].twinx()
    ax1.plot(t_true,vel_true[:,0],color='r',lw=lw)
    ax2.plot(t_true,vel_true[:,1],color='r',lw=lw)
    ax3.plot(t_true,vel_true[:,2],color='r',lw=lw)
    ax1.set_ylabel('$u_x$ [m/s]')
    ax2.set_ylabel('$u_y$ [m/s]')
    ax3.set_ylabel('$u_z$ [m/s]')
    ax1.set_ylim([np.mean(vel_true[:,0])-1.3*max(vel_true[:,0]-np.mean(vel_true[:,0])),np.mean(vel_true[:,0])+1.3*max(vel_true[:,0]-np.mean(vel_true[:,0]))])
    ax2.set_ylim([-1.1*max(vel_true[:,1]),1.1*max(vel_true[:,1])])
    ax3.set_ylim([-1.1*max(vel_true[:,2]),1.1*max(vel_true[:,2])])
    axs[2].legend(loc=1,ncol=2, bbox_to_anchor=(0.7,-0.7))
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.88, top=0.95, wspace=None, hspace=0.5)
    fig.savefig(path / 'AE_u_ts.svg',dpi=300)

    total_duration = t_true[-1]-t_true[0]
    ts_frequency = int((len(t_true)-1)/total_duration)
    dudt = np.diff(vel_true,axis=0)*ts_frequency
    fig, axs = plt.subplots(3,figsize=(6,4))
    axs[0].plot(AE_vel.index,AE_vel['Ux'],color='k',lw=lw,label='MAE $u_i$')
    axs[1].plot(AE_vel.index,AE_vel['Uy'],color='k',lw=lw,label='MAE $u_i$')
    axs[2].plot(AE_vel.index,AE_vel['Uz'],color='k',lw=lw,label='MAE $u_i$')
    axs[2].plot([],[],color='r',lw=lw,label='d$u_i$/d$t$')
    axs[0].set_ylabel('MAE $u_x$ [m/s]')
    axs[1].set_ylabel('MAE $u_y$ [m/s]')
    axs[2].set_ylabel('MAE $u_z$ [m/s]')
    axs[2].set_xlabel('time $t$ [s]')
    axs[0].set_ylim([0,1.2*max(AE_vel['Ux'])])
    axs[1].set_ylim([0,1.2*max(AE_vel['Uy'])])
    axs[2].set_ylim([0,1.2*max(AE_vel['Uz'])])
    axs[0].set_xlim([0,10])
    axs[1].set_xlim([0,10])
    axs[2].set_xlim([0,10])
    ax1 = axs[0].twinx()
    ax2 = axs[1].twinx()
    ax3 = axs[2].twinx()
    ax1.plot(t_true[:-1],dudt[:,0],color='r',lw=lw)
    ax2.plot(t_true[:-1],dudt[:,1],color='r',lw=lw)
    ax3.plot(t_true[:-1],dudt[:,2],color='r',lw=lw)
    ax1.set_ylabel('d$u_x$/d$t$ [m/s]')
    ax2.set_ylabel('d$u_y$/d$t$ [m/s]')
    ax3.set_ylabel('d$u_z$/d$t$ [m/s]')
    ax1.set_ylim([-1.1*max(dudt[:,0]),1.1*max(dudt[:,0])])
    ax2.set_ylim([-1.1*max(dudt[:,1]),1.1*max(dudt[:,1])])
    ax3.set_ylim([-1.1*max(dudt[:,2]),1.1*max(dudt[:,2])])
    axs[2].legend(loc=1,ncol=2, bbox_to_anchor=(0.7,-0.7))
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.88, top=0.95, wspace=None, hspace=0.5)
    fig.savefig(path / 'AE_dudt_ts.svg',dpi=300)

    # Histogram of velocity error ux
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(E_vel['Ux'].values, \
            bins=100,density=True, color='b', alpha=0.7,\
            range=(min(E_vel['Ux'].values),max(E_vel['Ux'].values)))
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Error $u_x$ [m/s]')
    plt.xlim([min(bins_p),max(bins_p)])
    plt.tight_layout()
    fig.savefig(path / 'histogram_error_ux.svg',dpi=300)

    # Histogram of velocity error ux
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(E_vel['Uy'].values, \
            bins=100,density=True, color='b', alpha=0.7,\
            range=(min(E_vel['Uy'].values),max(E_vel['Uy'].values)))
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Error $u_y$ [m/s]')
    plt.xlim([-max(abs(bins_p)),max(abs(bins_p))])
    plt.tight_layout()
    fig.savefig(path / 'histogram_error_uy.svg',dpi=300)

    # Histogram of velocity error ux
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(E_vel['Uz'].values, \
            bins=100,density=True, color='b', alpha=0.7,\
            range=(min(E_vel['Uz'].values),max(E_vel['Uz'].values)))
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Error $u_z$ [m/s]')
    plt.xlim([-max(abs(bins_p)),max(abs(bins_p))])
    plt.tight_layout()
    fig.savefig(path / 'histogram_error_uz.svg',dpi=300)

    bin_range=5
    # Histogram of velocity error ux
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(np.clip(RE_vel['Ux'].values,-1,1),\
            bins=100,density=True, color='b', alpha=0.7,\
            range=(min(RE_vel['Ux'].values),max(RE_vel['Ux'].values)))
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'RE $u_x$ [-]')
    plt.xlim([min(bins_p),max(bins_p)])
    plt.tight_layout()
    fig.savefig(path / 'histogram_ARE_ux.svg',dpi=300)

    # Histogram of velocity error ux
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(np.clip(RE_vel['Uy'].values,-bin_range,bin_range), \
            bins=100,density=True, color='b', alpha=0.7)
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'RE $u_y$ [-]')
    plt.xlim([-bin_range,bin_range])
    plt.tight_layout()
    fig.savefig(path / 'histogram_ARE_uy.svg',dpi=300)

    # Histogram of velocity error ux
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(np.clip(RE_vel['Uz'].values,-bin_range,bin_range), \
            bins=100,density=True, color='b', alpha=0.7)
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'RE $u_z$ [-]')
    plt.xlim([-bin_range,bin_range])
    plt.tight_layout()
    fig.savefig(path / 'histogram_ARE_uz.svg',dpi=300)

    # Histogram of velocities
    var_names = ['x','y','z']
    # Calculate the magnitude of the input velocities and standard dev.
    fp_vel_mag = np.sqrt(fp_vel.dot(fp_vel))
    sigma = fp_vel_mag * np.asarray(fp_ti)
    fig, axs = plt.subplots(ncols=3,\
            figsize=(3*2.5,2.5))
    for jj,var_name in enumerate(var_names):
        # Theoretical input distribution
        x = np.linspace(fp_vel[jj] - 3*sigma[jj], fp_vel[jj] + 3*sigma[jj], 10000)
        input_vel = stats.norm.pdf(x, fp_vel[jj], sigma[jj])
        axs[jj].plot(x, input_vel, color='k', label='input cont.')
        (n_c, bins_c, patches_c) = axs[jj].hist(vel_true_fluid[:,jj], \
            bins=50, density=True, color='b', alpha=0.7, label='sim. cont.',\
            range=(min(vel_true_fluid[:,jj]),max(vel_true_fluid[:,jj])))
        (n_c, bins_c, patches_c) = axs[jj].hist(vel_true[:,jj], \
            bins=50, density=True, color='r', alpha=0.7, label='sim. disp.',\
            range=(min(vel_true[:,jj]),max(vel_true[:,jj])))
        (n_c, bins_c, patches_c) = axs[jj].hist(vel_rec[:,jj], \
            bins=50, density=True, color='g', alpha=0.7, label='reconst. disp.',\
            range=(min(vel_rec[:,jj]),max(vel_rec[:,jj])))
        if jj == 0:
            axs[jj].set_ylabel('Frequency [-]')
            axs[jj].legend(loc=3,fontsize=10,bbox_to_anchor=(-0.15,-0.6),ncol=4,frameon=False)
        axs[jj].set_xlabel(f'Velocity $u_{var_name}$ [m/s]')
        axs[jj].set_xlim([min(x),max(x)])
        if jj == 1:
            axs[jj].set_yscale('log')
        if jj == 2:
            axs[jj].set_yscale('log')
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=None)
    fig.savefig(path / 'velocity_histrograms.svg',dpi=300)

    # Histogram of velocities
    var_names = ['x','y','z']
    # Calculate the magnitude of the input velocities and standard dev.
    fp_vel = np.asarray(fp_vel)
    fp_vel_mag = np.sqrt(fp_vel.dot(fp_vel))
    sigma = fp_vel_mag * np.asarray(fp_ti)
    fig, axs = plt.subplots(ncols=3,\
            figsize=(3*2.5,2.5))
    for jj,var_name in enumerate(var_names):
        (n_sd, bins_sd, patches_sd) = axs[jj].hist(vel_true[:,jj], \
            bins=50, density=True, color='r', alpha=0.7, label='sim. disp.',\
            range=(min(vel_true[:,jj]),max(vel_true[:,jj])))
        (n_rd, bins_rd, patches_rd) = axs[jj].hist(vel_rec[:,jj], \
            bins=50, density=True, color='g', alpha=0.7, label='reconst. disp.',\
            range=(min(vel_rec[:,jj]),max(vel_rec[:,jj])))
        if jj == 0:
            axs[jj].set_ylabel('Frequency [-]')
            axs[jj].legend(loc=3,fontsize=10,bbox_to_anchor=(-0.15,-0.6),ncol=4,frameon=False)
        axs[jj].set_xlabel(f'Velocity $u_{var_name}$ [m/s]')
        axs[jj].set_xlim([min(bins_rd),max(bins_rd)])
        if jj == 1:
            axs[jj].set_yscale('log')
        if jj == 2:
            axs[jj].set_yscale('log')
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.95, wspace=0.3, hspace=None)
    fig.savefig(path / 'velocity_histrograms_disp.svg',dpi=300)

if __name__ == "__main__":
    main()