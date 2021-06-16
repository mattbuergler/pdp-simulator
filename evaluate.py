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

    # Parse velocity time series
    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
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

    np.random.seed(42)
    n = 9
    bubbles = np.random.uniform(low=0, high=len(t_rec), size=n)
    delta_t=0.001
    fig, axs = plt.subplots(int(math.sqrt(n)),int(math.sqrt(n)),figsize=(6,6))
    axs=axs.reshape(-1)
    for ii,bubble in enumerate(bubbles):
        bid = int(bubble)
        axs[ii].plot(t_true,vel_true[:,0],color='k',lw=1, label='true velocity')
        axs[ii].hlines(vel_rec[bid,0],t_rec[bid,0],t_rec[bid,1], \
                    color='b',label='reconstr. velocity',zorder=100)
        axs[ii].set_xlim([t_rec[bid,0]-delta_t,t_rec[bid,1]+delta_t])
        axs[ii].set_ylim([0.9*min(vel_true[:,0][(t_true>t_rec[bid,0]) \
                                         & (t_true<t_rec[bid,1])]), \
                          1.1*max(vel_true[:,0][(t_true>t_rec[bid,0]) \
                                         &(t_true<t_rec[bid,1])])])
        axs[ii].axvspan(t_rec[bid,0], t_rec[bid,1], alpha=0.5, \
                    color='r',label='bubble-probe interaction',zorder=10)
        if (ii == n-int(math.sqrt(n))):
            axs[ii].legend(loc=4,bbox_to_anchor=(4,-0.8),ncol=3)
        if (ii % int(math.sqrt(n)) == 0):
            axs[ii].set_ylabel('Velocity [m/s]')
        if (ii >= n-int(math.sqrt(n))):
            axs[ii].set_xlabel('Time [s]')
    plt.subplots_adjust(left=0.1, bottom=0.2, right=0.95, top=0.95, \
                        wspace=0.4, hspace=0.3)
    #plt.tight_layout()
    fig.savefig(path / 'velocity_variation.svg',dpi=300)

    fig, axs = plt.subplots(3,figsize=(6,4))
    axs[0].plot(np.linspace(1,len(cumulative_mean_AE),len(cumulative_mean_AE)),cumulative_mean_AE[:,0],color='k',lw=lw)
    axs[1].plot(np.linspace(1,len(cumulative_mean_AE),len(cumulative_mean_AE)),cumulative_mean_AE[:,1],color='k',lw=lw)
    axs[2].plot(np.linspace(1,len(cumulative_mean_AE),len(cumulative_mean_AE)),cumulative_mean_AE[:,2],color='k',lw=lw)
    axs[0].set_ylabel('MAE $u_x$ [m/s]')
    axs[1].set_ylabel('MAE $u_y$ [m/s]')
    axs[2].set_ylabel('MAE $u_z$ [m/s]')
    axs[2].set_xlabel('number of bubbles [-]')
    # lim = [[fp_vel[0] - 1.2*max(abs(vel_true[:,0]-fp_vel[0])),
    #         fp_vel[0] + 1.2*max(abs(vel_true[:,0]-fp_vel[0]))],
    #        [fp_vel[1] - 1.2*max(abs(vel_true[:,1]-fp_vel[1])),
    #         fp_vel[1] + 1.2*max(abs(vel_true[:,1]-fp_vel[1]))],
    #        [fp_vel[2] - 1.2*max(abs(vel_true[:,2]-fp_vel[2])),
    #         fp_vel[2] + 1.2*max(abs(vel_true[:,2]-fp_vel[2]))]]
    # axs[0].set_ylim([lim[0][0],lim[0][1]])
    # axs[1].set_ylim([lim[1][0],lim[1][1]])
    # axs[2].set_ylim([lim[2][0],lim[2][1]])
    # axs[0].set_xlim([fp_dur/2.0,fp_dur/2.0+0.1])
    # axs[1].set_xlim([fp_dur/2.0,fp_dur/2.0+0.1])
    # axs[2].set_xlim([fp_dur/2.0,fp_dur/2.0+0.1])
    plt.tight_layout()
    fig.savefig(path / 'AE_convergence.svg',dpi=300)

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

    # Accelrations for one sampling frequencies
    fs1=7
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(E_vel['Ux'].values, \
            bins=100,density=True, color='b', alpha=0.7,\
            range=(min(E_vel['Ux'].values),max(E_vel['Ux'].values)))
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Error $u_x$ [m/s]')
    plt.xlim([-max(abs(bins_p)),max(abs(bins_p))])
    plt.tight_layout()
    fig.savefig(path / 'histogram_error_ux.svg',dpi=300)

    # Accelrations for one sampling frequencies
    fs1=7
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(E_vel['Uy'].values, \
            bins=100,density=True, color='b', alpha=0.7,\
            range=(min(E_vel['Uy'].values),max(E_vel['Uy'].values)))
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Error $u_y$ [m/s]')
    plt.xlim([-max(abs(bins_p)),max(abs(bins_p))])
    plt.tight_layout()
    fig.savefig(path / 'histogram_error_uy.svg',dpi=300)

    # Accelrations for one sampling frequencies
    fs1=7
    fig = plt.figure(figsize=(3.0,2.5))
    (n_p, bins_p, patches_p) = plt.hist(E_vel['Uz'].values, \
            bins=100,density=True, color='b', alpha=0.7,\
            range=(min(E_vel['Uz'].values),max(E_vel['Uz'].values)))
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Error $u_z$ [m/s]')
    plt.xlim([-max(abs(bins_p)),max(abs(bins_p))])
    plt.tight_layout()
    fig.savefig(path / 'histogram_error_uz.svg',dpi=300)

if __name__ == "__main__":
    main()