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
    # Read the velocity time series
    vel = reader.getDataSet('fluid/velocity')[:]
    # Read the time vector
    t_vel = reader.getDataSet('fluid/time')[:]
    # Read the bubble size
    b_size = reader.getDataSet('bubbles/size')[:]
    # Read the time vector
    t_b_size = reader.getDataSet('bubbles/arrival_time')[:]
    reader.close()
    # Create a H5-file reader
    reader = H5Reader(path / 'reconstructed.h5')
    # Read the reconstructed velocity time series
    vel_rec = reader.getDataSet('fluid/velocity')[:]
    # Read the reconstructed time vector
    t_vel_rec = reader.getDataSet('fluid/interaction_times')[:]
    # Read the bubble sizecd t
    b_size_rec = reader.getDataSet('bubbles/diameters')[:]
    t_b_size_rec = reader.getDataSet('bubbles/interaction_times')[:]
    reader.close()

    # Convert to pandas array
    RMSD_vel_reco = np.empty((len(vel_rec),3))
    RMSD_vel_true = np.empty((len(vel_rec),3))
    for ii in range(0,len(t_vel_rec)):
        # Get the range of the velocity timeseries that encloses the timeframe
        # of bubble-probe interaction (tfbpi)
        id_t_min = max(bisect.bisect_right(t_vel, t_vel_rec[ii,0])-1, 0)
        id_t_max = min(bisect.bisect_left(t_vel, t_vel_rec[ii,1]),len(t_vel))
        # Mean of the velocity time series in the tfbpi
        mean_vel = np.mean(vel[id_t_min:(id_t_max+1),:],axis=0)
        # The mean squared deviation (MSD) of the reconstructed and true
        # velocity
        MSD_vel_rec = np.mean((vel_rec[ii,:]- vel[id_t_min:(id_t_max+1),:])**2,\
            axis=0)
        MSD_vel = np.mean((vel[id_t_min:(id_t_max+1),:] \
                - np.mean(vel[id_t_min:(id_t_max+1),:],axis=0))**2,axis=0)
        # The root of the mean squared deviations (RMSD)
        RMSD_vel_reco[ii,:] = np.sqrt(MSD_vel_rec / (id_t_max-id_t_min))
        RMSD_vel_true[ii,:] = np.sqrt(MSD_vel / (id_t_max-id_t_min))

    # Convert to DataFrames
    RMSD_vel_reco = pd.DataFrame(RMSD_vel_reco, index=np.mean(t_vel_rec,axis=1),
        columns=['Ux','Uy','Uz'])
    RMSD_vel_true = pd.DataFrame(RMSD_vel_true, index=np.mean(t_vel_rec,axis=1),
        columns=['Ux','Uy','Uz'])

    # Calculate mean RMSD for the true and reconstructed velocity and their
    # ratio epsilon
    summary_vel = pd.DataFrame(np.nan, \
            index=['MRMSD_true','MRMSD_rec','mean_epsilon'], \
            columns=['Ux','Uy','Uz'])
    summary_vel.loc['MRMSD_true',:] = RMSD_vel_true.mean()
    summary_vel.loc['MRMSD_rec',:] = RMSD_vel_reco.mean()
    summary_vel.loc['mean_epsilon',:] = ((RMSD_vel_reco/RMSD_vel_true)).mean()

    # Calculate error for bubble size
    b_size = pd.DataFrame(b_size,index=t_b_size, columns=['A','B','C'])
    # Calculate volumen equivalent diameter
    b_size['D'] = (b_size['A']*b_size['B']*b_size['C'])**(1.0/3.0)
    # Convert to pandas array
    b_size_rec = pd.DataFrame(b_size_rec,index=np.mean(t_b_size_rec,axis=1), \
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
    errors_D = pd.DataFrame(error_D.values, index=np.mean(t_vel_rec,axis=1),
        columns=['E'])
    errors_D['RE'] = rel_error_D
    # Calculate squared error (SE) and squared relative errors (SRE)
    errors_D['SE'] = errors_D['E']**2
    errors_D['SRE'] = errors_D['RE']**2

    # Calculate mean average error (MAE) and root mean square error (RMSE)
    # for the bubble diameter
    summary_b_size = pd.DataFrame(np.nan, index=['MAE','RMSE','MARE','RMSRE'],\
                            columns=['D'])

    # Store summary of errors
    summary_b_size['D']['MAE'] = errors_D['E'].abs().mean()
    summary_b_size['D']['MARE'] = errors_D['RE'].abs().mean()
    summary_b_size['D']['RMSE'] = math.sqrt(errors_D['SE'].mean())
    summary_b_size['D']['RMSRE'] = math.sqrt(errors_D['SRE'].abs().mean())

    # Write output
    error_D.to_csv(path / 'errors_D.csv', index=True,index_label='t')
    RMSD_vel_reco.to_csv(path / 'RMSD_vel_reco.csv', index=True,index_label='t')
    RMSD_vel_true.to_csv(path / 'RMSD_vel_true.csv', index=True,index_label='t')
    summary_b_size.to_csv(path / 'error_summary_bubble_size.csv', index=True)
    summary_vel.to_csv(path / 'error_summary_velocity.csv', index=True)

    # Write number of bubbles
    file1 = open(path / "n_bubbles.txt","w")
    file1.write(f"Input: {len(t_b_size)}\n")
    file1.write(f"Reconstructed: {len(t_b_size_rec)}\n")
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
    axs[0].plot(t_vel,vel[:,0],color=color,lw=lw)
    axs[1].plot(t_vel,vel[:,1],color=color,lw=lw)
    axs[2].plot(t_vel,vel[:,2],color=color,lw=lw)
    axs[0].plot(np.mean(t_vel_rec,axis=1), vel_rec[:,0], color='r', \
                lw=0,marker='o')
    axs[1].plot(np.mean(t_vel_rec,axis=1), vel_rec[:,1], color='r', \
                lw=0,marker='o')
    axs[2].plot(np.mean(t_vel_rec,axis=1), vel_rec[:,2], color='r', \
                lw=0,marker='o')
    axs[0].set_ylabel('$u_x$ [m/s]')
    axs[1].set_ylabel('$u_y$ [m/s]')
    axs[2].set_ylabel('$u_z$ [m/s]')
    axs[2].set_xlabel('$t$ [s]')
    lim = [[fp_vel[0] - 1.2*max(abs(vel[:,0]-fp_vel[0])),
            fp_vel[0] + 1.2*max(abs(vel[:,0]-fp_vel[0]))],
           [fp_vel[1] - 1.2*max(abs(vel[:,1]-fp_vel[1])),
            fp_vel[1] + 1.2*max(abs(vel[:,1]-fp_vel[1]))],
           [fp_vel[2] - 1.2*max(abs(vel[:,2]-fp_vel[2])),
            fp_vel[2] + 1.2*max(abs(vel[:,2]-fp_vel[2]))]]
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
    bubbles = np.random.uniform(low=0, high=len(t_vel_rec), size=n)
    delta_t=0.001
    fig, axs = plt.subplots(int(math.sqrt(n)),int(math.sqrt(n)),figsize=(6,6))
    axs=axs.reshape(-1)
    for ii,bubble in enumerate(bubbles):
        bid = int(bubble)
        axs[ii].plot(t_vel,vel[:,0],color='k',lw=1, label='true velocity')
        axs[ii].hlines(vel_rec[bid,0],t_vel_rec[bid,0],t_vel_rec[bid,1], \
                    color='b',label='reconstr. velocity',zorder=100)
        axs[ii].set_xlim([t_vel_rec[bid,0]-delta_t,t_vel_rec[bid,1]+delta_t])
        axs[ii].set_ylim([0.9*min(vel[:,0][(t_vel>t_vel_rec[bid,0]) \
                                         & (t_vel<t_vel_rec[bid,1])]), \
                          1.1*max(vel[:,0][(t_vel>t_vel_rec[bid,0]) \
                                         &(t_vel<t_vel_rec[bid,1])])])
        axs[ii].axvspan(t_vel_rec[bid,0], t_vel_rec[bid,1], alpha=0.5, \
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

if __name__ == "__main__":
    main()