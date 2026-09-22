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
from matplotlib import pyplot as plt
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


def main():
    """
        Plot velocity time series
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
    reader = H5Reader(path / 'reconstructed.h5')
    # Read the time vector
    time_reconst = reader.getDataSet('bubbles/interaction_times')[:]
    # Read the bubbles velocity
    velocity_reconst = reader.getDataSet('bubbles/velocity')[:]
    reader.close()

    # Parse velocity time series
    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the time vector
    t_fluid = reader.getDataSet('fluid/time')[:]
    # Read the velocity vector
    u_fluid = reader.getDataSet('fluid/velocity')[:]
    # Read the mean velocity
    u_mean_fluid = reader.getDataSet('fluid/mean_velocity')[:]
    # Read the turbulent intensity
    turbulent_intensity_fluid = reader.getDataSet('fluid/turbulent_intensity')[:]
    # Read the Reynolds stresses
    reynolds_stress_fluid = reader.getDataSet('fluid/reynold_stresses')[:]
    # Read the bubble velocity vector
    u_bubbles = reader.getDataSet('bubbles/mean_velocity')[:]
    try:
        pierced_index = reader.getDataSet('bubbles/pierced_bubbles')[:]
    except:
        print(f'Could no read dataset <bubbles/pierced_bubbles> from {flow_data_file.name}')
        pierced_index = np.ones((len(u_bubbles),2))
    reader.close()
    n_bubbles = sum(~np.isnan(u_bubbles[:,0]))
    pierced_index = pierced_index.astype('float')
    pierced_index[pierced_index == 0] = np.nan
    not_pierced_index = np.any(np.isnan(pierced_index), axis=1)
    u_pierced_bubbles = np.copy(u_bubbles)
    u_pierced_bubbles[not_pierced_index, :] = np.nan
    n_pierced_bubbles = sum(~np.isnan(u_pierced_bubbles[:,0]))
    n_bubbles = sum(~np.isnan(u_bubbles[:,0]))
    print(n_bubbles)
    print(n_pierced_bubbles)

    cmap = plt.cm.get_cmap('viridis')
    plt.rcParams['font.size'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['figure.titlesize'] = 10
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['axes.labelsize'] = 9
    plt.rcParams["legend.frameon"] = False

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
    plt.rcParams['xtick.major.pad']='6'
    plt.rcParams['ytick.major.pad']='6'
    plt.rcParams["legend.frameon"]
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['xtick.major.pad']='6'
    plt.rcParams['ytick.major.pad']='6'
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,0],color='k')
    plt.xlabel('$t$ [s]')
    plt.ylabel(r'$u_x$ [m/s]')
    plt.ylim([-5,20])
    plt.grid(which='major', axis='both')
    plt.tight_layout()
    fig.savefig(path /'velocity_time_series.jpg',dpi=1000)

    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,0],color='k')
    plt.xlabel('$t$ [s]')
    plt.ylabel(r'$u_x$ [m/s]')
    plt.grid(which='major', axis='both')
    plt.xlim([0.5*np.nanmax(time_reconst),0.5*np.nanmax(time_reconst)+0.5])
    plt.tight_layout()
    fig.savefig(path /'velocity_time_series_zoom.jpg',dpi=1000)


    limits_u_x = [
        np.nanmin(
            np.array([
                np.nanmin(u_fluid[:,0])
                ])
            ),
        np.nanmax(
            np.array([
                np.nanmax(u_fluid[:,0])
                ])
            )
        ]

    duration = np.max(t_fluid)
    T_x_real = 0.01
    # Plot velocity time series
    color = [0.2,0.2,0.2]
    lw = 0.2
    f1, axs = plt.subplots(3,figsize=(6,4))
    axs[0].plot(t_fluid,u_fluid[:,0],color=color,lw=lw,label=r'$u_{sbg,f}$')
    axs[1].plot(t_fluid,u_fluid[:,1],color=color,lw=lw)
    axs[2].plot(t_fluid,u_fluid[:,2],color=color,lw=lw)
    axs[0].plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,0], color='r',lw=0,marker='.',label=r'$u_{m}$')
    axs[1].plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,1], color='r',lw=0,marker='.',label=r'$u_{m}$')
    axs[2].plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,2], color='r',lw=0,marker='.',label=r'$u_{m}$')
    axs[0].set_ylabel('$u_x$ [m/s]')
    axs[1].set_ylabel('$u_y$ [m/s]')
    axs[2].set_ylabel('$u_z$ [m/s]')
    axs[2].set_xlabel('$t$ [s]')
    lim = [[limits_u_x[0]-abs(limits_u_x[0])*0.1,limits_u_x[1]+abs(limits_u_x[1])*0.1],
           [-max(1.2*max(abs(u_fluid[:,1])),0.01),
            max(1.2*max(abs(u_fluid[:,1])),0.01)],
           [-max(1.2*max(abs(u_fluid[:,2])),0.01),
            max(1.2*max(abs(u_fluid[:,2])),0.01)]]
    axs[0].set_ylim([lim[0][0],lim[0][1]])
    axs[1].set_ylim([lim[1][0],lim[1][1]])
    axs[2].set_ylim([lim[2][0],lim[2][1]])
    axs[0].set_xlim([duration/2.0,duration/2.0+10*T_x_real])
    axs[1].set_xlim([duration/2.0,duration/2.0+10*T_x_real])
    axs[2].set_xlim([duration/2.0,duration/2.0+10*T_x_real])
    axs[0].grid(which='major', axis='both')
    axs[1].grid(which='major', axis='both')
    axs[2].grid(which='major', axis='both')
    axs[0].legend(loc=1,bbox_to_anchor=(1.2,1.0),frameon=False,fontsize=10)
    plt.subplots_adjust(left=0.1, bottom=0.13, right=0.85, top=0.98)
    plt.tight_layout()
    f1.savefig(path / 'velocity_comparison_10T_u.jpg',dpi=1000)
    f1.savefig(path / 'velocity_comparison_10T_u.pdf',dpi=1000)
    plt.close()

    velocity_fluctuations_fluid = u_fluid - u_mean_fluid
    shear_xz_fluid = velocity_fluctuations_fluid[:, 0] * velocity_fluctuations_fluid[:, 2]
    reynolds_stress_fluid = np.zeros((3, 3))
    for ii in range(3):
        for jj in range(3):
            reynolds_stress_fluid[ii, jj] = np.nanmean(velocity_fluctuations_fluid[:, ii] * velocity_fluctuations_fluid[:, jj])

    u_mean_bubbles = np.nanmean(u_bubbles,axis=0)
    velocity_fluctuations_bubbles = u_bubbles - u_mean_bubbles
    shear_xz_bubbles = velocity_fluctuations_bubbles[:, 0] * velocity_fluctuations_bubbles[:, 2]
    reynolds_stress_bubbles = np.zeros((3, 3))
    for ii in range(3):
        for jj in range(3):
            reynolds_stress_bubbles[ii, jj] = np.nanmean(velocity_fluctuations_bubbles[:, ii] * velocity_fluctuations_bubbles[:, jj])

    u_mean_pierced_bubbles = np.nanmean(u_pierced_bubbles,axis=0)
    velocity_fluctuations_pierced_bubbles = u_pierced_bubbles - u_mean_pierced_bubbles
    shear_xz_pierced_bubbles = velocity_fluctuations_pierced_bubbles[:, 0] * velocity_fluctuations_pierced_bubbles[:, 2]
    reynolds_stress_pierced_bubbles = np.zeros((3, 3))
    for ii in range(3):
        for jj in range(3):
            reynolds_stress_pierced_bubbles[ii, jj] = np.nanmean(velocity_fluctuations_pierced_bubbles[:, ii] * velocity_fluctuations_pierced_bubbles[:, jj])

    u_mean_reconst = np.nanmean(velocity_reconst,axis=0)
    velocity_fluctuations_reconst = velocity_reconst - u_mean_reconst
    shear_xz_reconst = velocity_fluctuations_reconst[:, 0] * velocity_fluctuations_reconst[:, 2]
    reynolds_stress_reconst = np.zeros((3, 3))
    for ii in range(3):
        for jj in range(3):
            reynolds_stress_reconst[ii, jj] = np.nanmean(velocity_fluctuations_reconst[:, ii] * velocity_fluctuations_reconst[:, jj])

    print(reynolds_stress_fluid[0,2])
    print(reynolds_stress_bubbles[0,2])
    print(reynolds_stress_pierced_bubbles[0,2])
    print(reynolds_stress_reconst[0,2])
    duration = np.max(t_fluid)
    T_x_real = 0.01
    # Plot velocity time series
    color = [0.2,0.2,0.2]
    lw = 0.2
    f1, axs = plt.subplots(3,figsize=(6,4))
    axs[0].plot(t_fluid,u_fluid[:,0],color=color,lw=lw,label=r'$u_{sbg,f}$')
    axs[1].plot(t_fluid,u_fluid[:,1],color=color,lw=lw)
    axs[2].plot(t_fluid,u_fluid[:,2],color=color,lw=lw)
    axs[0].plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,0], color='r',lw=0,marker='.',label=r'$u_{m}$')
    axs[1].plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,1], color='r',lw=0,marker='.')
    axs[2].plot(np.nanmean(time_reconst,axis=1),velocity_reconst[:,2], color='r',lw=0,marker='.')
    axs[0].set_ylabel('$u_x$ [m/s]')
    axs[1].set_ylabel('$u_y$ [m/s]')
    axs[2].set_ylabel('$u_z$ [m/s]')
    axs[2].set_xlabel('$t$ [s]')
    lim = [[limits_u_x[0]-abs(limits_u_x[0])*0.1,limits_u_x[1]+abs(limits_u_x[1])*0.1],
           [-max(1.2*max(abs(u_fluid[:,1])),0.01),
            max(1.2*max(abs(u_fluid[:,1])),0.01)],
           [-max(1.2*max(abs(u_fluid[:,2])),0.01),
            max(1.2*max(abs(u_fluid[:,2])),0.01)]]
    axs[0].set_ylim([lim[0][0],lim[0][1]])
    axs[1].set_ylim([lim[1][0],lim[1][1]])
    axs[2].set_ylim([lim[2][0],lim[2][1]])
    axs[0].set_xlim([duration/2.0,duration/2.0+10*T_x_real])
    axs[1].set_xlim([duration/2.0,duration/2.0+10*T_x_real])
    axs[2].set_xlim([duration/2.0,duration/2.0+10*T_x_real])
    axs[0].grid(which='major', axis='both')
    axs[1].grid(which='major', axis='both')
    axs[2].grid(which='major', axis='both')
    axs[0].legend(loc=1,bbox_to_anchor=(1.2,1.0),frameon=False,fontsize=10)
    plt.subplots_adjust(left=0.1, bottom=0.13, right=0.85, top=0.98)
    plt.tight_layout()
    f1.savefig(path / 'velocity_comparison_10T_u.jpg',dpi=1000)
    f1.savefig(path / 'velocity_comparison_10T_u.pdf',dpi=1000)
    plt.close()


    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(t_fluid,shear_xz_fluid,color=color,lw=lw,label=r'$u_{sbg,f}$')
    plt.plot(np.nanmean(time_reconst,axis=1),shear_xz_reconst, color='r',lw=0,marker='.',label=r'$u_{m}$')
    plt.xlabel('$t$ [s]')
    plt.ylabel(r'$uw$ [m2/s2]')
    plt.grid(which='major', axis='both')
    plt.ylim([-1,1])
    plt.xlim([0.5*np.nanmax(time_reconst),0.5*np.nanmax(time_reconst)+0.5])
    plt.tight_layout()
    fig.savefig(path /'shear_uw_zoom.jpg',dpi=1000)

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.sans-serif'] = ['Computer Modern Roman']
    plt.rcParams['font.size'] = 8

    plt.rcParams['mathtext.fontset'] = 'cm'

    plt.rcParams['figure.titlesize'] = 7

    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 2
    plt.rcParams['lines.markeredgewidth'] = 0.7

    plt.rcParams["legend.frameon"] = False
    plt.rcParams['legend.fontsize'] = 6
    plt.rcParams["legend.handletextpad"] = 0.2

    plt.rcParams['axes.labelsize'] = 8
    plt.rcParams['axes.labelpad'] = 0

    plt.rcParams['xtick.major.pad']='2'
    plt.rcParams['xtick.direction']='in'

    plt.rcParams['ytick.major.pad']='2'
    plt.rcParams['ytick.direction']='in'
    plt.rcParams["errorbar.capsize"] = 2
    plt.rcParams["axes.grid"]=False

    u_bubbles_x = u_bubbles[:,0]
    u_bubbles_y = u_bubbles[:,1]
    u_bubbles_z = u_bubbles[:,2]

    u_sbg = u_fluid
    limits_u_x = [np.nanmin(u_sbg[:,0])-0.01*abs(np.nanmin(u_sbg[:,0])),np.nanmax(u_sbg[:,0])+0.01*np.nanmax(u_sbg[:,0])]
    limits_u_y = [np.nanmin(u_sbg[:,1])-0.01*abs(np.nanmin(u_sbg[:,1])),np.nanmax(u_sbg[:,1])+0.01*np.nanmax(u_sbg[:,1])]
    limits_u_z = [np.nanmin(u_sbg[:,2])-0.01*abs(np.nanmin(u_sbg[:,2])),np.nanmax(u_sbg[:,2])+0.01*np.nanmax(u_sbg[:,2])]
    f1, (ax1, ax2) = plt.subplots(1, 2,figsize=(4.5,1.8))
    ax1.scatter(u_sbg[:,0], u_sbg[:,1],facecolors='none', linewidth=0.5,edgecolors=cmap(0/7),alpha=0.5)
    ax1.scatter(u_bubbles[:,0], u_bubbles[:,1],facecolors='none', linewidth=0.5,edgecolors=cmap(2/7),alpha=0.5)
    ax1.scatter(u_pierced_bubbles[:,0], u_pierced_bubbles[:,1],facecolors='none', linewidth=0.5,edgecolors=cmap(4/7),alpha=0.5)
    ax1.scatter(velocity_reconst[:,0], velocity_reconst[:,1],facecolors='none', linewidth=0.5,edgecolors=cmap(6/7))
    ax2.scatter(u_sbg[:,0], u_sbg[:,2],facecolors='none', linewidth=0.5,edgecolors=cmap(0/7),alpha=0.5)
    ax2.scatter(u_bubbles[:,0], u_bubbles[:,2],facecolors='none', linewidth=0.5,edgecolors=cmap(2/7),alpha=0.5)
    ax2.scatter(u_pierced_bubbles[:,0], u_pierced_bubbles[:,2],facecolors='none', linewidth=0.5,edgecolors=cmap(4/7))
    ax2.scatter(velocity_reconst[:,0], velocity_reconst[:,2],facecolors='none', linewidth=0.5,edgecolors=cmap(6/7))
    ax2.scatter([],[],facecolors='none', linewidth=0.5,edgecolors=cmap(0/7),alpha=1.0,label=r'$u_{sbg,f}$'+'\n($n$='+f'{len(u_sbg[:,0])})')
    ax2.scatter([],[],facecolors='none', linewidth=0.5,edgecolors=cmap(2/7),alpha=1.0,label=r'$u_{sbg,b,all}$'+'\n($n$='+f'{len(u_bubbles[:,0][~np.isnan(u_bubbles[:,0])])})')
    ax2.scatter([],[],facecolors='none', linewidth=0.5,edgecolors=cmap(4/7),alpha=1.0,label=r'$u_{sbg,b,pierced}$'+'\n($n$='+f'{len(u_pierced_bubbles[:,0][~np.isnan(u_pierced_bubbles[:,0])])})')
    ax2.scatter([],[],facecolors='none', linewidth=0.5,edgecolors=cmap(6/7),alpha=1.0,label=r'$u_{meas}$'+'\n($n$='+f'{len(velocity_reconst[:,0][~np.isnan(velocity_reconst[:,0])])})')
    ax1.set_xlim(limits_u_x)
    ax2.set_xlim(limits_u_x)
    ax1.set_ylim(limits_u_y)
    ax2.set_ylim(limits_u_z)
    ax1.set_xlabel('$u_x$ [m/s]')
    ax2.set_xlabel('$u_x$ [m/s]')
    ax1.set_ylabel('$u_y$ [m/s]')
    ax2.set_ylabel('$u_z$ [m/s]')
    ax1.grid(which='major', axis='both')
    ax2.grid(which='major', axis='both')
    ax2.legend(loc=1,bbox_to_anchor=(1.72,1.0),frameon=False,fontsize=8)
    plt.subplots_adjust(left=0.09, bottom=0.15, right=0.80, top=0.98,wspace=0.4)
    # f1.savefig(path / 'velocity_scatter.svg',dpi=300)
    # f1.savefig(path / 'velocity_scatter.pdf',dpi=300)
    f1.savefig(path / 'velocity_scatter.png',dpi=600)
    plt.close()

if __name__ == "__main__":
    main()