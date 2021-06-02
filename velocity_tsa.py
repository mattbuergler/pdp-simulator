#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:48:27 2021

@author: matthias
"""
import argparse
import json
import numpy as np
import pandas as pd
import time
import math
import pathlib
from statsmodels.tsa.stattools import acf, pacf
from scipy import signal
from matplotlib import pyplot as plt
try:
    from dataio.H5Writer import H5Writer
    from dataio.H5Reader import H5Reader
    from tools.globals import *
except ImportError:
    print("Error while importing modules")
    raise

"""
    Velocity timeseries analysis

"""

def main(args):
    """
        Main function of the velocity timeseries analysis

    """

    # Create parser to read in the configuration JSON-file to read from
    # the command line interface (CLI)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=str,
        help="The path to the scenario directory.")
    args = parser.parse_args(args)

    # Create Posix path for OS indepency
    path = pathlib.Path(args.path)
    # Create a H5-file reader
    reader = H5Reader(path / 'flow_data.h5')
    # Read the fluid velocity
    u_f = reader.getDataSet('fluid/velocity')[:]
    # Read the time vector
    t_f = reader.getDataSet('fluid/time')[:]
    # Read the fluid velocity
    u_p = reader.getDataSet('particle/velocity')[:]
    # Read the time vector
    t_p = reader.getDataSet('particle/time')[:]
    # Read the configuration data
    flow_props = json.loads(reader.getDataSet('.flow_properties')[0][0].decode( \
            "ascii").replace("'",'"'))
    reader.close()

    directions = ['x','y','z']
    total_duration = t_f[-1]-t_f[0]
    ts_frequency = int((len(t_f)-1)/total_duration)

    # Accelrations for various sampling frequencies
    fs1=7
    dUdt_vel = np.diff(u_f,axis=0)*ts_frequency
    log10_first = 1
    resampling_frequencies = np.flip(np.logspace( \
                                log10_first, \
                                math.log10(ts_frequency), \
                                int(math.log10(ts_frequency))))
    fig, axs = plt.subplots(len(resampling_frequencies),len(directions), \
            figsize=(len(directions)*2.5,len(resampling_frequencies)*2.5))
    for ii,res_frq in enumerate(resampling_frequencies):
        n_timesteps = res_frq*total_duration
        res_indices = np.linspace(0,len(t_f)-1,int(n_timesteps)+1).astype('int')
        resampled_vel = u_f[res_indices]
        dU = np.diff(resampled_vel,axis=0)
        dUdt = dU*res_frq
        for jj,di in enumerate(directions):
            (n, bins, patches) = axs[ii,jj].hist(dUdt[:,jj],bins=100,density=True, \
                range=(min(dUdt_vel.reshape(-1)),max(dUdt_vel.reshape(-1))))
            axs[ii,jj].set_title(r'$\tau$ = {0} s'.format(1/res_frq))
            axs[ii,jj].text(0.99*min(dUdt_vel.reshape(-1)),0.95*max(n), \
                r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt[:,jj])), \
                fontsize=fs1)
            axs[ii,jj].text(0.99*min(dUdt_vel.reshape(-1)),0.85*max(n), \
                r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt[:,jj])), \
                fontsize=fs1)
            axs[ii,jj].set_xlim([-max(abs(bins)),max(abs(bins))])
            if jj == 0:
                axs[ii,jj].set_ylabel('Frequency [-]')
            if ii == len(resampling_frequencies)-1:
                if jj == 0:
                    axs[ii,jj].set_xlabel(r'Acceleration $dU_x/dt$ [m/s$^{-2}$]')
                elif jj == 1:
                    axs[ii,jj].set_xlabel(r'Acceleration $dU_y/dt$ [m/s$^{-2}$]')
                else:
                    axs[ii,jj].set_xlabel(r'Acceleration $dU_z/dt$ [m/s$^{-2}$]')
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_xyz.svg',dpi=300)

    # Accelrations for various sampling frequencies
    fs1=7
    log10_first = 1
    resampling_frequencies = np.flip(np.logspace( \
            log10_first,math.log10(ts_frequency), \
            int(math.log10(ts_frequency))))
    fig, axs = plt.subplots(ncols=len(resampling_frequencies),\
            figsize=(len(resampling_frequencies)*2.5,2.5))
    for jj,res_frq in enumerate(resampling_frequencies):
        n_timesteps = res_frq*total_duration
        res_indices = np.linspace(0,len(t_f)-1,int(n_timesteps)+1).astype('int')
        resampled_vel = u_f[res_indices]
        dU = np.diff(resampled_vel,axis=0)
        dUdt = dU*res_frq
        axs[jj]
        (n, bins, patches) = axs[jj].hist(dUdt[:,0],bins=100 ,density=True, \
            range=(min(dUdt_vel.reshape(-1)),max(dUdt_vel.reshape(-1))))
        axs[jj].set_title(r'$\tau$ = {0} s'.format(1/res_frq))
        axs[jj].text(0.99*min(dUdt_vel.reshape(-1)),0.95*max(n), \
            r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt[:,0])),fontsize=fs1)
        axs[jj].text(0.99*min(dUdt_vel.reshape(-1)),0.85*max(n), \
            r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt[:,0])),fontsize=fs1)
        if jj == 0:
            axs[jj].set_ylabel('Frequency [-]')
        axs[jj].set_xlabel(r'Acceleration $dU_x/dt$ [m/s$^{-2}$]')
        axs[jj].set_xlim([-max(abs(bins)),max(abs(bins))])
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_x.svg',dpi=300)

    # Accelrations for various sampling frequencies
    fs1=7
    fig = plt.figure(figsize=(3.0,2.5))
    dU = np.diff(u_f,axis=0)
    dUdt = dU*ts_frequency
    (n, bins, patches) = plt.hist(dUdt[:,0],bins=100 ,density=True, \
            range=(min(dUdt_vel.reshape(-1)),max(dUdt_vel.reshape(-1))))
    plt.title(r'$\delta t$ = {0} s'.format(1/ts_frequency))
    plt.text(0.99*min(dUdt_vel.reshape(-1)),0.95*max(n), \
        r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt[:,0])),fontsize=fs1)
    plt.text(0.99*min(dUdt_vel.reshape(-1)),0.85*max(n), \
        r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt[:,0])),fontsize=fs1)
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Acceleration $dU_x/dt$ [m/s$^{-2}$]')
    plt.xlim([-max(abs(bins)),max(abs(bins))])
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_x_single.svg',dpi=300)

    fs2=10
    T_L = flow_props['integral_timescale']
    T_I = flow_props['turbulent_intensity']
    U_m = flow_props['mean_velocity']
    # Plot velocity time series
    color = [0.2,0.2,0.2]
    lw = 0.75
    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(t_f,u_f[:,0],color='k',lw=lw,label='fluid')
    plt.plot(t_p,u_p[:,0],color='r',lw=lw,label='particle')
    plt.ylabel('$u_x$ [m/s]')
    plt.xlabel('time $t$ [s]')
    plt.xlim([0,flow_props['duration']])
    plt.ylim([U_m[0]-5*U_m[0]*T_I[0],U_m[0]+5*U_m[0]*T_I[0]])
    plt.title(r'$T_{{I,x}}$ = {}, $T_{{L,x}}$ = {}s, $dt$={}s, $T_{{L,x}}/dt$={}'.format( \
                        T_I[0], \
                        T_L[0], \
                        1.0/ts_frequency,int(T_L[0]*ts_frequency)), \
                fontsize=fs2)
    plt.grid(which='major', axis='both')
    plt.legend(loc=2, ncol=2)
    plt.tight_layout()
    fig.savefig(path / 'velocity_ts.svg',dpi=300)

    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(t_f/T_L[0],(u_f[:,0]-U_m[0])/(U_m[0]*T_I[0]),color='k', lw=0.75, \
        label='fluid')
    plt.plot(t_p/T_L[0],(u_p[:,0]-U_m[0])/(U_m[0]*T_I[0]),color='r', lw=0.75, \
        label='particle')
    plt.xlim([0,10])
    plt.ylim([-4,4])
    plt.xlabel('$t/T_L$ [-]')
    plt.ylabel(r'$(u*-\bar{u*})/ \sigma$ [-]')
    plt.legend(loc=2, ncol=2)
    plt.grid(which='major', axis='both')
    plt.title(r'$T_{{I,x}}$ = {}, $T_{{L,x}}$ = {}s, $dt$={}s, $T_{{L,x}}/dt$={}'.format(
                        T_I[0], \
                        T_L[0], \
                        1.0/ts_frequency,int(T_L[0]*ts_frequency)), \
                fontsize=fs2)
    plt.tight_layout()
    fig.savefig(path /'normalized_velocity_ts.svg',dpi=300)

    lag = int(5*T_L[0]*ts_frequency)
    acf,ci = acf(u_f[:,0],nlags=lag,alpha=0.05,fft=False)
    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(np.linspace(0,lag,lag+1)/ts_frequency/T_L[0], \
        acf,color='k',linestyle='-',label='simulated')
    # plt.plot(np.linspace(0,lag,lag+1),ci,color='k',linestyle=':')
    plt.plot(np.linspace(0,lag,lag+1)/ts_frequency/T_L[0], \
        np.exp(-np.linspace(0,lag,lag+1)/ts_frequency/T_L[0]), \
        color='k',linestyle=':',label='theoritical')
    plt.xlim([0,lag/ts_frequency/T_L[0]])
    plt.ylim([0,1])
    plt.xlabel(r'$s/T_L$ [-]')
    plt.ylabel(r'$\rho(s)$ [-]')
    plt.legend(loc=1)
    plt.grid(which='major', axis='both')
    plt.tight_layout()
    fig.savefig(path / 'acf.svg',dpi=300)

    freqs_f, psd_f = signal.welch(u_f[:,0], fs=ts_frequency)
    freqs_p, psd_p = signal.welch(u_p[:,0], fs=ts_frequency)
    fig = plt.figure(figsize=(4.5,2.5))
    plt.loglog(freqs_f, psd_f, color='k', lw=0.75, label='fluid' )
    plt.loglog(freqs_p, psd_p, color='r', lw=0.75, label='particle' )
    plt.title('PSD: power spectral density')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.legend(loc=1, ncol=2)
    plt.grid(which='major', axis='both')
    plt.tight_layout()
    fig.savefig(path / 'psd.svg',dpi=300)


if __name__ == '__main__':
    main(sys.argv[1:])