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
    if type(args) == str:
        path = pathlib.Path(args)
    else:
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
    # Read the cont. velocity
    u_f = reader.getDataSet('fluid/velocity')[:]
    # Read the time vector
    t_f = reader.getDataSet('fluid/time')[:]
    # Read the cont. velocity
    u_p = reader.getDataSet('bubbles/velocity')[:]
    # Read the time vector
    t_p = reader.getDataSet('bubbles/time')[:]
    # Read the time vector
    Re = reader.getDataSet('bubbles/Re_bubble')[:]
    # Read the time vector
    C_d = reader.getDataSet('bubbles/C_d')[:]

    # Read the configuration data
    flow_props = json.loads(reader.getDataSet('.flow_properties')[0][0].decode( \
            "ascii").replace("'",'"'))
    reader.close()

    directions = ['x','y','z']
    total_duration = t_f[-1]-t_f[0]
    ts_frequency = int((len(t_f)-1)/total_duration)

    for ii,k in enumerate(['x','y','z']):
        fs2=10
        T_L = flow_props['integral_timescale']
        T_I = flow_props['turbulent_intensity']
        U_m = flow_props['mean_velocity']
        # Plot velocity time series
        color = [0.2,0.2,0.2]
        lw = 0.75
        fig = plt.figure(figsize=(4.5,2.5))
        plt.plot(t_p,u_p[:,ii],color='r',lw=lw,ls='--',label='disp.')
        plt.plot(t_f,u_f[:,ii],color='k',lw=lw,label='cont.')
        plt.ylabel(f'$u_{k}$ [m/s]')
        plt.xlabel('time $t$ [s]')
        plt.xlim([0,flow_props['duration']])
        plt.ylim([min(np.nanmin(u_f[:,ii]),np.nanmin(u_p[:,ii])),max(np.nanmax(u_f[:,ii]),np.nanmax(u_p[:,ii]))])
        plt.title(r'$T_{{I,{}}}$ = {:.3f}, $T_{{L,{}}}$ = {:.4f}s'.format( \
                            k, \
                            T_I[ii], \
                            k, \
                            T_L[ii], \
                            1.0/ts_frequency,
                            k, \
                            int(T_L[ii]*ts_frequency)), \
                    fontsize=fs2)
        plt.grid(which='major', axis='both')
        plt.legend(loc=2, ncol=2)
        plt.tight_layout()
        fig.savefig(path / f'velocity_{k}_ts.svg',dpi=300)

    data = pd.DataFrame(Re,columns=['Re'])
    data['C_d'] = C_d
    data = data.sort_values(by=['Re'])
    data = data[data['Re'] > 0]
    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(data['Re'],data['C_d'])
    plt.xlabel(f'$Re_b$ [-]')
    plt.ylabel('$C_d$ [-]')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim([0.1,10])
    plt.grid(which='major', axis='both')
    plt.tight_layout()
    fig.savefig(path / f'Cd_vs_Re.svg',dpi=300)
    sys.exit()

    # Accelrations for various sampling frequencies
    fs1=7
    dUdt_vel_f = np.diff(u_f,axis=0)*ts_frequency
    dUdt_vel_p = np.diff(u_p,axis=0)*ts_frequency
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
        resampled_f = u_f[res_indices]
        resampled_p = u_p[res_indices]
        dU_f= np.diff(resampled_f,axis=0)
        dU_p= np.diff(resampled_p,axis=0)
        dUdt_f = dU_f*res_frq
        dUdt_p = dU_p*res_frq
        for jj,di in enumerate(directions):
            (n_f, bins_f, patches_f) = axs[ii,jj].hist(dUdt_f[:,jj],bins=100,density=True, alpha=0.7, label='cont.',\
                range=(min(dUdt_vel_f.reshape(-1)),max(dUdt_vel_f.reshape(-1))))
            (n_p, bins_p, patches_p) = axs[ii,jj].hist(dUdt_p[:,jj],bins=100,density=True, alpha=0.7, label='disp.', color='r', \
                range=(min(dUdt_vel_f.reshape(-1)),max(dUdt_vel_f.reshape(-1))))
            axs[ii,jj].set_title(r'$\tau$ = {0} s'.format(1/res_frq))
            axs[ii,jj].text(0.99*min(dUdt_vel_f.reshape(-1)),0.95*max(n_p), \
                r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt_f[:,jj])), \
                fontsize=fs1)
            axs[ii,jj].text(0.99*min(dUdt_vel_f.reshape(-1)),0.85*max(n_p), \
                r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt_f[:,jj])), \
                fontsize=fs1)
            axs[ii,jj].set_xlim([-max(abs(bins_f)),max(abs(bins_f))])
            if jj == 0:
                axs[ii,jj].set_ylabel('Frequency [-]')
            if ii == len(resampling_frequencies)-1:
                if jj == 0:
                    axs[ii,jj].set_xlabel(r'Acceleration $du_x/dt$ [m/s$^{-2}$]')
                elif jj == 1:
                    axs[ii,jj].set_xlabel(r'Acceleration $du_y/dt$ [m/s$^{-2}$]')
                else:
                    axs[ii,jj].set_xlabel(r'Acceleration $du_z/dt$ [m/s$^{-2}$]')
            if (ii==len(resampling_frequencies)-1) & (jj==len(directions)-1):
                axs[ii,jj].legend(loc=1,fontsize=7)
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_xyz_disp_cont.svg',dpi=300)

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
        resampled_f = u_f[res_indices]
        resampled_p = u_p[res_indices]
        dU_f= np.diff(resampled_f,axis=0)
        dU_p= np.diff(resampled_p,axis=0)
        dUdt_f = dU_f*res_frq
        dUdt_p = dU_p*res_frq
        (n_f, bins_f, patches_f) = axs[jj].hist(dUdt_f[:,0], \
            bins=100, density=True, alpha=0.7, label='cont.',\
            range=(min(dUdt_vel_f.reshape(-1)),max(dUdt_vel_f.reshape(-1))))
        (n_p, bins_p, patches_p) = axs[jj].hist(dUdt_p[:,0], \
            bins=100, density=True, alpha=0.7, color='r', label='disp.',\
            range=(min(dUdt_vel_f.reshape(-1)),max(dUdt_vel_f.reshape(-1))))
        axs[jj].set_title(r'$\tau$ = {0} s'.format(1/res_frq))
        axs[jj].text(0.99*min(dUdt_vel_f.reshape(-1)),0.95*max(n_p), \
            r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt_f[:,0])),fontsize=fs1)
        axs[jj].text(0.99*min(dUdt_vel_f.reshape(-1)),0.85*max(n_p), \
            r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt_f[:,0])),fontsize=fs1)
        if jj == 0:
            axs[jj].set_ylabel('Frequency [-]')
        axs[jj].set_xlabel(r'Acceleration $du_x/dt$ [m/s$^{-2}$]')
        axs[jj].set_xlim([-max(abs(bins_f)),max(abs(bins_f))])
        if (ii==len(resampling_frequencies)-1):
            axs[jj].legend(loc=1,fontsize=7)
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_x_disp_cont.svg',dpi=300)

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
        resampled_p = u_p[res_indices]
        dU_p= np.diff(resampled_p,axis=0)
        dUdt_p = dU_p*res_frq
        (n_p, bins_p, patches_p) = axs[jj].hist(dUdt_p[:,0], \
            bins=100, density=True, alpha=0.7, color='r', label='disp.',\
            range=(min(dUdt_p.reshape(-1)),max(dUdt_p.reshape(-1))))
        axs[jj].set_title(r'$\tau$ = {0} s'.format(1/res_frq))
        axs[jj].text(0.99*min(dUdt_p.reshape(-1)),0.95*max(n_p), \
            r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt_f[:,0])),fontsize=fs1)
        axs[jj].text(0.99*min(dUdt_p.reshape(-1)),0.85*max(n_p), \
            r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt_f[:,0])),fontsize=fs1)
        if jj == 0:
            axs[jj].set_ylabel('Frequency [-]')
        axs[jj].set_xlabel(r'Acceleration $du_x/dt$ [m/s$^{-2}$]')
        axs[jj].set_xlim([-max(abs(bins_p)),max(abs(bins_p))])
        if (ii==len(resampling_frequencies)-1):
            axs[jj].legend(loc=1,fontsize=7)
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_x_disp.svg',dpi=300)

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
        resampled_f = u_f[res_indices]
        dU_f= np.diff(resampled_f,axis=0)
        dUdt_f = dU_f*res_frq
        (n_f, bins_f, patches_f) = axs[jj].hist(dUdt_f[:,0], \
            bins=100, density=True, alpha=0.7, label='cont.',\
            range=(min(dUdt_vel_f.reshape(-1)),max(dUdt_vel_f.reshape(-1))))
        axs[jj].set_title(r'$\tau$ = {0} s'.format(1/res_frq))
        axs[jj].text(0.99*min(dUdt_vel_f.reshape(-1)),0.95*max(n_f), \
            r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt_f[:,0])),fontsize=fs1)
        axs[jj].text(0.99*min(dUdt_vel_f.reshape(-1)),0.85*max(n_f), \
            r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt_f[:,0])),fontsize=fs1)
        if jj == 0:
            axs[jj].set_ylabel('Frequency [-]')
        axs[jj].set_xlabel(r'Acceleration $du_x/dt$ [m/s$^{-2}$]')
        axs[jj].set_xlim([-max(abs(bins_f)),max(abs(bins_f))])
        if (ii==len(resampling_frequencies)-1):
            axs[jj].legend(loc=1,fontsize=7)
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_x_cont.svg',dpi=300)

    # Accelrations for various one frequencies
    fs1=7
    fig = plt.figure(figsize=(3.0,2.5))
    dU_f= np.diff(u_f,axis=0)
    dU_p= np.diff(u_p,axis=0)
    dUdt_f = dU_f*ts_frequency
    dUdt_p = dU_p*ts_frequency
    (n_f, bins_f, patches_f) = plt.hist(dUdt_f[:,0], \
            bins=100, density=True, alpha=0.7, label='cont.',\
            range=(min(dUdt_vel_f.reshape(-1)),max(dUdt_vel_f.reshape(-1))))
    (n_p, bins_p, patches_p) = plt.hist(dUdt_p[:,0], \
            bins=100, density=True, alpha=0.7, color='r', label='disp.',\
            range=(min(dUdt_vel_f.reshape(-1)),max(dUdt_vel_f.reshape(-1))))
    plt.title(r'$dt$ = {0} s'.format(1/ts_frequency))
    plt.text(0.99*min(dUdt_vel_f.reshape(-1)),0.95*max(n_p), \
        r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt_f[:,0])),fontsize=fs1)
    plt.text(0.99*min(dUdt_vel_f.reshape(-1)),0.85*max(n_p), \
        r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt_f[:,0])),fontsize=fs1)
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Acceleration $du_x/dt$ [m/s$^{-2}$]')
    plt.xlim([-max(abs(bins_f)),max(abs(bins_f))])
    plt.legend(loc=1,fontsize=7)
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_x_single_disp_cont.svg',dpi=300)

    # Accelrations for one sampling frequencies
    fs1=7
    fig = plt.figure(figsize=(3.0,2.5))
    dU_p= np.diff(u_p,axis=0)
    dUdt_p = dU_p*ts_frequency
    (n_p, bins_p, patches_p) = plt.hist(dUdt_p[:,0], \
            bins=100,density=True, color='r', alpha=0.7, label='disp.',\
            range=(min(dUdt_p.reshape(-1)),max(dUdt_p.reshape(-1))))
    plt.title(r'$dt$ = {0} s'.format(1/ts_frequency))
    plt.text(0.99*min(dUdt_p.reshape(-1)),0.95*max(n_p), \
        r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt_p[:,0])),fontsize=fs1)
    plt.text(0.99*min(dUdt_p.reshape(-1)),0.85*max(n_p), \
        r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt_p[:,0])),fontsize=fs1)
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Acceleration $dU_x/dt$ [m/s$^{-2}$]')
    plt.xlim([-max(abs(bins_p)),max(abs(bins_p))])
    plt.legend(loc=1,fontsize=7)
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_x_single_disp.svg',dpi=300)

    # Accelrations for one sampling frequencies
    fs1=7
    fig = plt.figure(figsize=(3.0,2.5))
    dU_f= np.diff(u_f,axis=0)
    dUdt_f = dU_f*ts_frequency
    (n_f, bins_f, patches_f) = plt.hist(dUdt_f[:,0], \
            bins=100,density=True, alpha=0.7, label='cont.',\
            range=(min(dUdt_f.reshape(-1)),max(dUdt_f.reshape(-1))))
    plt.title(r'$dt$ = {0} s'.format(1/ts_frequency))
    plt.text(0.99*min(dUdt_f.reshape(-1)),0.95*max(n_f), \
        r'$\mu$ = {:.1f}m/s$^{{-2}}$'.format(np.mean(dUdt_f[:,0])),fontsize=fs1)
    plt.text(0.99*min(dUdt_f.reshape(-1)),0.85*max(n_f), \
        r'$\sigma$ = {:.0f}m/s$^{{-2}}$'.format(np.std(dUdt_f[:,0])),fontsize=fs1)
    plt.ylabel('Frequency [-]')
    plt.xlabel(r'Acceleration $dU_x/dt$ [m/s$^{-2}$]')
    plt.xlim([-max(abs(bins_f)),max(abs(bins_f))])
    plt.legend(loc=1,fontsize=7)
    plt.tight_layout()
    fig.savefig(path / 'acceleration_histrograms_x_single_cont.svg',dpi=300)

    for ii,k in enumerate(['x','y','z']):
        fs2=10
        T_L = flow_props['integral_timescale']
        T_I = flow_props['turbulent_intensity']
        U_m = flow_props['mean_velocity']
        # Plot velocity time series
        color = [0.2,0.2,0.2]
        lw = 0.75
        fig = plt.figure(figsize=(4.5,2.5))
        plt.plot(t_f,u_f[:,ii],color='k',lw=lw,label='cont.')
        plt.plot(t_p,u_p[:,ii],color='r',lw=lw,label='disp.')
        plt.ylabel(f'$u_{k}$ [m/s]')
        plt.xlabel('time $t$ [s]')
        plt.xlim([0,flow_props['duration']])
        plt.ylim([U_m[ii]-5*U_m[0]*T_I[ii],U_m[ii]+5*U_m[0]*T_I[ii]])
        plt.title(r'$T_{{I,{}}}$ = {}, $T_{{L,{}}}$ = {}s, $dt$={}s, $T_{{L,{}}}/dt$={}'.format( \
                            k, \
                            T_I[ii], \
                            k, \
                            T_L[ii], \
                            1.0/ts_frequency,
                            k, \
                            int(T_L[ii]*ts_frequency)), \
                    fontsize=fs2)
        plt.grid(which='major', axis='both')
        plt.legend(loc=2, ncol=2)
        plt.tight_layout()
        fig.savefig(path / f'velocity_{k}_ts.svg',dpi=300)


    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(t_f/T_L[0],(u_f[:,0]-U_m[0])/(U_m[0]*T_I[0]),color='k', lw=0.75, \
        label='cont.')
    plt.plot(t_p/T_L[0],(u_p[:,0]-U_m[0])/(U_m[0]*T_I[0]),color='r', lw=0.75, \
        label='disp.')
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
    acf_f,ci = acf(u_f[:,0],nlags=lag,alpha=0.05,fft=False)
    acf_p,ci = acf(u_p[:,0],nlags=lag,alpha=0.05,fft=False)
    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(np.linspace(0,lag,lag+1)/ts_frequency/T_L[0], \
        acf_f,color='k',linestyle='-',label='cont.')
    plt.plot(np.linspace(0,lag,lag+1)/ts_frequency/T_L[0], \
        acf_p,color='r',linestyle='--',label='disp.')
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

    lag = int(100*T_L[0]*ts_frequency)
    acf_f,ci = acf(u_f[:,0],nlags=lag,alpha=0.05,fft=False)
    acf_p,ci = acf(u_p[:,0],nlags=lag,alpha=0.05,fft=False)
    fig = plt.figure(figsize=(4.5,2.5))
    plt.plot(np.linspace(0,lag,lag+1)/ts_frequency/T_L[0], \
        acf_f,color='k',linestyle='-',label='cont.')
    plt.plot(np.linspace(0,lag,lag+1)/ts_frequency/T_L[0], \
        acf_p,color='r',linestyle='--',label='disp.')
    # plt.plot(np.linspace(0,lag,lag+1),ci,color='k',linestyle=':')
    plt.plot(np.linspace(0,lag,lag+1)/ts_frequency/T_L[0], \
        np.exp(-np.linspace(0,lag,lag+1)/ts_frequency/T_L[0]), \
        color='k',linestyle=':',label='theoritical')
    plt.xlim([0.1,lag/ts_frequency/T_L[0]])
    plt.ylim([0,1])
    plt.xlabel(r'$s/T_L$ [-]')
    plt.ylabel(r'$\rho(s)$ [-]')
    plt.legend(loc=1)
    plt.grid(which='major', axis='both')
    plt.xscale('log')
    plt.tight_layout()
    fig.savefig(path / 'acf_logx.svg',dpi=300)

    freqs_f, psd_f = signal.welch(u_f[:,0], fs=ts_frequency)
    freqs_p, psd_p = signal.welch(u_p[:,0], fs=ts_frequency)
    fig = plt.figure(figsize=(4.5,2.5))
    plt.loglog(freqs_f, psd_f, color='k', lw=0.75, label='cont.' )
    plt.loglog(freqs_p, psd_p, color='r', lw=0.75, label='disp.' )
    plt.title('PSD: power spectral density')
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.legend(loc=1, ncol=2)
    plt.grid(which='major', axis='both')
    plt.tight_layout()
    fig.savefig(path / 'psd.svg',dpi=300)


if __name__ == '__main__':
    main(sys.argv[1:])