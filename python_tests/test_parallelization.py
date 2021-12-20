import numpy as np
import pandas as pd
import bisect
import random
import time
import math
from contextlib import closing
import multiprocessing as mp
import os
from joblib import Parallel, delayed
import sys

def _init(shared_arr_):
    # The shared array pointer is a global variable so that it can be accessed by the
    # child processes. It is a tuple (pointer, dtype, shape).
    global shared_arr
    shared_arr = shared_arr_


def shared_to_numpy(shared_arr, dtype, shape):
    """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
    No copy is involved, the array reflects the underlying shared buffer."""
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


def create_shared_array(dtype, shape):
    """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
    Note that the buffer values are not initialized.
    """
    dtype = np.dtype(dtype)
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    # Create the RawArray instance.
    shared_arr = mp.RawArray(cdtype, sum(shape))
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr

signal = np.zeros((10*10+1,4))

dt = 0.01
#Version Parallel 2
def process_parallel(kk):
    if (signal[(kk-1):(kk+2),:] == 0).all():
        # delete this row
        time.sleep(dt)
        return True
    else:
        return False

t1 = time.time()
results = Parallel(n_jobs=2)(delayed(process_parallel)(kk) for kk in range(1,len(signal)-1))
t2 = time.time()

print(f'time = {t2-t1}s')
results.insert(0,False)
results.append(False)
results = np.asarray(results)


# Version Sequentiel
# n= 1 time = 1.4384872913360596s

data_type = np.int8
shape = (len(signal),)
del_idx = np.zeros(shape,dtype=np.int8)

def process_seq(kk):
    if (signal[(kk-1):(kk+2),:] == 0).all():
        # delete this row
        time.sleep(dt)
        del_idx[kk] = True

t1 = time.time()
for kk in range(1,len(signal)-1):
        process_seq(kk)
t2 = time.time()
print(f'time = {t2-t1}s')

    # Initialize signals to zero
    signal = np.zeros((n_probe, n_sensors)).astype('uint8')
    # Loop over all bubbles
    print('\nSampling the sensor signal')
    t1 = time.time()
    # sequential
    print(f'signal = {signal}')
    print(f'sum signal = {np.sum(signal)}')
    for kk in range(0,nb):
        get_signal(kk, t_traj, X, X_rand, AT, b_size, um, max_probe_size, t_probe, c_probe, sensor_delta, signal, progress, nb)
    print(f'sum signal = {np.sum(signal)}')
    t2 = time.time()
    print(f'time = {t2-t1}s')

def get_signal(kk, t_traj, X, X_rand, AT, b_size, um, max_probe_size, t_probe, c_probe, sensor_delta, signal, progress, nb):
    # Interpolate the location of the bubble at t = AT
    X_b_AT = SBG_interp_trajectory(t_traj, X, AT[kk])
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
        # Get probe locations that lie within the estimated timeframe
        c_probe_kk = c_probe[(t_probe >= t_min) & (t_probe <= t_max),:]
        # Set x-coordinate of probe to x-coordinate ob bubble center at AT
        c_probe_kk[:,0] = X_b_AT[0]
        # Set y-coordinate of probe to y-coordinate ob bubble center + random
        # shift with ~Uniform[-B/2,B/2]
        c_probe_kk[:,1] = c_probe_kk[:,1] + X_b_AT[1] + X_rand[kk,0]
        # Set z-coordinate of probe to z-coordinate ob bubble center + random
        # shift with ~Uniform[-B/2,B/2]
        c_probe_kk[:,2] = c_probe_kk[:,2] + X_b_AT[2] + X_rand[kk,1]
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
                    (((c_probe_kk[:,0]+delta[0])-X_resampled[:,0])/abc[0])**2 \
                  + (((c_probe_kk[:,1]+delta[1])-X_resampled[:,1])/abc[1])**2 \
                  + (((c_probe_kk[:,2]+delta[2])-X_resampled[:,2])/abc[2])**2
            # Check for which time steps the bubble is pierced
            idxs = np.where(radius <= 1) + row
            # pierced, set signal to 1
            signal[idxs,idx] = 1;
        # Display progress
        if progress:
            printProgressBar(kk + 1, nb, prefix = 'Progress:', suffix = 'Complete', length = 50)

# def get_signal_parallel(kk, t_traj, X, X_rand, AT, b_size, um, max_probe_size, t_probe, c_probe, sensor_delta, signal, progress, nb):
#     # Interpolate the location of the bubble at t = AT
#     X_b_AT = SBG_interp_trajectory(t_traj, X, AT[kk])
#     # Estimate timeframe for tracking the movement of bubble kk
#     critical_time_pre = 3.0*np.linalg.norm(b_size[kk,:])/np.linalg.norm(um)
#     critical_time_post = 3.0*(np.linalg.norm(b_size[kk,:]) + \
#                          max_probe_size)/np.linalg.norm(um)
#     t_min = AT[kk] - critical_time_pre
#     t_max = AT[kk] + critical_time_post
#     # Get probe sampling times that lie within the estimated timeframe
#     t_probe_kk = t_probe[(t_probe >= t_min) & (t_probe <= t_max)]
#     # Check if number of samples lies inside the timeframe is larger than 0
#     if len(t_probe_kk) > 0:
#         # Get the range of the trajectory that encloses this timeframe
#         id_t_min_traj = max(bisect.bisect_left(t_traj, min(t_probe_kk))-1,
#             0)
#         id_t_max_traj = min(bisect.bisect_right(t_traj, max(t_probe_kk))+1,len(t_traj))
#         t_traj_kk = t_traj[id_t_min_traj:id_t_max_traj]
#         X_traj_kk = X[id_t_min_traj:id_t_max_traj]
#         # Resample the trajectory to the sampling times of the probe
#         t_resampled, X_resampled = SBG_get_Signal_traj(t_traj_kk, \
#                             X_traj_kk, t_probe_kk)
#         # Get probe locations that lie within the estimated timeframe
#         c_probe_kk = c_probe[(t_probe >= t_min) & (t_probe <= t_max),:]
#         # Set x-coordinate of probe to x-coordinate ob bubble center at AT
#         c_probe_kk[:,0] = X_b_AT[0]
#         # Set y-coordinate of probe to y-coordinate ob bubble center + random
#         # shift with ~Uniform[-B/2,B/2]
#         c_probe_kk[:,1] = c_probe_kk[:,1] + X_b_AT[1] + X_rand[kk,0]
#         # Set z-coordinate of probe to z-coordinate ob bubble center + random
#         # shift with ~Uniform[-B/2,B/2]
#         c_probe_kk[:,2] = c_probe_kk[:,2] + X_b_AT[2] + X_rand[kk,1]
#         # Loop over all time steps within the timeframe
#         abc = b_size[kk,:] / 2.0
#         # Determine the row in the signal time series where to write the signal
#         row = np.where(t_probe == min(t_probe_kk))[0][0]
#         # Loop over each sensor and check if it is inside the bubble
#         for idx,delta in sensor_delta.items():
#             # Check if ellipsoid is pierced by sensor idx
#             # Standard euqation: (x/a)2 + (y/b)2 + (z/c)2 = 1
#             # with x = (cx+delta - x_bubble)
#             radius = \
#                     (((c_probe_kk[:,0]+delta[0])-X_resampled[:,0])/abc[0])**2 \
#                   + (((c_probe_kk[:,1]+delta[1])-X_resampled[:,1])/abc[1])**2 \
#                   + (((c_probe_kk[:,2]+delta[2])-X_resampled[:,2])/abc[2])**2
#             # Check for which time steps the bubble is pierced
#             idxs = np.where(radius <= 1) + row
#             # pierced, set signal to 1
#             signal[idxs,idx] = 1;
#         # Display progress
#         if progress:
#             printProgressBar(kk + 1, nb, prefix = 'Progress:', suffix = 'Complete', length = 50)

def get_signal_parallel_new(kk, t_traj, X, X_rand, AT, b_size, um, max_probe_size, t_probe, c_probe, sensor_delta, progress, nb):
    # Interpolate the location of the bubble at t = AT
    X_b_AT = SBG_interp_trajectory(t_traj, X, AT[kk])
    # Estimate timeframe for tracking the movement of bubble kk
    critical_time_pre = 3.0*np.linalg.norm(b_size[kk,:])/np.linalg.norm(um)
    critical_time_post = 3.0*(np.linalg.norm(b_size[kk,:]) + \
                         max_probe_size)/np.linalg.norm(um)
    t_min = AT[kk] - critical_time_pre
    t_max = AT[kk] + critical_time_post
    # Get probe sampling times that lie within the estimated timeframe
    t_probe_kk = t_probe[(t_probe >= t_min) & (t_probe <= t_max)]
    # Check if number of samples lies inside the timeframe is larger than 0
    results = {}
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
        # Get probe locations that lie within the estimated timeframe
        c_probe_kk = c_probe[(t_probe >= t_min) & (t_probe <= t_max),:]
        # Set x-coordinate of probe to x-coordinate ob bubble center at AT
        c_probe_kk[:,0] = X_b_AT[0]
        # Set y-coordinate of probe to y-coordinate ob bubble center + random
        # shift with ~Uniform[-B/2,B/2]
        c_probe_kk[:,1] = c_probe_kk[:,1] + X_b_AT[1] + X_rand[kk,0]
        # Set z-coordinate of probe to z-coordinate ob bubble center + random
        # shift with ~Uniform[-B/2,B/2]
        c_probe_kk[:,2] = c_probe_kk[:,2] + X_b_AT[2] + X_rand[kk,1]
        # Loop over all time steps within the timeframe
        abc = b_size[kk,:] / 2.0
        # Determine the row in the signal time series where to write the signal
        row = np.where(t_probe == min(t_probe_kk))[0][0]
        # Loop over each sensor and check if it is inside the bubble
        results = {}
        for idx,delta in sensor_delta.items():
            # Check if ellipsoid is pierced by sensor idx
            # Standard euqation: (x/a)2 + (y/b)2 + (z/c)2 = 1
            # with x = (cx+delta - x_bubble)
            radius = \
                    (((c_probe_kk[:,0]+delta[0])-X_resampled[:,0])/abc[0])**2 \
                  + (((c_probe_kk[:,1]+delta[1])-X_resampled[:,1])/abc[1])**2 \
                  + (((c_probe_kk[:,2]+delta[2])-X_resampled[:,2])/abc[2])**2
            # Check for which time steps the bubble is pierced
            idxs = np.where(radius <= 1) + row
            # pierced, set signal to 1
            results[idx] = idxs
        # Display progress
        if progress:
            printProgressBar(kk + 1, nb, prefix = 'Progress:', suffix = 'Complete', length = 50)
        return results
    else:
        for idx,delta in sensor_delta.items():
            results[idx] = np.array([])
        return results

    # # parallel 1
    # t1 = time.time()
    # folder = './joblib_memmap'
    # try:
    #     Path.mkdir(path / folder)
    # except FileExistsError:
    #     pass

    # # signal_filename_memmap = path / folder / 'signal_memmap'
    # # dump(signal, signal_filename_memmap)
    # signal = np.zeros((n_probe, n_sensors)).astype('uint8')

    # signal_output_filename_memmap =  path / folder / 'signal_output_memmap'

    # signal_output = np.memmap(signal_output_filename_memmap, dtype=signal.dtype,
    #                shape=signal.shape, mode='w+')

    # # signal = load(signal_filename_memmap, mmap_mode='r')
    # print(f'signal_output = {signal_output}')
    # print(f'sum signal_output = {np.sum(signal_output)}')
    # Parallel(n_jobs=nthreads)(delayed(get_signal_parallel)(kk, t_traj, X, X_rand, AT, b_size, um, max_probe_size, t_probe, c_probe, sensor_delta, signal_output, progress, nb) for kk in range(0,nb))
    # print(f'sum signal_output = {np.sum(signal_output)}')

    # try:
    #     shutil.rmtree(folder)
    # except:  # noqa
    #     print('Could not clean-up automatically.')
    # t2 = time.time()
    # print(f'time = {t2-t1}s')

    # # parallel 2
    # t1 = time.time()

    # signal = np.zeros((n_probe, n_sensors)).astype('uint8')
    # results = Parallel(n_jobs=nthreads)(delayed(get_signal_parallel_new)(kk, t_traj, X, X_rand, AT, b_size, um, max_probe_size, t_probe, c_probe, sensor_delta, progress, nb) for kk in range(0,nb))
    # for bubble in results:
    #     for key in bubble:
    #         signal[bubble[key],key] = 1
    # t2 = time.time()
    # print(f'time = {t2-t1}s')

    # idea: delete duration with constant signal to reduce space
    # --> not a good idea, since AWCC is based on entire signal

    # data_type = np.int8
    # shape = (len(signal),)
    # del_idx = np.zeros(shape,dtype=np.int8)

    # def process(kk):
    #     if (signal[(kk-1):(kk+2),:] == 0).all():
    #         # delete this row
    #         del_idx[kk] = True

    # t1 = time.time()
    # for kk in range(1,len(signal)-1):
    #         process(kk)
    # t2 = time.time()
    # print(f'time = {t2-t1}s')

    # signal = np.delete(signal,del_idx.nonzero(),0)
    # t_probe = np.delete(t_probe,del_idx.nonzero(),0)