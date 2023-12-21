#!/usr/bin/env python3

import sys
import pathlib
import subprocess
import shutil
from typing import List
import numpy as np
import math
main=pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(main / 'dataio'))
try:
    from H5Writer import H5Writer
    from H5Reader import H5Reader
except ImportError:
    print('Failed to import modules')
    raise
"""
    global variables and functions
"""
SMALLNUMBER = 1.e-12
LARGENUMBER= 1.e12
LARGENEGNUMBER=-1.e12

# Improved by using f-string for formatting
def PRINTLOG(log, string):
    print(f'* {log + ": " if log else ""}{string}')

def PRINTTITLE(string, char='*'):
    border = char * (len(string) + 2)
    print(f'\n{border}')
    PRINTLOG('', string)
    print(border)

def PRINTWARNING(string):
    PRINTLOG('warning', string)

def PRINTERRORANDEXIT(string):
    PRINTLOG('error', string)
    sys.exit(2)

# Using f-string for percent calculation
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    percent = f"{100 * iteration / total:.{decimals}f}"
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total:
        print()

# No changes required for these functions. They're already optimal given their purpose.
def round_nearest(num: float, to: float) -> float:
    return round(num / to) * to

def round_down(num: float, to: float) -> float:
    nearest = round_nearest(num, to)
    return nearest if math.isclose(num, nearest) or nearest < num else nearest - to

def round_up(num: float, to: float) -> float:
    nearest = round_nearest(num, to)
    return nearest if math.isclose(num, nearest) or nearest > num else nearest + to

# np.abs and argmin are already quite optimal for this purpose
def find_nearest(array, value):
    array = np.asarray(array)
    return array[np.abs(array - value).argmin()]

def find_nearest_idx(array, value):
    array = np.asarray(array)
    return np.abs(array - value).argmin()

# Using np's built-in capabilities to enhance performance
def find_nearest_smaller_idx(array, value):
    array = np.asarray(array)
    idxs = np.where(array <= value)
    return idxs[0][np.abs(array[idxs] - value).argmin()]

# Using numpy for vector operations
def calculate_unit_vector(point1, point2):
    vec = np.subtract(point2, point1)
    magnitude = np.linalg.norm(vec)
    if magnitude == 0:
        return None
    return tuple(vec / magnitude)

# Improved the function by handling zero division upfront
def inverse_den(x):
    if abs(x) < SMALLNUMBER:
        return 0.0
    return 1.0 / x

# The following functions are already quite optimal given their purpose. They use standard library functions in a straightforward manner.
def copy_file_to(file: pathlib.Path, dir: pathlib.Path):
    shutil.copy(str(file), str(dir))

def delete_files_in_directory(directory: pathlib.Path):
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()

def create_dir_if_not_exists(path: pathlib.Path):
    path.mkdir(exist_ok=True)

def copy_files(from_dir: pathlib.Path, to_dir: pathlib.Path):
    for item in from_dir.iterdir():
        if item.is_file():
            shutil.copy(str(item), str(to_dir))

def delete_file_if_exists(path: pathlib.Path):
    if path.is_file():
        path.unlink()

def run_process(
    arguments: List[str], work_dir: pathlib.Path, capture_output: bool = True
):
    print(work_dir)
    create_dir_if_not_exists(work_dir)
    if capture_output:
        return subprocess.run(
            arguments,
            check=True,
            cwd=str(work_dir),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    return subprocess.run(
        arguments, check=True, cwd=str(work_dir), universal_newlines=True
    )

# git functions are straightforward wrappers around subprocess calls and don't need optimization.
def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_git_version() -> str:
    return subprocess.check_output(['git', 'describe', '--always','--long']).decode('ascii').strip()

def printHeader():
    print(f'MULTIPHADE {get_git_version()}\n')

def compress_signal(signal):
    signal = np.array(signal)
    # Find the indices where the value changes
    change_indices = np.nonzero(np.diff(signal))[0] + 1
    # Append the initial value and change indices
    compressed = np.insert(change_indices, 0, signal[0])
    return compressed

def decompress_signal(compressed, length):
    signal = np.zeros(length, dtype=int)
    signal[compressed[1:]] = 1 - signal[compressed[1:]]  # Toggle values at change points
    signal = np.cumsum(signal) % 2  # Cumulative sum and modulo to alternate between 0 and 1
    return signal

def get_signal_frequency(t_signal):
    f_s = 1.0/((t_signal[-1]-t_signal[0])/(len(t_signal)-1))
    return f_s

def write_compressed_signal(path,t_signal,signal,sensor_ids):
    f_s = get_signal_frequency(t_signal)
    length = len(t_signal)
    compressed_signal = {}
    for ii,sensor_id in enumerate(sensor_ids):
        compressed_signal[sensor_id] = compress_signal(signal[:,ii])
    # Create the H5-file writer
    writer = H5Writer(path / 'compressed_signal.h5', 'w')
    # Write the time vector
    writer.writeDataSet('length', np.array([length],dtype='int32'), 'int32')
    writer.writeDataSet('f_s', np.array([f_s],dtype='float64'), 'float64')
    writer.writeDataSet('sensor_ids', np.array(sensor_ids,dtype='int16'), 'int16')
    for ii,sensor_id in enumerate(sensor_ids):
        writer.writeDataSet(f'signal{int(sensor_id)}', compressed_signal[sensor_id], 'int32')
        ds_sig = writer.getDataSet(f'signal{int(sensor_id)}')
        ds_sig.attrs['sensor_id'] = int(sensor_id)
    writer.close()

def read_compressed_signal(path):
    # Create a H5-file reader
    reader = H5Reader(path / 'compressed_signal.h5')
    # Read the sampling frequency
    f_s = reader.getDataSet('f_s')[:][0]
    # Read the signal length
    length = reader.getDataSet('length')[:][0]
    # Get the sensor ids
    sensor_ids = reader.getDataSet('sensor_ids')[:]
    signal = np.empty((length,len(sensor_ids)), dtype='int8')
    for ii,sensor_id in enumerate(sensor_ids):
        # Read the compressed signal
        compressed_signal = reader.getDataSet(f'signal{int(sensor_id)}')
        signal[:,ii] = decompress_signal(compressed_signal,length)
    reader.close()
    t_signal = np.linspace(0,(length-1)/f_s,length)
    return t_signal,signal,sensor_ids
