import sys
import h5py
import numpy as np
import pathlib

main=pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(main / 'tools'))
sys.path.append(str(main / 'dataio'))
try:
    from H5Writer import H5Writer
    from H5Reader import H5Reader
    from globals import *
except ImportError:
    print('Failed to import modules')
    raise

path = pathlib.Path().cwd()
n_probe = 500000
t_probe = np.linspace(0.0, 1.0, n_probe+1);
print(f't_probe = {repr(t_probe[0:10])}')
print(f'type(t_probe[0]) = {type(t_probe[0])}')
# Create the H5-file writer
writer = H5Writer(path / 'test.h5', 'w')
writer.writeDataSet('time', t_probe, 'float64')

# Create the H5-file writer
reader = H5Reader(path / 'test.h5')
t = reader.getDataSet('time')
print(f't = {repr(t[0:10])}')
print(f'type(t[0]) = {type(t[0])}')


t_ifd = t[0:(len(t)-1)] + (t[1:len(t)] - t[0:(len(t)-1)])/2.0
print(f't_ifd = {repr(t_ifd[0:10])}')
print(f'type(t_ifd[0]) = {type(t_ifd[0])}')
delta_t_k_h = {}

delta_t_k_h[0] = t_ifd[9] - t_ifd[3]
print(f'delta_t_k_h[0] = {repr(delta_t_k_h[0])}')
print(f'delta_t_k_h[0] true = {1.9e-05-7e-06}')
print(f'delta_t_k_h[0] diff = {delta_t_k_h[0]-(1.9e-05-7e-06)}')

t_vel = np.array([0.0,1.0,2.0,3.0])
vel = np.array([1.1,1.2,1.3,1.4])
t = np.array([1.0,2.0])

idmin = max(bisect.bisect_left(t_vel, t[0]), 0)
idmax = min(bisect.bisect_right(t_vel, t[1]),len(t_vel))
print(f'vel_range = {vel[idmin:idmax]}')