import numpy as np
import h5py
import time
from dataio.H5Writer import H5Writer

# Define the correlation parameters with standardized multivariate Gaussian
# Zero mean
mean = [0.0, 0.0, 0.0]
# Zero Z-Components of covariance in a statistically 2D flow (Pope, 2000)
pXZ = 0.0
pYZ = 0.0
# rho_{uv}=-0.45 at y+~98 (Table 7.2 Pope, 2000)
pXY = -0.45
# Covariance with diagonal components = 1
cov = [[1.0, pXY, pXZ], \
       [pXY, 1.0, pYZ], \
       [pXZ, pYZ, 1.0]]

n = 10

# Option 1: generate single lines
np.random.seed(42)

vel = np.zeros((n,3))
for ii in range(0,n):
    vel[ii,:] = np.random.multivariate_normal(mean, cov, 1)

print(f"single lines = {vel}")

# Option 2: generate all at once
np.random.seed(42)

vel = np.zeros((n,3))
vel[:,:] = np.random.multivariate_normal(mean, cov, n)

print(f"all at once = {vel}")

# Option 3: write in chunks of size n_chunk to .h5
np.random.seed(42)
vel = np.zeros((n,3))
kk = 0
while (kk < n):
    n_chunk = min(2,n-kk)
    veltmp = np.zeros((n_chunk,3))
    R = np.random.multivariate_normal(mean, cov, n_chunk)
    for ii in range(0,n_chunk):
        veltmp[ii,:] = R[ii,:]
    vel[kk:(kk+veltmp.shape[0]),:] = veltmp
    kk+=n_chunk
print(f"chunk wise = {vel}")
