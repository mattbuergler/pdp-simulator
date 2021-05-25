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

# test = np.ones((10,3))
# test[:,0] = 1.0
# test[:,1] = 2.0
# test[:,2] = 3.0
# h5f = h5py.File('test1.h5', 'w')
# h5f.create_dataset('test', data=test)
# h5f.close()
# writer = H5Writer('test2.h5', 'w')
# writer.writeDataSet('test',test)
# writer.close()
# writer = H5Writer('test3.h5', 'w')
# writer.createDataSet('test',test.shape,dtype='f')
# writer.write2DataSet('test',test,row=0,col=0)
# writer.close()

n = 10
n_chunk_max = 3

# Option 1: write line by line to .h5
time1 = time.time()
np.random.seed(42)
writer = H5Writer('mytestfile_cont.h5', 'w')
writer.createDataSet('velocity', (n,3), 'f')
for ii in range(0,n):
    writer.write2DataSet('velocity', np.random.multivariate_normal(mean, cov, 1), row=ii, col=0)
writer.close()



# Option 2: write everything in np-array and then write to .h5
time2 = time.time()
np.random.seed(42)
vel = np.zeros((n,3))
for ii in range(0,n):
    vel[ii,:] = np.random.multivariate_normal(mean, cov, 1)
writer = H5Writer('mytestfile_once.h5', 'w')
writer.writeDataSet('velocity', vel, 'f')
writer.close()


# Option 3: write in chunks of size n_chunk to .h5
time3 = time.time()
np.random.seed(42)
writer = H5Writer('mytestfile_chunk.h5', 'w')
writer.createDataSet('velocity', (n,3), 'f')
kk = 0
while (kk < n):
    n_chunk = min(n_chunk_max,n-kk)
    vel = np.zeros((n_chunk,3))
    for ii in range(0,n_chunk):
        vel[ii,:] = np.random.multivariate_normal(mean, cov, 1)
    writer.write2DataSet('velocity', vel, col=0, row=kk)
    kk+=n_chunk
writer.close()


# Option 4: write chunks of size n_chunk to .h5 and reduce np.random calls
time4 = time.time()
np.random.seed(42)
writer = H5Writer('mytestfile_chunk_opt.h5', 'w')
writer.createDataSet('velocity', (n,3), 'f')
kk = 0
while (kk < n):
    n_chunk = min(n_chunk_max,n-kk)
    vel = np.zeros((n_chunk,3))
    T = np.random.multivariate_normal(mean, cov, n_chunk)
    for ii in range(0,n_chunk):
        vel[ii,:] = T[ii,:]
    writer.write2DataSet('velocity', vel, col=0, row=kk)
    kk+=n_chunk
writer.close()

time5 = time.time()

print(f'continuous writing: {time2-time1}s')
print(f'once writing: {time3-time2}s')
print(f'chunk writing: {time4-time3}s')
print(f'optimized chunk writing: {time5-time4}s')