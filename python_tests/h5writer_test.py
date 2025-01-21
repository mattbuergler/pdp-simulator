import numpy as np
import h5py
import time
from dataio.H5Writer import H5Writer

n = 10
print('creating writer')
writer = H5Writer('h5writer_test.h5', 'w')
# Create the velocity data set for the entire time series of length n
# Initialize with mean flow velocity and write to file
print('wrote test1')
test = np.ones((3,1))
writer.writeDataSet('test1',test, dtype='f')
print('wrote test1')
writer.createDataSet('test2', (n,3), 'f')
print('created test2')
writer.write2DataSet('test2', test, row=0, col=0)
print('wrote test2')

writer.close()