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
    from global_functions import *
except ImportError:
    print('Failed to import modules')
    raise

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


def main():

    path = pathlib.Path('../lda_validation/run_20_test/run')

    # Create a H5-file reader
    reader = H5Reader(path / 'binary_signal.h5')
    # Read the time vector
    t_signal = reader.getDataSet('time')[:]
    # Read the signal time series
    ds_signal = reader.getDataSet('signal')
    # Get the signal
    signal = np.array(ds_signal, dtype='int8')
    # Get the corresponding sensor ids
    sensor_ids = ds_signal.attrs['sensor_id']
    reader.close()

    write_compressed_signal(path,t_signal,signal,sensor_ids)
    t_signal_decompressed,signal_decompressed,sensor_ids_decompressed = read_compressed_signal(path)
    print((t_signal_decompressed == t_signal).all())
    print((signal_decompressed == signal).all())
    print((sensor_ids_decompressed == sensor_ids).all())


if __name__ == "__main__":
    main()