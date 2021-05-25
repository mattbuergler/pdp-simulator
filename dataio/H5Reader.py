import sys
import h5py
import numpy as np
import pathlib
main=pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(main / 'tools'))
sys.path.append(str(main / 'dataio'))
try:
    from H5Base import H5Base
    from globals import *
except ImportError:
    print('Failed to import modules')
    raise

# class to read HDF5 files
class H5Reader(H5Base):
    def __init__(self, fname):
        # check if file exits
        if fname.is_file():
            super(H5Reader, self).__init__(fname, 'r')
        else:
            PRINTERRORANDEXIT(f"file <{fname}> does not exist");


    def getDataSets(self):
        return list(self.getF5().keys())

    def getDataSet(self, path):
        if not self.existDataset(path):
            PRINTERRORANDEXIT(f'dataset <{path}> does not exist');
        else:
            ds = self.getF5()[path];
        return ds