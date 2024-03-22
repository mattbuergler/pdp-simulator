#!/usr/bin/env python3

"""
    Filename: H5Writer.py
    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
    Date created: January 1, 2024
    Description:

    Base class for writing H5 files.

"""

# (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero,
# Benjamin Hohermuth, David F. Vetsch, Robert M. Boes,
# D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)
# This software is released under the the GNU General Public License v3.0.
# https://https://opensource.org/license/gpl-3-0


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

# class to write HDF5 files
class H5Writer(H5Base):
    def __init__(self, fname, access):
        super(H5Writer, self).__init__(fname, access)

    def getDataSet(self, path):
        if not self.existDataset(path):
            PRINTERRORANDEXIT(f'dataset <{path}> does not exist')
        else:
            ds = self.getF5()[path]
        return ds

    def createDataSet(self, path, dims, dtype):
        if self.existDataset(path):
            del self.getF5()[path]
        self.getF5().create_dataset(path, dims, dtype=dtype)

    def writeDataSet(self, path, data, dtype):
        if self.existDataset(path):
            del self.getF5()[path]
        self.getF5().create_dataset(path, data=data, dtype=dtype)

    def write2DataSet(self, path, data, row, col):
        rows = data.shape[0]
        ds = self.getDataSet(path)
        row_start = row
        row_end = row_start + rows
        if len(data.shape) > 1:
            if data.shape[1] > 1:
                cols = data.shape[1]
                col_start = col
                col_end = col_start + cols
                ds[row_start:row_end, col_start:col_end] = data
            else:
                ds[row_start:row_end] = data
        else:
            ds[row_start:row_end] = data

    def writeStringDataSet(self, path, string):
        asciiString = string.encode("ascii", "ignore")
        self.getF5().create_dataset(path, (1,1),
                                   h5py.string_dtype(), asciiString)
