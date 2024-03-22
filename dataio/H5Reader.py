#!/usr/bin/env python3

"""
    Filename: H5Reader.py
    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
    Date created: January 1, 2024
    Description:

    Base class for reading H5 files.

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

# class to read HDF5 files
class H5Reader(H5Base):
    def __init__(self, fname):
        # check if file exits
        if fname.is_file():
            super(H5Reader, self).__init__(fname, 'r')
        else:
            PRINTERRORANDEXIT(f"file <{fname}> does not exist")


    def getDataSets(self):
        return list(self.getF5().keys())

    def getDataSet(self, path):
        if not self.existDataset(path):
            PRINTERRORANDEXIT(f'dataset <{path}> does not exist')
        else:
            ds = self.getF5()[path]
        return ds

    def getRowsFromDataSet(self, path, row_start, row_end):
        if not self.existDataset(path):
            PRINTERRORANDEXIT(f'dataset <{path}> does not exist')
        else:
            ds = self.getF5()[path]
            if len(ds[:].shape) > 1:
                ds = ds[row_start:(row_end+1),:]
            else:
                ds = ds[row_start:(row_end+1)]
        return ds