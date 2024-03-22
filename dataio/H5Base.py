#!/usr/bin/env python3

"""
    Filename: H5Base.py
    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
    Date created: January 1, 2024
    Description:

    Base class for handling H5 files.

"""

# (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero,
# Benjamin Hohermuth, David F. Vetsch, Robert M. Boes,
# D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)
# This software is released under the the GNU General Public License v3.0.
# https://https://opensource.org/license/gpl-3-0

import h5py
import numpy as np
import pathlib

# class to read HDF5 files
class H5Base:
    def __init__ (self, fname, access):
        self.__filename = pathlib.Path(fname)
        self.__f5 = h5py.File(fname, access)

    def getFile(self):
        return self.__filename

    def getF5(self):
        return self.__f5

    def close(self):
        self.__f5.close()

    def existDataset(self, path):
        return path in self.__f5

    def existGroup(self, path):
        return path in self.__f5


    def getGroup(self, path):
        index = path.find("/");
        if (index >= 0):
            return path.split('/')[0];
        else:
            return ""

    def getDataset(self, path):
        index = path.find("/")
        if (index >= 0):
            return path.split('/')[1];
        else:
            return path;

    def createGroup(self, path):
        if not self.existGroup(path):
            g = self.getGroup(path);
            self.__f5.create_group(g);
