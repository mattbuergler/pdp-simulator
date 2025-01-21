#!/usr/bin/env python3

"""
    Filename: TestDefinition.py
    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
    Date created: January 1, 2024
    Description:

    TestDefinition class.
    Holds data file name, information on columns that should be compare etc.

"""

# (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero,
# Benjamin Hohermuth, David F. Vetsch, Robert M. Boes,
# D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)
# This software is released under the the GNU General Public License v3.0.
# https://https://opensource.org/license/gpl-3-0


import math
import sys
import re
import numbers
import h5py
import numpy as np
import pathlib
import typing

main=pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(main / 'tools'))
try:
    from globals import *
    from TestDefinition import *
except ImportError:
    print('Failed to import nodules')
    raise


# trying to import matplotlib
_PLT_FIGURE_NUM = 0  # type: typing.Optional[int]
try:
    import matplotlib.pyplot as plt
except ImportError:
    _PLT_FIGURE_NUM = None


class TestDefinition:
    def __init__(
        self,
        tagName,
        fileName,
        dataSet,
        columns,
        rows,
        relativeError,
        absoluteError
    ):
        self.tagName = tagName
        self.runFile = fileName
        self.dataSet = dataSet
        self.columns = columns
        self.col_list = []
        self.rows = rows
        self.row_list = []
        self.relativeError = float(relativeError)
        self.absoluteError = float(absoluteError)
        if self.relativeError <= 0 and self.absoluteError < 0:
            PRINTERRORANDEXIT("the error definitions for %s are bullshit! " % tagName)

    def __str__(self):
        out = "test definition %s:\n" % self.tagName
        out += "  runfile: %s\n" % self.runFile
        out += "  columns: %s\n" % self.columns
        out += "  rows:    %s\n" % self.rows
        out += "  relErr:  %f\n" % self.relativeError
        out += "  absErr:  %f\n" % self.absoluteError
        return out

    def getColumns(self, errors: typing.List[str], size=None):
        if len(self.col_list) == 0 and size != None:
            self.col_list = self.createListEntries(self.columns, size, errors)
        return self.col_list

    def getRows(self, errors: typing.List[str], size=None):
        if len(self.row_list) == 0 and size != None:
            self.row_list = self.createListEntries(self.rows, size, errors)
        return self.row_list

    def createListEntries(self, entryString, size, errors: typing.List[str]):
        entryList = []  # type: typing.List[int]
        if entryString != -1:
            splitList = entryString.split(":")  # ':'
            err = False
            for ii, ci in enumerate(splitList):
                if len(ci) == 0:
                    # add range
                    if ii <= 0 or ii >= len(splitList):
                        err = True
                    else:
                        ind1 = int(splitList[ii - 1])
                        ind2 = int(splitList[ii + 1])
                        entryList.extend(range(ind1, ind2 + 1))
                else:
                    # add single
                    ci = int(ci)
                    if ci < 0 or ci >= size:
                        err = True
                    else:
                        entryList.append(ci)
            if err:
                errors.append("invalid range: " + str(entryString))
        else:
            entryList = list(range(0, size))
        return list(set(entryList))

    def compare(self, rootDirectory: pathlib.Path, visual=0):
        """
            compare test values with reference
        """
        reasons = []  # type: typing.List[str]
        # read run data
        dataRun = self.readH5DataSet(
            rootDirectory / "run" / self.runFile, self.dataSet, reasons,
        )
        if len(reasons) == 0:
            # cut the appropriate data range
            dataRun = self.cutDataRange(dataRun, reasons)
        if len(reasons) == 0:
            # read ref data
            dataRef = self.readXYDataFile(
                rootDirectory / "ref" / self.getRefFileName(), reasons,
            )
        if len(reasons) == 0:
            # check for correct dimensions
            if dataRun.shape != dataRef.shape:
                reasons.append(
                    "dimensions in run and ref not equal: (%i,%i) <-> (%i,%i)"
                    % (
                        dataRun.shape[0],
                        dataRun.shape[1],
                        dataRef.shape[0],
                        dataRef.shape[1],
                    )
                )
        if len(reasons) == 0 and not visual:
            # compare values
            ret = self.compareValues(dataRun, dataRef)
            if not ret == -1:
                reasons.append(ret)
        if len(reasons) == 0:
            print(
                " PASSED - %s in %s"
                % (
                    rootDirectory,
                    self.tagName,
                )
            )
            return 0
        else:
            print(" FAILED - %s in %s" % (rootDirectory, self.tagName))
            for r in reasons:
                print("          %s" % r)
            return 1

    def updateTest(self, rootDirectory, refDir: pathlib.Path):
        reasons = []  # type: typing.List[str]
        fileName = refDir / self.getRefFileName()
        # read run data
        dataRun = self.readH5DataSet(
            rootDirectory / "run" / self.runFile, self.dataSet, reasons
        )
        if len(reasons) == 0:
            # cut the appropriate data range
            dataRun = self.cutDataRange(dataRun, reasons)
        if len(reasons) == 0:
            # write reference
            if dataRun.dtype == "object":
                formatter = "%s"
            else:
                formatter = "%16.12f"
            np.savetxt(fileName, dataRun, fmt=formatter)
        if len(reasons) == 0:
            print(" UPDATED %s" % fileName)
            return 0
        else:
            print(" FAILED  %s" % fileName)
            for r in reasons:
                print("         %s" % r)
            return 1

    def getRefFileName(self):
        out = self.runFile.replace(".h5", "_") + self.dataSet.replace("/", "_")

        if self.columns != -1:
            cols = self.columns.replace(":", "-")
            out += "_c" + cols  # self.columns
        out += ".dat"
        return out

    def cutDataRange(self, dataRun, errors):
        rows = self.getRows(errors, dataRun.shape[0])
        cols = self.getColumns(errors, dataRun.shape[1])
        if len(errors) == 0:
            data = dataRun
            deleteIndex = np.setdiff1d(range(dataRun.shape[0]), rows)
            data = np.delete(data, deleteIndex, axis=0)
            deleteIndex = np.setdiff1d(range(dataRun.shape[1]), cols)
            data = np.delete(data, deleteIndex, axis=1)
            return data
        else:
            return None

    def compareValues(self, dataRun, dataRef):
        """
            compare data in column cidx
        """
        ret = []  # type: typing.List[str]
        num_errors = 0
        errors = []  # type: typing.List[str]
        shape = dataRun.shape
        for ii in range(shape[0]):
            ridx = self.getRows(errors)[ii]
            for jj in range(shape[1]):
                cidx = self.getColumns(errors)[jj]
                a = dataRun[ii][jj]
                b = dataRef[ii][jj]
                if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
                    num_errors = self.compareNumericValues(
                        a, b, (ridx, cidx), num_errors, ret
                    )
                else:
                    num_errors = self.compareObjects(
                        a, b, (ridx, cidx), num_errors, ret
                    )
        if len(ret) > 0:
            ret.append(
                "\n          %d out of %d data points failed"
                % (num_errors, shape[0] * shape[1])
            )
            out = ""
            for line in ret:
                out += line
            return out
        else:
            return -1

    def addErrorOutputPrefix(self, ret, numErrors):
        if numErrors == 10:
            ret.append("\n          too many errors! stopping output!\n")
            return False
        elif numErrors < 10:
            if len(ret) > 0:
                ret.append("\n          ")
            return True

    def compareNumericValues(self, run, ref, inds, numErrors, ret):
        f = 1
        if run != 0:
            f = run
        elif ref != 0:
            f = ref
        actualErrorRelative = abs(run - ref) / abs(f)
        actualErrorAbsolute = abs(run - ref)

        ErrorAbsolute = False  # type: typing.Union[bool, float]
        ErrorRelative = False  # type: typing.Union[bool, float]
        if math.isnan(run) is not math.isnan(ref):
            ErrorAbsolute = float("nan")
        if self.absoluteError != -1 and actualErrorAbsolute > self.absoluteError:
            ErrorAbsolute = actualErrorAbsolute
        if self.relativeError != -1 and actualErrorRelative > self.relativeError:
            # skip it if absolute values are smaller than the absolute error threshold
            if abs(run) > self.absoluteError and abs(ref) > self.absoluteError:
                ErrorRelative = actualErrorRelative

        if ErrorRelative != False or ErrorAbsolute != False:
            if self.addErrorOutputPrefix(ret, numErrors):
                if ErrorRelative != False:
                    ret.append(
                        "row: %d, col: %d, max. rel. err. of %g > %g (abs val: %g)"
                        % (
                            inds[0],
                            inds[1],
                            ErrorRelative,
                            self.relativeError,
                            abs(run),
                        )
                    )
                if ErrorAbsolute != False:
                    if ErrorRelative != False:
                        ret.append(" AND ")
                        ret.append(
                            "max. abs. err. of %g > %g "
                            % (ErrorAbsolute, self.absoluteError)
                        )
                    else:
                        ret.append(
                            "row: %d, col: %d, max. abs. err. of %g > %g (abs val: %g)"
                            % (
                                inds[0],
                                inds[1],
                                ErrorAbsolute,
                                self.absoluteError,
                                abs(run),
                            )
                        )
            numErrors += 1
        return numErrors

    def compareObjects(self, run, ref, inds, numErrors, ret):
        if run != ref:
            if self.addErrorOutputPrefix(ret, numErrors):
                ret.append(
                    "row: %d, col: %d, '%s' not equal to '%s'"
                    % (inds[0], inds[1], run, ref)
                )
            numErrors += 1
        return numErrors

    def readXYDataFile(self, fileName: pathlib.Path, errors):
        """
            read and parse XY Data file consisting either of doubles or even complex numbers
        """
        commentProg = re.compile("^\s*[#%]")
        doubleProg = re.compile("([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)")
        complexProg = re.compile(
            "\(([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?),([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)"
        )
        alphaProg = re.compile("[ABCDFGHJKLMNOPQRSTUVWXYZabcdfghjklmnopqrstuvw]")
        ret = []
        try:
            f = open(str(fileName), "r")
        except IOError:
            errors.append("could not open reference data file %s" % str(fileName))
            return None

        for line in f.readlines():
            if not commentProg.search(line):
                dataLine = []  # type: typing.List[typing.Union[str, complex]]
                # replace tabs with whitespaces
                line = line.replace("\t", " ")
                # replace commas with whitespaces
                line = line.replace(",", " ")
                items = line.split(" ")
                for item in items:
                    item = item.strip()
                    if not alphaProg.search(item):
                        if complexProg.search(item):
                            mobj = complexProg.match(item)
                            assert not mobj is None
                            real = float(mobj.expand("\\1"))
                            imag = float(mobj.expand("\\5"))
                            dataLine.append(complex(real, imag))
                        elif doubleProg.search(item):
                            dataLine.append(float(item))
                    else:
                        if (item == 'nan'):
                            dataLine.append(float(item))
                        else:
                            dataLine.append(item)
                if len(dataLine) > 0:
                    ret.append(dataLine)
        if len(ret) == 0:
            errors.append("could not read reference data file %s" % str(fileName))
            return None
        return np.array(ret)

    def readH5DataSet(self, fileName: pathlib.Path, dataset, errors):
        """
            read XY Data from h5 container
        """
        try:
            h5 = h5py.File(str(fileName), "r")
        except IOError:
            errors.append("could not open data file %s" % str(fileName))
            return None
        try:
            d = h5[dataset]
        except KeyError:
            h5.close()
            errors.append("could not read data from dataset %s" % dataset)
            return None
        data = np.array(d)
        if len(data.shape) == 1:
            # (N,)
            data = data.reshape(data.shape[0], 1)
        h5.close()
        return data
