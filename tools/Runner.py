import time
import json
import numpy as np
import pathlib
import typing
import subprocess
import sys

main=pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(main / 'tools'))
try:
    from globals import *
except ImportError:
    print('Failed to import nodules')
    raise

"""
    Runner Class Executing and Evalulating
"""

class Runner:

    runs = []  # type: typing.List[pathlib.Path]
    timings = []  # type: typing.List[float]

    def __init__(self, config):
        self.velocity_tsa = config['velocity_tsa']
        self.roc = config['ROC']
        self.bin = config['bin']
        self.nthreads = config['nthreads']
        self.tasks = config['run']
        self.runs = []  # type: typing.List[str]


    def findRuns(self, directory: pathlib.Path):
        """
        searches current directory tree for input directories
        """
        for directory in directory.rglob("input"):
            self.runs.append(directory.parent)
        self.runs.sort()

    def getNumRuns(self):
        return len(self.runs)

    def setupPostProcess(self, config):
        self.filter = config['filter']
        self.filter_type = config['filter_type']
        self.roc_pp = config['roc']

    def displayNumRuns(self):
        print((" found %d run(s)") % (len(self.runs)))

    def run(self):
        """
        run runs
        """
        errors = 0
        self.timings = []
        for runDirectory in self.runs:
            error = 0
            if self.tasks in ['full', 'timeseries']:
                self.setup(runDirectory)
                t, error = self.runSBG_ts(runDirectory)
                errors += error
                self.timings.append(t)
            if self.tasks in ['full', 'signal']:
                t, error = self.runSBG_sig(runDirectory)
                errors += error
                self.timings.append(t)
            if self.tasks in ['full', 'mssrc']:
                t, error = self.runRA(runDirectory)
                errors += error
                self.timings.append(t)
            if self.tasks in ['full', 'evaluation']:
                t, error = self.runEval(runDirectory)
                errors += error
                self.timings.append(t)

        return errors

    def runPostProcessing(self):
        """
        postprocess runs
        """
        errors = 0
        self.timings = []
        for runDirectory in self.runs:
            error = 0

            t, error = self.postProcess(runDirectory)
            errors += error
            self.timings.append(t)

        return errors

    def setup(self, directory: pathlib.Path):
        """
        delete contents in run directory and copy files from input to run
        """
        create_dir_if_not_exists(directory / "run")
        delete_files_in_directory(directory / "run")
        copy_files(directory / "input", directory / "run")


    def runSBG_ts(self, runDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running time series generation in %s " % str(runDirectory), "-")
        # get the filename of the runfile
        runfile = runDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "sbg.py")
        ]
        cmd.append("-r")
        cmd.append('timeseries')
        cmd.append("-n")
        cmd.append(self.nthreads)
        cmd.append(".")
        return self.runCommon(cmd, runDirectory, "a")

    def runSBG_sig(self, runDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running signal generation in %s " % str(runDirectory), "-")
        # get the filename of the runfile
        runfile = runDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "sbg.py")
        ]
        cmd.append("-r")
        cmd.append('signal')
        cmd.append("-n")
        cmd.append(self.nthreads)
        cmd.append(".")
        return self.runCommon(cmd, runDirectory, "a")

    def runRA(self, runDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running reconstruction in %s " % str(runDirectory), "-")
        # get the filename of the runfile
        runfile = runDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "mssrc.py"),
            "-roc",
            str(self.roc),
            "-n",
            self.nthreads,
            "."
            ]
        return self.runCommon(cmd, runDirectory, "a")


    def runEval(self, runDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running evalutation in %s " % str(runDirectory), "-")
        # get the filename of the runfile
        runfile = runDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "evaluate.py"),
            "."
        ]
        return self.runCommon(cmd, runDirectory, "a")

    def postProcess(self, runDirectory: pathlib.Path):
        """
        post-process velocity values of given simulation in given directory
        """
        PRINTTITLE(" running post-processing in %s " % str(runDirectory), "-")
        # get the filename of the runfile
        reconst_file = runDirectory / "run" / "reconstructed.h5"
        # check for runfile
        if not reconst_file.exists():
            PRINTERRORANDEXIT("runfile <" + str(reconst_file) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "postprocess.py"),
        ]
        if self.filter:
            cmd.append('--filter-type')
            cmd.append(str(self.filter_type))
        cmd.append('--ROC')
        cmd.append(str(self.roc_pp))
        cmd.append(".")
        return self.runCommon(cmd, runDirectory, "a")

    def runTSA(self, runDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running evalutation in %s " % str(runDirectory), "-")
        # get the filename of the runfile
        runfile = runDirectory / "run" / "config.json"
        # check for runfile
        if not runfile.exists():
            PRINTERRORANDEXIT("runfile <" + str(runfile) + "> does not exists")
        # create simulation call
        cmd = [
            "python",
            str(pathlib.Path(self.bin) / "velcoity_tsa.py"),
            "."
        ]
        return self.runCommon(cmd, runDirectory, "a")

    def runCommon(
        self, cmd: typing.List[str], runDirectory: pathlib.Path, fileAccess="w"
    ):
        out = ""
        success = 0
        startingTime = time.time()
        try:
            out = run_process(cmd, runDirectory / "run").stdout
            success = 1
        except subprocess.CalledProcessError as error:
            out = error.stdout
        endTime = time.time()
        with open(str(runDirectory / "run" / "run.output"), fileAccess) as file:
            file.write(out)
        return endTime - startingTime, success