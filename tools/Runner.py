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

            self.setup(runDirectory)
            t, error = self.runSBG(runDirectory)
            errors += error
            self.timings.append(t)
            t, error = self.runRA(runDirectory)
            errors += error
            self.timings.append(t)
            t, error = self.runEval(runDirectory)
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


    def runSBG(self, runDirectory: pathlib.Path):
        """
        run given simulation in given directory
        """
        PRINTTITLE(" running SBG in %s " % str(runDirectory), "-")
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
        cmd.append('all')
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
        # get a velocity estimate
        config = json.loads((runfile).read_bytes())
        vel = np.asarray(config["FLOW_PROPERTIES"]["mean_velocity"])
        mag_vel = np.sqrt(vel.dot(vel))
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