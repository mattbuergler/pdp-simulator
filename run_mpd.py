#!/usr/bin/env python3

import os
import sys
import time
import pathlib
import argparse
import shutil
import json
import numpy as np
import re
import socket
import collections
import typing
import subprocess

main=pathlib.Path(__file__).resolve().parents[0]
tools=main / 'tools'
dataio=main / 'dataio'
sys.path.append(str(main))
sys.path.append(str(tools))
sys.path.append(str(dataio))
try:
    from H5Writer import H5Writer
    from H5Reader import H5Reader
    from globals import *
    from Runner import *
except ImportError:
    print("Error while importing modules")
    raise



"""
    simple running of multi-phase-detection simulation

    1. every run setup is given in a separate directory
    2. the directory content is given by
         - input/  the unmodified input data
         - run/    the directory where input data is copied to and executed
"""

if __name__ != "main":

    parser = argparse.ArgumentParser(
        description="TESTING FRAMEWORK FOR THE STOCHASTIC BUBBLE GENERATOR",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "runs",
        metavar="runs",
        type=str,
        nargs="+",
        help="diretories that contain run definitions",
    )
    parser.add_argument(
        "-n",
        "--nthreads",
        metavar="NTHREADS",
        default=1,
        help="set the number of threads for parallel execution",
    )
    parser.add_argument(
        "-b", "--bin",
        metavar="bin",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent),
        help="diretories that contain the functions (e.g. sbg.py, mssrc.py, etc.)",
    )
    parser.add_argument('-tsa', '--velocity_tsa', action='store_true',
        help="Vizualize the results.", default=False)
    parser.add_argument('-roc', '--ROC', default=True,
        help='Perform robust outlier cutoff (ROC) based on the maximum' +
                'absolute deviation and the universal threshold (True/False).')
    args = vars(parser.parse_args())
    start_time = time.time()
    errors = 0
    config = {}
    config['velocity_tsa'] = args["velocity_tsa"]
    config['ROC'] = args["ROC"]
    config['bin'] = args["bin"]
    config['nthreads'] = str(args["nthreads"])

    runner = Runner(config)
    # and the runs
    for target_dir in args["runs"]:
        if not pathlib.Path(target_dir).is_dir():
            PRINTERRORANDEXIT("directory " + target_dir + " does not exist")
        runner.findRuns(pathlib.Path(target_dir))

    runner.displayNumRuns()
    errors = runner.run()

    end_time = time.time()
    PRINTTITLE(
        "mpd.py was running for %d seconds" % int(end_time - start_time), "-"
    )

    # return success (or not: number of failed runs)
    sys.exit(-errors)