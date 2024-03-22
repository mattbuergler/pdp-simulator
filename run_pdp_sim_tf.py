#!/usr/bin/env python3

"""
    Filename: run_pdp_sim_tf.py
    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
    Date created: January 1, 2024
    Description:

    Script for running the various steps of a phase-detection probe simulations.
    The possible steps include:
        - stochastic time series generation
        - synthetic signal generation
        - processing of the signal

"""

# (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero,
# Benjamin Hohermuth, David F. Vetsch, Robert M. Boes,
# D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)
# This software is released under the the GNU General Public License v3.0.
# https://https://opensource.org/license/gpl-3-0

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
    Tool for running of phase-detection probe simulations in turbulent flows

    1. Every simulation is given in a separate directory
    2. the directory content is given by
         - input/  the unmodified input data (config.json)
         - run/    the directory where input data is copied to and executed
"""

if __name__ != "main":

    parser = argparse.ArgumentParser(
        description="""\
        Framework for running of phase-detection probe simulations in turbulent flows

        1. Every simulation is given in a separate directory
        2. the directory content is given by
            - input/  the unmodified input data (config.json)
            - run/    the directory where input data is copied to and executed
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="diretories that contain the functions (e.g. stsg_ssg.py, mssp.py, etc.)",
    )
    parser.add_argument(
        "-r", "--run",
        metavar="OPTION",
        help="the following options are available:\n"
        + "full        - timeseries and signal generation, signal processing and evaluation\n"
        + "timeseries  - timeseries generation\n"
        + "signal      - signal generation\n"
        + "mssp        - signal processing\n"
        + "evaluation  - run evaluation (mostly figure plotting)",
    )
    parser.add_argument(
        "-p",
        "--postprocess",
        metavar="OPTION",
        default=None,
        help="the following options are available:\n"
        + "filter     - filtering of velocity values based on a the mean and deviation\n"
        + "roc        - perform robust outlier detection\n"
        + "all        - perform roc and filtering\n"
    )
    parser.add_argument(
        "-ft",
        "--filter-type",
        metavar="OPTION",
        default="max",
        help="Filtering of velocity time series. The following options are available:\n"
        + "awcc_roc   - based on mean and RMS velocity from AWCC\n"
        + "max_roc    - ROC based on mean from awcc and median deviation of velocity time series values larger than the mean.\n"
        + "max        - based on mean from awcc and standard deviation of velocity time series values larger than the mean.\n"
        + "max_med    - based on mean from awcc and median deviation of velocity time series values larger than the mean.\n"
        + "none       - no filtering.\n"
    )
    parser.add_argument('-roc', '--ROC', default=False, metavar='BOOL',
        help='Perform robust outlier cutoff (ROC) based on the maximum' +
                'absolute deviation and the universal threshold (True/False).')
    args = vars(parser.parse_args())
    start_time = time.time()
    errors = 0
    config = {}
    config['run'] = args["run"]
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
    if args["run"]:
        if not args["run"] in ["full", "timeseries", "signal", "mssp", "evaluation"]:
            PRINTERRORANDEXIT("Option <" + args["run"] + "> does not exist")
        error = runner.run()
        errors += error

    if args['postprocess']:
        config = {}
        config['filter'] = False
        config['roc'] = False
        if args['postprocess'] in ['filter', 'all']:
            config['filter'] = True
            config['filter_type'] = args['filter_type']
        if args['postprocess'] in ['roc', 'all']:
            config['roc'] = True
        runner.setupPostProcess(config)
        error = runner.runPostProcessing()
        errors += error
    end_time = time.time()
    PRINTTITLE(
        "mpd.py was running for %d seconds" % int(end_time - start_time), "-"
    )

    # return success (or not: number of failed runs)
    sys.exit(-errors)