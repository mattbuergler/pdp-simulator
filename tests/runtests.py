#!/usr/bin/env python3

"""
    Filename: runtests.py
    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
    Date created: January 1, 2024
    Description:

    Class for running tests used for integrated testing.

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

main=pathlib.Path(__file__).resolve().parents[1]
tools=main / 'tools'
dataio=main / 'dataio'
sys.path.append(str(main))
sys.path.append(str(tools))
sys.path.append(str(dataio))
try:
    from H5Writer import H5Writer
    from H5Reader import H5Reader
    from globals import *
    from TestRunner import *
    from TestDefinition import *
except ImportError:
    print("Error while importing modules")
    raise

"""
    Simple integrated testing

    1. every test setup is given in a separate directory
    2. the directory content is given by
         - input/  the unmodified input data
         - run/    the directory where input data is copied to and executed
         - ref/    the reference output (so a copy of an older run/ directory)
         - testdef is the file defining the files and columns that should be compared

"""

if __name__ != "main":

    parser = argparse.ArgumentParser(
        description="TESTING FRAMEWORK FOR THE STOCHASTIC BUBBLE GENERATOR",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "tests",
        metavar="test",
        type=str,
        nargs="+",
        help="diretories that contain test definitions",
    )
    parser.add_argument(
        "-c", "--clean", action="store_true", help="set tests to no-passed"
    )
    parser.add_argument(
        "-C", "--Clean", action="store_true", help="removing all files from <test/run>"
    )
    parser.add_argument(
        "-u",
        "--utility",
        metavar="OPTION",
        default=None,
        help="the following options are available:\n"
        + "update     - store run data in <test/run> to <test/ref>\n"
        + "passed     - force tests to pass\n"
        + "check      - only check if tests passed or not\n"
        + "evaluate   - only evaluate available results against reference",
    )
    parser.add_argument(
        "-r", "--run", action="store_true", help="run the tests"
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
        default=str(pathlib.Path(__file__).resolve().parent.parent),
        help="diretories that contain the functions (e.g. stsg_ssg.py, mssp.py, etc.)",
    )
    parser.add_argument('-roc', '--ROC', default=False,
        help='Perform robust outlier cutoff (ROC) based on the maximum' +
                'absolute deviation and the universal threshold (True/False).')
    args = vars(parser.parse_args())
    start_time = time.time()
    errors = 0
    config = {}
    config['ROC'] = args["ROC"]
    config['bin'] = args["bin"]
    config['nthreads'] = str(args["nthreads"])

    runner = TestRunner(config)
    # and the tests
    for target_dir in args["tests"]:
        if not pathlib.Path(target_dir).is_dir():
            PRINTERRORANDEXIT("directory " + target_dir + " does not exist")
        runner.findTests(pathlib.Path(target_dir))
    # do some checks
    if args["utility"] and args["run"]:
        PRINTERRORANDEXIT(
            "calling 'utility' and 'run' at the same time is not possible"
        )
    if args["utility"] == "update" and runner.getNumTests() > 1:
        PRINTERRORANDEXIT("can only update ONE test at the same time")

    # eventually clean up first
    if args["Clean"]:
        runner.cleanUp()
    elif args["clean"]:
        runner.cleanUp("soft")
    # then run specific utility jobs
    if args["utility"] == "check":
        runner.getPassed()
    elif args["utility"] == "evaluate":
        runner.evaluate_only()
    elif args["utility"] == "update":
        runner.update()
    elif args["utility"] == "passed":
        for testdir in args["tests"]:
            runner.writePassed(pathlib.Path(testdir))
    else:
        runner.displayNumTests()
        if args["run"]:
            errors = runner.run()

    end_time = time.time()
    PRINTTITLE(
        "runtests.py was running for %d seconds" % int(end_time - start_time), "-"
    )

    # return success (or not: number of failed tests)
    sys.exit(-errors)