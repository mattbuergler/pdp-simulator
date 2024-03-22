#!/usr/bin/env python3

"""
    Filename: stsg_ssg.py
    Authors: Matthias Bürgler, Daniel Valero, Benjamin Hohermuth, David F. Vetsch, Robert M. Boes
    Date created: January 1, 2024
    Description:

        Stochastic Time Series Generator and Synthetic Signal Generator (STSG-SSG)
"""

# (c) 2024 ETH Zurich, Matthias Bürgler, Daniel Valero,
# Benjamin Hohermuth, David F. Vetsch, Robert M. Boes,
# D-BAUG, Laboratory of Hydraulics, Hydrology and Glaciology (VAW)
# This software is released under the the GNU General Public License v3.0.
# https://https://opensource.org/license/gpl-3-0

import sys
import argparse
import pathlib
import json
import jsonschema
import numpy as np
import pandas as pd
import time
import matplotlib as plt

try:
    import stsg_ssg_functions
    from tools.globals import *
except ImportError:
    print("")
    raise



def main():
    """
        Stochastic Time Series Generator and Synthetic Signal Generator (STSG-SSG)

        In this function, the configuration JSON-File is parsed, the
        stochastic time series are generated and the synthetic signal
        is generated. The signal time series are then written to the output
        file.
    """

    # Create parser to read in the configuration JSON-file to read from
    # the command line interface (CLI)

    parser = argparse.ArgumentParser(
        description="""\
        Stochastic Time Series Generator and Synthetic Signal Generator (STSG-SSG)

        The flow properties and probe-characteristics are passed via
        the input_file (JSON). The configuration JSON-File is parsed, the
        stochastic time series are generated and the synthetic signal
        is generated. The signal time series are then written to the output
        file.

        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('path', type=str,
        help="The path to the scenario directory.")
    parser.add_argument(
        "-r",
        "--run",
        metavar="COMMAND",
        default=None,
        help="the following commands are available:\n"
        + "timeseries - create only the velocity time series\n"
        + "signal     - create only the probe signal from existing velocity time series\n"
        + "all        - create velocity time series, and also the probe signal\n"
    )
    parser.add_argument('-p', '--progress', action='store_true',
        help='Show progress bar.')
    parser.add_argument('-c', '--compressed_signal', default='False',
        help='Expect a compressed signal (True/False).')
    parser.add_argument(
        "-n",
        "--nthreads",
        metavar="NTHREADS",
        default=1,
        help="set the number of threads for parallel execution",
    )
    args = parser.parse_args()

    printHeader()

    # Create Posix path for OS indepency
    path = pathlib.Path(args.path)
    input_file = path / 'config.json'
    # Read the configuation JSON-file
    config = json.loads(input_file.read_bytes())
    # Load the schema
    schema_file = pathlib.Path(sys.argv[0]).parents[0] / 'schemadef' / 'config_schema.json'
    schema = json.loads(schema_file.read_bytes())
    # Validate the configuration file
    jsonschema.validate(instance=config, schema=schema)

    command = args.run
    if command in ["timeseries", 'all']:
        print('Stochastic Time Series Generator (STSG)\n')
        # Generate the stochastic velocity and trajectory time series
        stsg_ssg_functions.SBG_fluid_velocity(
            path=path,
            flow_properties=config['FLOW_PROPERTIES'],
            reproducibility=config['REPRODUCIBILITY'],
            progress=args.progress
        )
    if command in ["signal", 'all']:
        print('Synthetic Signal Generator (SSG)\n')
        if 'UNCERTAINTY_QUANTIFICATION' not in config:
            config['UNCERTAINTY_QUANTIFICATION'] = {}
        # Generate the probe signal
        stsg_ssg_functions.SBG_signal(
            path=path,
            flow_properties=config['FLOW_PROPERTIES'],
            probe=config['PROBE'],
            reproducibility=config['REPRODUCIBILITY'],
            uncertainty=config["UNCERTAINTY_QUANTIFICATION"],
            progress=args.progress,
            nthreads=int(args.nthreads),
            compressed_signal=args.compressed_signal
            )

if __name__ == "__main__":
    main()
