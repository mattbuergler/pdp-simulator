#!/usr/bin/env python3

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
    import sbg_functions
    import velocity_tsa
    from tools.globals import *
except ImportError:
    print("")
    raise


"""
    Stochastic Bubble Generator (SBG)

    The user-defined parameters are passed via the input_file (JSON).
    A time series is generated and saved to a file.
"""
def main():
    """
        Main function of the Stochastic Bubble Generator (SBG)

        In this function, the configuration JSON-File is parsed, the
        stochastic time series are generated and the synthetic Signal
        is calculated. The signal time series are then written to the output
        file.
    """

    # Create parser to read in the configuration JSON-file to read from
    # the command line interface (CLI)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=str,
        help="The path to the scenario directory.")
    parser.add_argument(
        "-r",
        "--run",
        metavar="COMMAND",
        default=None,
        help="the following commands are available:\n"
        + "timeseries - create only the velocity time series and trajectory\n"
        + "signal     - create only the probe signal from existing velocity time series and trajectory\n"
        + "all        - create velocity time series, trajectory and also the probe signal\n"
    )
    parser.add_argument('-tsa', '--velocity_tsa', action='store_true',
        help="Run the velocity time series analysis.")
    parser.add_argument('-p', '--progress', action='store_true',
        help='Show progress bar.')
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
        sbg_functions.SBG_fluid_velocity(
            path=path,
            flow_properties=config['FLOW_PROPERTIES'],
            reproducible=config['REPRODUCIBLE'],
            progress=args.progress
        )
        sbg_functions.SBG_bubble_generator(
            path=path,
            flow_properties=config['FLOW_PROPERTIES'],
            probe=config['PROBE'],
            reproducible=config['REPRODUCIBLE'],
            progress=args.progress,
            nthreads=int(args.nthreads)
        )
    if command in ["signal", 'all']:
        print('Synthetic Signal Generator (SSG)\n')
        if 'UNCERTAINTY_QUANTIFICATION' not in config:
            config['UNCERTAINTY_QUANTIFICATION'] = {}
        # Generate the probe signal
        sbg_functions.SBG_signal(
            path=path,
            flow_properties=config['FLOW_PROPERTIES'],
            probe=config['PROBE'],
            reproducible=config['REPRODUCIBLE'],
            uncertainty=config["UNCERTAINTY_QUANTIFICATION"],
            progress=args.progress,
            nthreads=int(args.nthreads)
            )

    # Plot results if necessary
    if args.velocity_tsa:
        velocity_tsa.main(str(path))

if __name__ == "__main__":
    main()
