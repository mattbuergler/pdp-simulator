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
    import sbg_plot
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
    parser.add_argument('-v', '--visualize', action='store_true',
        help="Vizualize the results.", default=False)
    parser.add_argument('-p', '--progress', action='store_true',
        help='Show progress bar.')
    args = parser.parse_args()

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
        # Generate the stochastic velocity and trajectory time series
        sbg_functions.SBG_auto_corr(
            path=path,
            flow_properties=config['FLOW_PROPERTIES'],
            reproducible=config['REPRODUCIBLE'],
            progress=args.progress
        )
    if command in ["signal", 'all']:
        # Generate the probe signal
        sbg_functions.SBG_signal(
            path=path,
            flow_properties=config['FLOW_PROPERTIES'],
            probe=config['PROBE'],
            reproducible=config['REPRODUCIBLE'],
            progress=args.progress
            )

    # Plot results if necessary
    if args.visualize:
        sbg_plot.plot_results(
            path=path,
            flow_properties=config['FLOW_PROPERTIES']
            )

if __name__ == "__main__":
    main()
