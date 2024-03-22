"""Run BASEMENT for all models in the current working directory.

This version is largely identical to the Bash (.sh) and Batch (.bat)
versions, but uses Python as a cross-platform alternative. It also
features more thorough error checking and status reporting.
"""

import argparse
import os
import subprocess
import sys
from typing import List, Union
from distutils.dir_util import copy_tree
import pathlib
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def _dir_is_batch(dirname: str) -> bool:
    """Return whether a given directory is a batch.

    Args:
        dirname (str): Name of the directory to validate.

    Returns:
        bool: Whether the directory matches the file naming pattern for
            batches.
    """
    if not dirname.startswith('batch_'):
        return False
    try:
        _ = int(dirname.split('_', maxsplit=1)[1])
    except ValueError:
        return False
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--node_path', '-m', type=str, default='',
        help='Path to the node directory.')
    args = parser.parse_args()
    node_path = os.path.abspath(args.node_path)
    simulation_results_file = f'simulation_results.csv'
    simulation_results = pd.read_csv(pathlib.Path(node_path) / simulation_results_file, index_col=None)

    print(f'Scanning for batches in directory {node_path}')
    # Find all subfolders named "batch_<int>"
    subfolders: List[str] = []
    for name in os.listdir(node_path):
        path = os.path.join(node_path, name)
        if not os.path.isdir(path) or not _dir_is_batch(name):
            continue
        subfolders.append(name)
    subfolders = sorted(subfolders)
    print(f'{len(subfolders)} batches found.')
    for index, batch_name in enumerate(subfolders):
        batch_path = pathlib.Path(os.path.join(node_path, batch_name))
        batch_id = batch_name.split('_', maxsplit=1)[1]
        batch_simulation_results = pd.read_csv(batch_path / simulation_results_file.replace('.csv',f'_{batch_id}.csv'), index_col=None)
        simulation_results = pd.concat([simulation_results,batch_simulation_results]).reset_index(drop=True)
    simulation_results = simulation_results.sort_values(['id'])
    simulation_results.to_csv(pathlib.Path(node_path) / simulation_results_file, index=False,na_rep='nan')

    print('\nDone!')
