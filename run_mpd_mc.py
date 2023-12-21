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
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def _dir_is_model_run(dirname: str) -> bool:
    """Return whether a given directory is a model run.

    Args:
        dirname (str): Name of the directory to validate.

    Returns:
        bool: Whether the directory matches the file naming pattern for
            model run directories.
    """
    if not dirname.startswith('run_'):
        return False
    try:
        _ = int(dirname.split('_', maxsplit=1)[1])
    except ValueError:
        return False
    return True


def _run_as_task(task: str, path: str, cmd: Union[str, List[str]]) -> bool:
    """Run the given command as a sub-task of the given name.

    Args:
        task (str): The identifier of the task used for the log file.
        path (str): The target directory for the command.
        cmd (List[str]): The command and arguments to execute.

    Returns:
        bool: Whether an exception occurred. Should be False.
    """
    with open(os.path.join(path, f'{task}.log'), 'w') as log:
        process = subprocess.Popen(
            cmd, cwd=path, stdout=subprocess.PIPE,
            universal_newlines=True, stderr=subprocess.STDOUT)
        try:
            line: str
            for line in iter(process.stdout.readline, ''):  # type: ignore
                log.write(line)
                # Check for error messages
                if ('error' in line) | ('Error' in line) | ('ERROR' in line):
                    print(line.rstrip())
                    return True
        finally:
            process.stdout.close()  # type: ignore
    # Wait for subprocess completion and raise appropriate errors
    return_code = process.wait()
    return bool(return_code)

def writePassed(path: pathlib.Path):
    with open(str(path / "SUCCESS"), "w") as file:
        file.write("1")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_runs_path', '-m', type=str, default='',
        help='Path to the model runs.')
    parser.add_argument('--roc','-r', default=False, metavar='BOOL',
        help='Perform robust outlier cutoff (ROC) based on the maximum' +
                'absolute deviation and the universal threshold (True/False).')
    parser.add_argument(
        "-o", "--overwrite", action='store_true', help="overwrite existing data.",
    )
    args = parser.parse_args()
    model_runs_path = os.path.abspath(args.model_runs_path)
    roc = args.roc
    mpd_bin = str(pathlib.Path(__file__).resolve().parent)
    time0 = time.time()
    print(f'Scanning for model runs in directory {model_runs_path}')
    # get the batch_id
    batch_id = model_runs_path.split('_', maxsplit=1)[1]
    # Find all subfolders named "run_<int>"
    subfolders: List[str] = []
    for name in os.listdir(model_runs_path):
        path = os.path.join(model_runs_path, name)
        if not os.path.isdir(path) or not _dir_is_model_run(name):
            continue
        copy_tree(os.path.join(path, 'input'), os.path.join(path, 'run'))
        subfolders.append(name)
    subfolders = sorted(subfolders)
    print(f'{len(subfolders)} jobs found, starting batch process')
    failed = 0  # Number of failed runs
    for index, run_name in enumerate(subfolders):
        path = os.path.join(model_runs_path, run_name, 'run')
        if (not (pathlib.Path(path) / 'SUCCESS').exists()) | (args.overwrite):
            time1 = time.time()
            # delete SUCCESS file if it exists
            (pathlib.Path(path) / 'SUCCESS').unlink(missing_ok=True)
            run_id = int(run_name.split('_', maxsplit=1)[1])
            err: bool = False  # Flag for exceptions raised in subprocesses
            print(f'Running simulation {run_name} '
                  f'(job {index+1} of {len(subfolders)}):')
            # timeseries
            cmd: List[str] = [
                'python',
                os.path.join(mpd_bin, 'sbg.py'),
                '-r', 'timeseries',
                '-n', '1',
                '.']
            task: str = 'timeseries'  # Identifier for the current step in the job
            print(f'  Task 1 of 4: {task}')
            err = _run_as_task(task, path, cmd)
            # signal
            if not err:
                task = 'signal'
                cmd = [
                    'python',
                    os.path.join(mpd_bin, 'sbg.py'),
                    '-r', 'signal',
                    '-n', '1',
                    '.']
                print(f'  Task 2 of 4: {task}')
                err = _run_as_task(task, path, cmd)
            # reconstruction
            if not err:
                task = 'reconstruction'
                cmd = [
                    'python',
                    os.path.join(mpd_bin, 'mssrc.py'),
                    '-roc', str(roc),
                    '-n', '1',
                    '.']

                print(f'  Task 3 of 4: {task}')
                err = _run_as_task(task, path, cmd)
            # evaluation
            if not err:
                task = 'evaluation'
                cmd = [
                    'python',
                    os.path.join(mpd_bin, 'evaluate.py'),
                    '--id', str(run_id),
                    '--simulation_results_file', os.path.join(model_runs_path, f'simulation_results_{batch_id}.csv'),
                    '-o',
                    '.']
                print(f'  Task 4 of 4: {task}')
                err = _run_as_task(task, path, cmd)
            time2 = time.time()
            duration = time2-time1
            if err:
                print(f'ERROR: An exception occurred during job \'{run_name}\', '
                      f'refer to the logs in the run directory for details (Duration: {duration:.1f}s).')
                failed += 1
            else:
                writePassed(pathlib.Path(path))
                print(f'Successfully finished job \'{run_name}\' in {duration:.1f}s.')
        else:
            print(f'Simulation {run_name} is already finished. Skipping this one.')
    duration = time.time()-time0
    print(f'\nDone (duration: {duration/86400.0:.1f}d)!')
    if failed > 0:
        print(f'{failed} jobs failed, check the logs for details')
