#!/usr/bin/env python3

import sys
import pathlib
import subprocess
import shutil
from typing import List

"""
    global variables and functions
"""
SMALLNUMBER = 1.e-12
LARGENUMBER= 1.e12
LARGENEGNUMBER=-1.e12

# printing stuff to stdout
def PRINTLOG(log, string):
    s = '* '
    if len(log)>0:
        s += log + ': '
    print(s + string)

def PRINTTITLE(string, char='*'):
    print('\n'+char*(len(string)+2))
    PRINTLOG('', string)
    print(char*(len(string)+2))

def PRINTWARNING(string):
    PRINTLOG('warning', string)

def PRINTERRORANDEXIT(string):
    PRINTLOG('error', string)
    sys.exit(2)

# print progress bar to command line
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def inverse_den(x):
    """
    Calculate inverse of a number.
    """
    if abs(x) < SMALLNUMBER:
        return 0.0
    else:
        return 1.0 / x

def copy_file_to(file: pathlib.Path, dir: pathlib.Path):
    """
    Copy file to directory or file.
    """
    shutil.copy(str(file), str(dir))

def delete_files_in_directory(directory: pathlib.Path):
    """
    Deletes all files in the directory.
    """
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()

def create_dir_if_not_exists(path: pathlib.Path):
    """
    Create a directory at path.
    Already existing directories are ignored.
    """
    path.mkdir(exist_ok=True)

def copy_files(from_dir: pathlib.Path, to_dir: pathlib.Path):
    """
    Copies all files from from_dir to to_dir.
    """
    for item in from_dir.iterdir():
        if item.is_file():
            shutil.copy(str(item), str(to_dir))

def delete_file_if_exists(path: pathlib.Path):
    """
    Removes the file.
    Not existing files are ignored.
    """
    if path.is_file():
        path.unlink()

def run_process(
    arguments: List[str], work_dir: pathlib.Path, capture_output: bool = True
):
    """
    Runs the process defined by the arguments from work_dir.
    Stderr is redirected to stdout and both are captured if capture_output is True.
    Throws an exception if the process failed.
    """
    print(work_dir)
    create_dir_if_not_exists(work_dir)
    if capture_output:
        return subprocess.run(
            arguments,
            check=True,
            cwd=str(work_dir),
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

    return subprocess.run(
        arguments, check=True, cwd=str(work_dir), universal_newlines=True
    )

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

def get_git_version() -> str:
    return subprocess.check_output(['git', 'describe', '--always','--long']).decode('ascii').strip()

def printHeader():
    print(f'MULTIPHADE v-{get_git_version()}\n')