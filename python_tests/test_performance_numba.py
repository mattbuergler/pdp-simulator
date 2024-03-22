#!/usr/bin/env python3

import numpy as np
import pandas as pd
import bisect
import random
import time
from numba import jit

def interpolate_time_series_old(t_old, value_old, t_new):
    result = np.interp(t_new, t_old, value_old)
    return result

N = 1001
def run_normal(n):
    for ii in range(0,n):
        t_p = np.linspace(0,1000,1001)
        x_p = np.linspace(0,1000,1001)
        t_p_new = np.linspace(0,1000,N)
        x_p_new = interpolate_time_series_old(t_p, x_p, t_p_new)

@jit(nopython=True)
def interpolate_time_series(t_old, value_old, t_new):
    result = np.interp(t_new, t_old, value_old)
    return result

@jit(nopython=True)
def run_numba(n):
    for ii in range(0,n):
        t_p = np.linspace(0,1000,1001)
        x_p = np.linspace(0,1000,1001)
        t_p_new = np.linspace(0,1000,N)
        x_p_new = interpolate_time_series(t_p, x_p, t_p_new)



n = 10000

time1 = time.time()

run_normal(n)

time2 = time.time()
run_numba(1)
time3 = time.time()
run_numba(n)
time4 = time.time()

print(f'interpolate_time_series_old: {time2-time1}s')
print(f'interpolate_time_series: {time4-time3}s')
