import os
import sys
import time
import subprocess

import numpy  as np
import pyDOE  as doe
import pandas as pd

# Defaulting to importing wrappers for launching and processing simulations 
#     from an external util file.
from simulation_utils import run_path, launch_simulation, process_simulation


# Script for launching an initial set of simulations to start the optimization process.
#     This does not actually use bohydra, but it generates the data to start the BO process.


# SET PARAMETER BOUNDS AND NAMES

lower_bound = np.array([100.,   0.,   0.,  10.])
upper_bound = np.array([300., 100., 100., 200.])
    param_names = ["Param Name 1", "Param Name 2", "Param Name 3", "Param Name 4"]
 
n_params = lower_bound.size
n_init   = 10


# GENERATE INITIAL PARAMETER VECTORS USING A LATIN HYPERCUBE
x = doe.lhs(n_params, samples=n_init, criterion='maximin')
for ii in range(n_params):
    x[:, ii] = x[:, ii] * (upper_bound[ii] - lower_bound[ii]) + lower_bound[ii]

y = np.zeros(n_init)

print("Launching initial cases")

run_nums = [launch_simulation(x[ii,:]) for ii in range(n_init)]
data_df = pd.DataFrame(x,columns = param_names, index = run_nums)
data_df.to_csv("running_cases.csv")

