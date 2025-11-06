import os
import numpy as np
import pandas as pd
import bohydra as bo

from simulation_utils import run_path, launch_simulation, process_simulation


# SET PARAMETER BOUNDS AND NAMES

lower_bound = np.array([100.,   0.,   0.,  10.])
upper_bound = np.array([300., 100., 100., 200.])
    param_names = ["Param Name 1", "Param Name 2", "Param Name 3", "Param Name 4"]
 
n_params = lower_bound.size


data_df = pd.read_csv(run_path+"running_cases.csv", index_col=0)
run_nums = list(data_df.index)

y  = np.zeros(data_df.shape[0])
x  = data_df[param_names].values

for ii, run_num in enumerate(run_nums):
    y[ii] = process_simulation(run_num)

data_df["target"] = y
data_df.to_csv(run_path+"finished_cases.csv")

print("Start Optimization")

valid_mask   = np.logical_not(np.isinf(y))
invalid_mask = np.isinf(y)

valid_x   = x[valid_mask,:] 
valid_y   = y[valid_mask]
invalid_x = x[invalid_mask,:]
invalid_y = y[invalid_mask]
invalid_n = np.sum(invalid_mask)

x_I = np.vstack([valid_x, invalid_x]) 
y_I = np.concatenate([valid_y, invalid_y])

data_dict = {"x":    valid_x,
             "y":    valid_y,
             "nugget": 1.e-6}


# LAUNCH FIRST 10 OPTIMIZATION STEPS IN PARALLEL

n_parallel = 10
for ii in range(n_parallel):
    simulation_opt = bo.Opt(process_simulation, data_dict, emulator_type = "GP")
    if x_I.shape[0] > valid_x.shape[0]:
        simulation_opt.emulator.add_impute_data(x_I, y_I)
    candidate = simulation_opt.find_candidate()
    run_num = launch_simulation(candidate)
    run_num = ii + 10
    x_I = np.vstack([x_I, candidate])
    y_I = np.concatenate([y_I, [0.]])

    if ii == 0:
        running_df = pd.DataFrame(candidate[None,:], index=[run_num], columns=param_names)
    else:
        running_df = pd.concat([running_df, pd.DataFrame(candidate[None,:], index=[run_num], columns=param_names)])
    running_df.to_csv(run_path+"running_cases.csv")

