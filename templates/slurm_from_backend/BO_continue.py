import os
import numpy as np
import pandas as pd

import bohydra as bo

# Defaulting to importing wrappers for launching and processing simulations 
#     from an external util file.
from simulation_utils import run_path, launch_simulation, process_simulation

def main():
    # Parameters and bounds for problem
    n_max = 200
    lower_bound = np.array([100.,   0.,   0.,  10.])
    upper_bound = np.array([300., 100., 100., 200.])
    param_names = ["Param Name 1", "Param Name 2", "Param Name 3", "Param Name 4"]
     
    n_params = lower_bound.size
    
    # Get the result of the current case
    #     Incorporate into data structures
    run_num   = int(os.getcwd().split("run_")[1])
    new_output = process_simulation(run_num)


    # Read in current running and finished cases
    #     Parse into data structures for BO
    running_df = pd.read_csv(run_path+"running_cases.csv",  index_col=0)
    data_df    = pd.read_csv(run_path+"finished_cases.csv", index_col=0)

    print(f"Initial Running:\n {running_df}")
    print(f"Initial Finished:\n {data_df}")
     
    x = data_df[param_names].values
    y = data_df["target"].values
    
    valid_mask   = np.logical_not(np.isinf(y))
    invalid_mask = np.isinf(y)
    
    valid_x   = x[valid_mask,:] 
    valid_y   = y[valid_mask]
    invalid_x = x[invalid_mask,:]
    invalid_y = y[invalid_mask]
    invalid_n = np.sum(invalid_mask)
    
    x_I = np.vstack([valid_x, invalid_x]) 
    y_I = np.concatenate([valid_y, invalid_y])
    
    x_I = np.vstack([x_I, running_df[param_names].values])
    y_I = np.concatenate([y_I, [0.]*running_df.shape[0]])
    
    data_dict = {"x":    valid_x,
                 "y":    valid_y,
                 "nugget": 1.e-4}
    
    
    new_df    = running_df.loc[[run_num],:]
    candidate = new_df[param_names].values
    
    running_df.drop(run_num, axis=0, inplace=True)
    
    if new_output == 0:
        invalid_x  = np.vstack([invalid_x, candidate])
        invalid_y  = np.concatenate([invalid_y, [0.]])
        invalid_n += 1
    else:
        data_dict["x"] = np.vstack([data_dict["x"], candidate])
        data_dict["y"] = np.concatenate([data_dict["y"], [new_output]])
    
    # Save out new data frame of completed cases
    new_df["target"] = new_output
    data_df         = pd.concat([data_df, new_df])
    print(f"Updated Running After Processing:\n {running_df}")
    print(f"Updated Finished After Processing:\n {data_df}")
    data_df.to_csv(run_path+"finished_cases.csv")
    running_df.to_csv(run_path+"running_cases.csv")
    
    
    # Check to see if we have the last cases running, or 
    if (data_df.shape[0] + running_df.shape[0]) < n_max:
        # Build optimization object to select new case and launch that case
        simulation_opt = bo.Opt(process_simulation, data_dict, emulator_type = "GP")
        
        simulation_opt.emulator.add_impute_data(x_I, y_I)
        candidate = simulation_opt.find_candidate()
        new_run_num = launch_simulation(candidate)
        
        # Update csv to list the currently running cases and complete
        new_running = pd.DataFrame(candidate[None,:], columns = param_names, index = [new_run_num])

        running_df = pd.read_csv(run_path+"running_cases.csv",index_col=0)
        running_df = pd.concat([running_df, new_running]) 
        running_df.to_csv(run_path+"running_cases.csv")
        print(f"Updated Running After Launch:\n {running_df}")
        print(f"Updated Finished After Launch:\n {data_df}")


# Check to see if the run completed. 
#    If not, this is being called before a restart and we dont 
#    want to process yet.
if "dshell-DO_NOT_RUN" in os.listdir(os.getcwd()):
    main()

