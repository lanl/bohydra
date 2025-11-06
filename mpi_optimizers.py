import numpy    as np
import pandas   as pd
import pyDOE    as doe

from .optimizers import Opt

has_mpi = True
try:
    from mpi4py      import MPI
except:
    has_mpi = False
    print("mpi4py not installed or configured properly. Functions that require MPI functionality will require this to be fixed to operate.")

def make_data_df(case_ind, param, y, param_names):
    run_df = pd.DataFrame( {"Run Number": case_ind, "target": y}, index = [case_ind])

    for ii in range( len(param_names) ):
        run_df[param_names[ii]] = param[ii]
    return run_df 


def bo_coordinator(n_total, n_init, n_params, bounds, param_names=None):
    if not has_mpi:
        print("THIS FUNCTION REQUIRES A WORKING MPI4PY.")
        return 0
    comm      = MPI.COMM_WORLD
    n_workers = comm.Get_size() - 1

    if param_names is None:
        param_names = ["parameter" + str(i) for i in range(n_params)]

    design   = doe.lhs(n_params, samples=n_init, criterion='maximin')
    for ii in range(n_params):
        design[:, ii] = design[:, ii] * (bounds[ii][1] - bounds[ii][0]) + bounds[ii][0]

    run_list = []

    if n_init > n_workers:
        for worker in range(n_workers):
            comm.send(design[worker,:], dest = worker+1, tag=worker)
    
        next_case = n_workers
        
        for worker in range(n_workers, n_init):
            status   = MPI.Status()
            param, y = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
            case_ind = status.Get_tag()
            source   = status.Get_source()

            print( "Got ", case_ind, " at ", param, "with value ", y)

            run_list.append( make_data_df(case_ind, param, y, param_names) )
            comm.send(design[next_case,:], dest = source, tag=next_case)
            next_case += 1
    
        for last_inits in range(n_init - n_workers, n_init):
            status   = MPI.Status()
            param, y = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
            case_ind = status.Get_tag()
            source   = status.Get_source()

            print( "Got ", case_ind, " at ", param, "with value ", y)

            run_list.append( make_data_df(case_ind, param, y, param_names) )

    else:
        for worker in range(n_init):
            comm.send(design[worker,:], dest = worker+1, tag=worker)

        next_case = n_init

        for last_inits in range(n_init):
            status   = MPI.Status()
            param, y = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
            case_ind = status.Get_tag()
            source   = status.Get_source()

            print( "Got ", case_ind, " at ", param, "with value ", y)

            run_list.append( make_data_df(case_ind, param, y, param_names) )

    data_df = pd.concat(run_list)
    data_df["Predicted Mean"]  = np.nan
    data_df["Predicted Upper"] = np.nan
    data_df["Predicted Lower"] = np.nan
    
    data_df.to_csv("finished_cases.csv")
    
    # TODO: THIS FILTERING IS PROBLEM SPECIFIC. ADD A FILTERING FUNCTION FOR INVALID CASES
    #
    # if filter_condition is not None:
    #     data_df.loc[filter_condition(data_df["target"]),"target"] = np.nan
    
    running_df = data_df.copy()
    running_df.drop(data_df.index, axis = 0, inplace=True)

    print("All initial runs completed.")

    ######### Launch initial runs at asyncronous stage
    for case_loop in range(n_init, n_init+n_workers):
        valid_inds   = data_df.index[ np.logical_not(data_df["target"].isna()) ]
        invalid_inds = data_df.index[ data_df["target"].isna() ]
        valid_x      = data_df.loc[valid_inds, param_names].values
        valid_y      = data_df.loc[valid_inds, "target"].values
        invalid_x    = data_df.loc[invalid_inds, param_names].values

        surrogate = Opt(lambda x:x, {"x":valid_x, "y":valid_y})

        if invalid_x.shape[0] > 1:
            impute_y = surrogate.emulator.predict(invalid_x)[0]

            train_x  = np.vstack([valid_x, invalid_x])
            train_y  = np.concatenate([valid_y, impute_y])
            surrogate.emulator.add_impute_data(train_x, train_y)
        else:
            train_x  = valid_x
            train_y  = valid_y

        if running_df.shape[0] > 0:
            fill_y  = surrogate.emulator.predict(running_df[param_names].values)[0]

            train_x = np.vstack([train_x, running_df[param_names].values])
            train_y = np.concatenate([train_y, fill_y])
            surrogate.emulator.add_impute_data(train_x, train_y)

        candidate     = surrogate.find_candidate().reshape([1,n_params])
        new_candidate = pd.DataFrame(candidate, index = [next_case], columns = param_names)

        pred_mean, pred_sd  = surrogate.emulator.predict(candidate)
        lower, upper        = (pred_mean - 1.96*pred_sd, pred_mean + 1.96*pred_sd)

        new_candidate["Predicted Mean"]  = pred_mean
        new_candidate["Predicted Lower"] = lower
        new_candidate["Predicted Upper"] = upper
        new_candidate["Run Number"]      = next_case

        running_df = pd.concat([running_df, new_candidate])
        data_df.loc[invalid_inds, "target"] = np.nan

        comm.send(candidate.flatten(), dest = case_loop - n_init + 1, tag = next_case)
        next_case += 1

    ######### Continue async optimization
    for case_loop in range(next_case, n_total):
        status   = MPI.Status()
        param, y = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
        case_ind = status.Get_tag()
        source   = status.Get_source()

        print( "Got ", case_ind, " at ", param, "with value ", y)

        new_data           = running_df.loc[[case_ind]].copy()
        new_data["target"] = y

        data_df = pd.concat([data_df, new_data])
        data_df.to_csv("finished_cases.csv")

        running_df.drop(case_ind, axis = 0, inplace = True)
        running_df.to_csv("running_cases.csv")

        if next_case < n_total:
            valid_inds   = data_df.index[ np.logical_not(data_df["target"].isna()) ]
            invalid_inds = data_df.index[ data_df["target"].isna() ]
            valid_x      = data_df.loc[valid_inds, param_names].values
            valid_y      = data_df.loc[valid_inds, "target"].values
            invalid_x    = data_df.loc[invalid_inds, param_names].values
    
            surrogate = Opt(lambda x:x, {"x":valid_x, "y":valid_y})

            if invalid_x.shape[0] > 1:
                impute_y = surrogate.emulator.predict(invalid_x)[0]

                train_x  = np.vstack([valid_x, invalid_x])
                train_y  = np.concatenate([valid_y, impute_y])
                surrogate.emulator.add_impute_data(train_x, train_y)
            else:
                train_x  = valid_x
                train_y  = valid_y

            if running_df.shape[0] > 0:
                fill_y  = surrogate.emulator.predict(running_df[param_names].values)[0]

                train_x = np.vstack([train_x, running_df[param_names].values])
                train_y = np.concatenate([train_y, fill_y])
                surrogate.emulator.add_impute_data(train_x, train_y)
    
            candidate     = surrogate.find_candidate().reshape([1,n_params])
            new_candidate = pd.DataFrame(candidate, index = [next_case], columns = param_names)
            
            pred_mean, pred_sd = surrogate.emulator.predict(candidate)
            lower, upper       = (pred_mean - 1.96*pred_sd, pred_mean + 1.96*pred_sd)

            new_candidate["Predicted Mean"]  = pred_mean
            new_candidate["Predicted Lower"] = lower
            new_candidate["Predicted Upper"] = upper
            new_candidate["Run Number"]      = next_case
             
            running_df = pd.concat([running_df, new_candidate])
            data_df.loc[invalid_inds, "target"] = np.nan
            
            comm.send(candidate.flatten(), dest = source, tag=next_case)
            next_case += 1

    ######### Listen for final runs to finish
    for case_loop in range(n_workers):
        status   = MPI.Status()
        param, y = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status=status)
        case_ind = status.Get_tag()
        source   = status.Get_source()

        print( "Got ", case_ind, " at ", param, "with value ", y)

        new_data           = running_df.loc[[case_ind]].copy()
        new_data["target"] = y

        data_df = pd.concat([data_df, new_data])
        data_df.to_csv("finished_cases.csv")

        running_df.drop(case_ind, axis = 0, inplace = True)
        running_df.to_csv("running_cases.csv")

    for worker in range(n_workers):
        comm.send("Done", dest = worker+1, tag = 0)
    comm.Barrier() 

    print("All Simulations Complete")
    return 1


def bo_worker(query_function):
    if not has_mpi:
        print("THIS FUNCTION REQUIRES A WORKING MPI4PY.")
        return 0
    comm      = MPI.COMM_WORLD
    rank      = comm.Get_rank()
    status    = MPI.Status()

    design   = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
    case_ind = status.Get_tag()

    while type(design) != type("Done"):
        target = query_function(design,case_ind)

        comm.send([design, target], dest = 0, tag = case_ind)

        status   = MPI.Status()
        design   = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        case_ind = status.Get_tag()
        print("Rank ", rank, " got message ", case_ind)
    comm.Barrier()
    return 1
