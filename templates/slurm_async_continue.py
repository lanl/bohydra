import os
import numpy as np
import pandas as pd
import multifidelity_opt as mf

from .slurm_async_opt import (
    launch_simulation,
    process_simulation_async,
    get_target,
    base_path,
    n_parallel,
    n_total,
    lower_bound,
    upper_bound,
    param_names,
)


def _load_checkpoint():
    """Load checkpoint CSV. Prefer finished_cases.csv; fallback to finished_init.csv.
    Returns dataframe and resolved checkpoint path.
    """
    candidates = [
        os.path.join(base_path, "finished_cases.csv"),
        os.path.join(base_path, "finished_init.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df, path if os.path.basename(path) == "finished_cases.csv" else os.path.join(base_path, "finished_cases.csv")
    raise FileNotFoundError(
        f"No checkpoint found. Expected one of: {candidates}"
    )


def _ensure_columns(df):
    required = list(param_names) + ["Target"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Checkpoint missing required columns: {missing}")


def main():
    # Load checkpoint
    df, checkpoint_path = _load_checkpoint()
    _ensure_columns(df)

    # Extract current data
    x = df[param_names].values
    y = df["Target"].values

    valid_mask = np.isfinite(y)
    invalid_mask = ~valid_mask

    valid_x = x[valid_mask, :]
    valid_y = y[valid_mask]
    invalid_x = x[invalid_mask, :]
    invalid_y = y[invalid_mask]

    # Initialize optimizer with valid points only
    data_dict = {"x": valid_x, "y": valid_y, "nugget": 1e-4}
    simulation_opt = mf.Opt(get_target, data_dict, emulator_type="GP")

    running_ids = []
    running_cands = []

    # Helper to update imputed data (invalids + running as placeholders)
    def update_impute():
        x_list = [simulation_opt.emulator.x]
        y_list = [simulation_opt.emulator.y]
        if invalid_x.size:
            x_list.append(invalid_x)
            y_list.append(invalid_y)
        for cand in running_cands:
            x_list.append(cand[None, :])
            y_list.append(np.array([np.inf]))
        X_I = np.vstack(x_list)
        Y_I = np.concatenate(y_list)
        # Overwrite imputed data state each time
        simulation_opt.emulator.add_impute_data(X_I, Y_I)

    # Number already started/completed
    total_started = x.shape[0]

    # Bootstrap up to n_parallel jobs, without exceeding total budget
    n_to_launch = max(0, min(n_parallel, n_total - total_started))
    for _ in range(n_to_launch):
        if invalid_x.size or running_cands:
            update_impute()
        cand = simulation_opt.find_candidate()
        job_id = launch_simulation(cand)
        running_ids.append(job_id)
        running_cands.append(cand)
        total_started += 1

    # Main loop until reaching total budget
    while total_started < n_total:
        # Wait/process one finished job
        new_target, run_id = process_simulation_async(running_ids)
        try:
            idx = running_ids.index(run_id)
        except ValueError:
            # Unknown run id; skip
            idx = None
        if idx is not None:
            cand = running_cands.pop(idx)
            running_ids.pop(idx)
        else:
            # If we cannot match, skip update and continue
            cand = None

        if cand is not None:
            if np.isfinite(new_target):
                # Update optimizer with successful evaluation
                simulation_opt.data["x"] = np.vstack([simulation_opt.emulator.x, cand])
                simulation_opt.data["y"] = np.hstack([simulation_opt.emulator.y, new_target])
                simulation_opt.emulator = mf.initialize_emulator(
                    simulation_opt.emu_type, simulation_opt.data
                )
            else:
                # Track invalid result for imputation and record-keeping
                invalid_x = np.vstack([invalid_x, cand]) if invalid_x.size else cand[None, :]
                invalid_y = np.hstack([invalid_y, np.inf]) if invalid_y.size else np.array([np.inf])

            # Append to checkpoint dataframe and write
            new_row = pd.DataFrame(
                [dict(zip(param_names, cand.tolist())) | {"Target": new_target}]
            )
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(checkpoint_path, index=False)

        # Launch next if budget allows
        if total_started < n_total:
            if invalid_x.size or running_cands:
                update_impute()
            cand = simulation_opt.find_candidate()
            job_id = launch_simulation(cand)
            running_ids.append(job_id)
            running_cands.append(cand)
            total_started += 1

    # Drain remaining jobs
    while running_ids:
        new_target, run_id = process_simulation_async(running_ids)
        try:
            idx = running_ids.index(run_id)
        except ValueError:
            idx = None
        if idx is not None:
            cand = running_cands.pop(idx)
            running_ids.pop(idx)
            new_row = pd.DataFrame(
                [dict(zip(param_names, cand.tolist())) | {"Target": new_target}]
            )
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(checkpoint_path, index=False)


if __name__ == "__main__":
    main()
