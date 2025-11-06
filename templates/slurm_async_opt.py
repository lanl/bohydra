import os
import time
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import multifidelity_opt as mf

# USER CONFIGURATION (edit these for your environment)
base_path = '/path/to/run/directory/'  # absolute or resolved path to the working directory
template_dir = 'template_run'          # directory under base_path containing template files
input_deck = 'deck.input'              # input deck filename inside the run directory
slurm_script = 'slurm-script.sh'            # Slurm submission script inside the run directory

# SET PARAMETER BOUNDS AND NAMES
n_init = 20
n_parallel = 10
n_total = 500

lower_bound = np.array([100.0, 0.0, 0.0, 10.0])
upper_bound = np.array([300.0, 100.0, 100.0, 200.0])
param_names = ['Parameter 0', 'Parameter 1', 'Parameter 2', 'Parameter 3']


def _resolve_paths():
    base = Path(base_path)
    tdir = base / template_dir
    indeck_src = tdir / input_deck
    if not base.exists():
        raise FileNotFoundError(f'base_path does not exist: {base}')
    if not tdir.exists():
        raise FileNotFoundError(f'template_dir not found under base_path: {tdir}')
    if not indeck_src.exists():
        raise FileNotFoundError(f'input_deck not found in template_dir: {indeck_src}')
    return base, tdir, indeck_src


def launch_simulation(param_vec):
    """
    Launch a run based on the input parameter vector and template files.

    - Copies the template_dir into a new run directory run_XXX under base_path.
    - Replaces PARAM_LABEL{i} markers in the input deck with values from param_vec.
    - Submits the Slurm job via sbatch and returns the run number (zero-based).

    Returns None on submission failure.
    """
    base, tdir, indeck_src = _resolve_paths()

    # Determine new run number (simple template approach)
    run_num = len([d for d in base.iterdir() if d.is_dir() and d.name.startswith('run_')])
    run_dir = base / f'run_{run_num:03d}'

    # Copy template tree into new run dir
    shutil.copytree(tdir, run_dir)

    # Replace parameter labels in the input deck
    labels = [f'PARAM_LABEL{i}' for i in range(len(param_vec))]
    template_input = indeck_src.read_text().splitlines(True)
    indeck_dst = run_dir / input_deck
    with indeck_dst.open('w') as writer:
        for fline in template_input:
            wrote = False
            for i, lab in enumerate(labels):
                if lab in fline:
                    writer.write(fline.replace(lab, str(param_vec[i])))
                    wrote = True
                    break
            if not wrote:
                writer.write(fline)

    # Submit job from within run_dir
    try:
        subprocess.run(['sbatch', slurm_script], cwd=run_dir, env=os.environ, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f'Submission failed for {run_dir}: {e}')
        return None

    return run_num


def process_simulation(run_num, poll_interval=10):
    """
    Block until run completes, then parse and return target metric.
    Returns np.inf on failure (invalid run).
    """
    base, _, _ = _resolve_paths()
    run_dir = base / f'run_{run_num:03d}'

    while not (run_dir / 'RUN-COMPLETE').exists():
        time.sleep(poll_interval)

    target_metric = np.inf  # sentinel for failure by default
    try:
        # TODO: Replace with code to parse outputs in run_dir and compute target_metric
        # target_metric = parse_outputs(run_dir)
        pass
    except Exception as e:
        print(f'READING OUTPUT FAILED for run {run_num}: {e}')
    return float(target_metric)


def process_simulation_async(run_nums, max_checks=120, poll_interval=60):
    """
    Watch a batch of runs and return the (target_metric, run_num) for the first one that completes.
    On timeout, returns (np.inf, None).
    """
    base, _, _ = _resolve_paths()
    for _ in range(max_checks):
        for run_num in run_nums:
            run_dir = base / f'run_{run_num:03d}'
            if (run_dir / 'RUN-COMPLETE').exists():
                target_metric = np.inf
                try:
                    # TODO: Replace with code to parse outputs in run_dir and compute target_metric
                    # target_metric = parse_outputs(run_dir)
                    pass
                except Exception as e:
                    print(f'READING OUTPUT FAILED for run {run_num}: {e}')
                return float(target_metric), run_num
        time.sleep(poll_interval)
    return float(np.inf), None


def get_target(param_vec, _job_id=None):
    run_id = launch_simulation(param_vec)
    if run_id is None:
        return float(np.inf)
    return process_simulation(run_id)


def main():
    base, _, _ = _resolve_paths()

    n_params = lower_bound.size

    # Deterministic initial design using Latin Hypercube
    sampler = qmc.LatinHypercube(d=n_params, seed=0)
    x_unit = sampler.random(n=n_init)
    x = lower_bound + x_unit * (upper_bound - lower_bound)

    y = np.zeros(n_init, dtype=float)

    print('Launching initial cases')
    run_nums = [launch_simulation(x[ii, :]) for ii in range(n_init)]

    for ii, run_num in enumerate(run_nums):
        if run_num is None:
            y[ii] = np.inf
        else:
            y[ii] = process_simulation(run_num)

    # Canonical ledger for all cases
    data_df = pd.DataFrame(x, columns=param_names)
    data_df['Target'] = y
    ledger_path = base / 'finished_cases.csv'
    data_df.to_csv(ledger_path, index=False)

    print('Start Optimization')

    valid_mask = np.isfinite(y)
    invalid_mask = ~valid_mask

    valid_x = x[valid_mask, :]
    valid_y = y[valid_mask]
    invalid_x = x[invalid_mask, :]
    invalid_y = y[invalid_mask]

    # Prepare optimizer with valid data only
    data_dict = {'x': valid_x, 'y': valid_y, 'nugget': 1.0e-4}
    simulation_opt = mf.Opt(get_target, data_dict, emulator_type='GP', random_state=0)

    running_cases = []
    running_cands = []

    # Imputation helper
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
        simulation_opt.emulator.add_impute_data(X_I, Y_I)

    # LAUNCH FIRST n_parallel OPTIMIZATION STEPS IN PARALLEL
    to_launch = max(0, min(n_parallel, n_total - n_init))
    for _ in range(to_launch):
        if invalid_x.size or running_cands:
            update_impute()
        candidate = simulation_opt.find_candidate()
        job_id = launch_simulation(candidate)
        if job_id is not None:
            running_cases.append(job_id)
            running_cands.append(candidate)

    # RUN NEXT CASES ASYNCHRONOUSLY
    next_case = n_init + len(running_cases)
    for _ in range(max(0, n_total - next_case)):
        new_target, run_num = process_simulation_async(running_cases)
        if run_num is None:
            # timeout; skip launching replacement and continue
            continue
        try:
            case_ind = running_cases.index(run_num)
        except ValueError:
            continue
        candidate = running_cands.pop(case_ind)
        running_cases.pop(case_ind)

        if not np.isfinite(new_target):
            invalid_x = np.vstack([invalid_x, candidate]) if invalid_x.size else candidate[None, :]
            invalid_y = np.hstack([invalid_y, np.inf]) if invalid_y.size else np.array([np.inf])
        else:
            simulation_opt.data['x'] = np.vstack([simulation_opt.emulator.x, candidate])
            simulation_opt.data['y'] = np.concatenate([simulation_opt.emulator.y, [new_target]])
            simulation_opt.emulator = mf.initialize_emulator(simulation_opt.emu_type, simulation_opt.data)

        # Update ledger
        new_df = pd.DataFrame([dict(zip(param_names, candidate.tolist())) | {'Target': new_target}])
        data_df = pd.concat([data_df, new_df], ignore_index=True)
        data_df.to_csv(ledger_path, index=False)

        if invalid_x.size or running_cands:
            update_impute()
        candidate = simulation_opt.find_candidate()
        job_id = launch_simulation(candidate)
        if job_id is not None:
            running_cases.append(job_id)
            running_cands.append(candidate)

    # READ THE LAST RUNNING CASES (drain)
    while running_cases:
        new_target, run_num = process_simulation_async(running_cases)
        if run_num is None:
            break
        try:
            case_ind = running_cases.index(run_num)
        except ValueError:
            continue
        candidate = running_cands.pop(case_ind)
        running_cases.pop(case_ind)

        new_df = pd.DataFrame([dict(zip(param_names, candidate.tolist())) | {'Target': new_target}])
        data_df = pd.concat([data_df, new_df], ignore_index=True)
        data_df.to_csv(ledger_path, index=False)


if __name__ == '__main__':
    main()
