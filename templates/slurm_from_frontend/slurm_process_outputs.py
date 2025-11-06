from pathlib import Path
import numpy as np
import pandas as pd

from .slurm_async_opt import (
    process_simulation,
    base_path,
    input_deck,
    lower_bound,
    param_names,
)

n_params = lower_bound.size
n_init = 10
LEDGER = Path(base_path) / "finished_init.csv"


def process_inputs(run_num: int) -> np.ndarray:
    """
    Parse the input deck for run_{run_num:03d} and return parameter vector.

    NOTE: This is a template. Replace placeholder string matching with your deck’s actual format.
    """
    run_dir = Path(base_path) / f"run_{run_num:03d}"
    deck_path = run_dir / input_deck
    if not deck_path.exists():
        raise FileNotFoundError(f"Input deck not found: {deck_path}")
    lines = deck_path.read_text().splitlines()

    p_vec = np.zeros(n_params, dtype=float)
    for line in lines:
        # Example placeholders; adapt to your deck format and add more parameters as needed
        if "PARAM_LABEL0" in line and n_params > 0:
            try:
                p_vec[0] = float(line.split()[-1])
            except Exception:
                pass
        if "PARAM_LABEL1" in line and n_params > 1:
            try:
                p_vec[1] = float(line.split()[-1])
            except Exception:
                pass
        if "PARAM_LABEL2" in line and n_params > 2:
            try:
                # Example transformation; modify as needed for your deck
                p_vec[2] = 1000.0 - 1.0e3 * float(line.split()[-1])
            except Exception:
                pass
        # Extend for further parameters as needed...

    return p_vec


def main():
    run_nums = list(range(n_init))

    xs = [process_inputs(i) for i in run_nums]
    ys = [process_simulation(i) for i in run_nums]

    X = np.vstack(xs)
    y = np.array(ys, dtype=float)

    df = pd.DataFrame(X, columns=param_names, index=run_nums)
    df["Target"] = y
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LEDGER, index=True)


if __name__ == "__main__":
    main()
