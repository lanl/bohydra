import sys
sys.path.append("../../.")
import numpy as np
import pandas as pd
import multifidelity_opt as mf


def six_hump_camel(x):
    """
    2D Six-Hump Camel function, negated to frame as a maximization problem.
    Domain: x0 in [-3, 3], x1 in [-2, 2].
    Returns a 1D array for each row of x.
    """
    x = np.atleast_2d(x).astype(float)
    y = (4.0 - 2.1 * x[:, 0] ** 2 + x[:, 0] ** 4 / 3.0) * x[:, 0] ** 2
    y += x[:, 0] * x[:, 1]
    y += (-4.0 + 4.0 * x[:, 1] ** 2) * x[:, 1] ** 2
    return -y


def main():
    rng = np.random.default_rng(0)
    x_lower = np.array([-3.0, -2.0])
    x_upper = np.array([3.0, 2.0])

    n_init = 10
    X0 = np.column_stack(
        [
            rng.uniform(x_lower[0], x_upper[0], size=n_init),
            rng.uniform(x_lower[1], x_upper[1], size=n_init),
        ]
    )
    y0 = six_hump_camel(X0)

    data_dict = {"x": X0, "y": y0, "nugget": 1e-4}
    opt = mf.Opt(
        six_hump_camel,
        data_dict,
        emulator_type="GP",
        x_lower=x_lower,
        x_upper=x_upper,
        random_state=0,
    )

    # Build initial records
    rows = []
    for i in range(n_init):
        rows.append(
            {
                "Run Number": i,
                "parameter0": float(X0[i, 0]),
                "parameter1": float(X0[i, 1]),
                "target": float(y0[i]),
                "Predicted Mean": np.nan,
                "Predicted Lower": np.nan,
                "Predicted Upper": np.nan,
            }
        )

    # Iterative BO loop
    n_iter = 90
    for it in range(n_iter):
        x_cand = opt.find_candidate()  # shape (2,)
        mean, sd = opt.emulator.predict(x_cand[None, :])
        y_new = six_hump_camel(x_cand[None, :])

        # Update optimizer state in place
        opt.data["x"] = np.vstack([opt.emulator.x, x_cand])
        opt.data["y"] = np.hstack([opt.emulator.y, y_new])
        opt.emulator = mf.initialize_emulator(opt.emu_type, opt.data)

        rows.append(
            {
                "Run Number": n_init + it,
                "parameter0": float(x_cand[0]),
                "parameter1": float(x_cand[1]),
                "target": float(y_new[0]),
                "Predicted Mean": float(mean[0]),
                "Predicted Lower": float(mean[0] - 1.96 * sd[0]),
                "Predicted Upper": float(mean[0] + 1.96 * sd[0]),
            }
        )

    df = pd.DataFrame.from_records(rows)
    df.to_csv("finished_cases.csv", index=False)


if __name__ == "__main__":
    main()
