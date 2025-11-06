import numpy as np
import multifidelity_opt as mf


def hi(x):
    """
    High-fidelity objective over x in [-1, 1]^2:
    f_hi(x) = -x0^2 + exp(-||x - 0.5||^2)
    Returns a 1D array of values for each row of x.
    """
    x = np.atleast_2d(x)
    return -x[:, 0] ** 2 + np.exp(-np.sum((x - 0.5) ** 2, axis=1))


def low(x):
    """
    Low-fidelity surrogate: f_low(x) = -x0^2.
    """
    x = np.atleast_2d(x)
    return -x[:, 0] ** 2


def main():
    rng = np.random.default_rng(0)

    # Explicit domain bounds
    x_lower = np.array([-1.0, -1.0])
    x_upper = np.array([1.0, 1.0])

    # Reproducible initial designs
    x_low = rng.uniform(x_lower, x_upper, size=(50, 2))
    x_high = rng.uniform(x_lower, x_upper, size=(10, 2))
    y_low = low(x_low)
    y_high = hi(x_high)

    print("Start Multi-Fidelity Emulator - Multi-Fidelity Optimization")

    data_dict = {
        "x": x_high,
        "y": y_high,
        "x_low": x_low,
        "y_low": y_low,
        "nugget": 1e-4,
    }

    mf_opt = mf.OptMF(
        func_low=low,
        func_high=hi,
        data_dict=data_dict,
        emulator_type="MFGP",
        x_lower=x_lower,
        x_upper=x_upper,
        random_state=0,
    )

    # Fixed reference set reused across iterations
    x_ref = rng.uniform(x_lower, x_upper, size=(200, 2))

    for _ in range(10):
        mf_opt.run_opt(x_reference=x_ref, iterations=1, cost_ratio=0.5)
        # Best high-fidelity observed so far
        best_h = np.argmax(mf_opt.emulator.y)
        # Last evaluation info by fidelity
        last_fid = mf_opt.evaluated_fidelities[-1]
        if last_fid == "high":
            print(
                "last(high):",
                mf_opt.emulator.x[-1, :],
                mf_opt.emulator.y[-1],
                "best_high:",
                mf_opt.emulator.x[best_h, :],
                mf_opt.emulator.y[best_h],
                "fidelity:",
                last_fid,
            )
        else:
            print(
                "last(low):",
                mf_opt.emulator.x_low[-1, :],
                mf_opt.emulator.y_low[-1],
                "best_high:",
                mf_opt.emulator.x[best_h, :],
                mf_opt.emulator.y[best_h],
                "fidelity:",
                last_fid,
            )


if __name__ == "__main__":
    main()
