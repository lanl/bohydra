import time
import numpy as np
import bohydra as bo


def hi(x):
    """
    High-fidelity objective over x in [-1, 1]^2:
    f_hi(x) = -x0^2 + 0.4 * exp(-||x - 0.25||^2 / 0.3^2)
    Returns a 1D array of values for each row of x.
    """
    x = np.atleast_2d(x)
    return -x[:, 0] ** 2 + 0.4 * np.exp(-np.sum((x - 0.25) ** 2, axis=1) / 0.3 ** 2)


def low(x):
    """
    Low-fidelity surrogate: f_low(x) = -x0^2.
    """
    x = np.atleast_2d(x)
    return -x[:, 0] ** 2


def main():
    rng = np.random.default_rng(0)
    x_lower = np.array([-1.0, -1.0])
    x_upper = np.array([1.0, 1.0])

    # Initial designs
    x_low = rng.uniform(x_lower, x_upper, size=(50, 2))
    x_high = rng.uniform(x_lower, x_upper, size=(10, 2))

    y_low = low(x_low)
    y_high = hi(x_high)

    print("Start Single-Fidelity Emulator - Single-Fidelity Optimization")

    t_start = time.time()
    data_dict = {"x": x_high, "y": y_high, "nugget": 1e-4}
    high_opt = bo.Opt(
        hi,
        data_dict,
        emulator_type="GP",
        x_lower=x_lower,
        x_upper=x_upper,
        random_state=0,
    )

    for _ in range(10):
        high_opt.run_opt(iterations=1)
        best_ind = np.argmax(high_opt.emulator.y)
        print(
            high_opt.emulator.x[-1, :],
            high_opt.emulator.y[-1],
            high_opt.emulator.x[best_ind, :],
            high_opt.emulator.y[best_ind],
        )
    print("Ended after time =", time.time() - t_start)

    print("Start Multi-Fidelity Emulator (as prior) - Single-Fidelity Optimization")
    t_start = time.time()

    low_dict = {"x": x_low, "y": y_low, "nugget": 1e-4}
    low_emu = bo.initialize_emulator("GP", low_dict)

    data_dict = {
        "x": x_high,
        "y": y_high,
        "prior_emu": low_emu,
        "nugget": 1e-4,
    }
    high_opt = bo.Opt(
        hi,
        data_dict,
        emulator_type="MFGP",
        x_lower=x_lower,
        x_upper=x_upper,
        random_state=0,
    )

    for _ in range(10):
        high_opt.run_opt(iterations=1)
        best_ind = np.argmax(high_opt.emulator.y)
        print(
            high_opt.emulator.x[-1, :],
            high_opt.emulator.y[-1],
            high_opt.emulator.x[best_ind, :],
            high_opt.emulator.y[best_ind],
        )
    print("Ended after time =", time.time() - t_start)


if __name__ == "__main__":
    main()
