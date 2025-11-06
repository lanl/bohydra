import numpy as np
import bohydra as bo


def test_function(x):
    """
    Objective: f(x) = -x0^2 + exp(-||x - 0.5||^2) over x in [-1, 1]^2.
    Returns a 1D array of objective values for each row of x.
    """
    x = np.atleast_2d(x)
    return -x[:, 0] ** 2 + np.exp(-np.sum((x - 0.5) ** 2, axis=1))


def constraint_function(x):
    """
    Inequality constraint: x0 <= 0 (feasible region is left half-plane).
    Returns the left-hand side g(x) with feasibility defined as g(x) <= 0.
    """
    x = np.atleast_2d(x)
    return x[:, 0]


def main():
    rng = np.random.default_rng(234)

    # Problem domain and initial design
    x_lower = np.array([-1.0, -1.0])
    x_upper = np.array([1.0, 1.0])
    x0 = rng.uniform(x_lower, x_upper, size=(10, 2))
    y0 = test_function(x0)

    print("Start Single Fidelity Emulator - Single Fidelity Constrained Optimization")

    data_dict = {"x": x0, "y": y0, "nugget": 1e-4}
    const_dicts = [
        {
            "x": x0,
            "y": constraint_function(x0),
            "value": 0.0,
            "sign": "lessThan",
            "nugget": 1e-8,
        }
    ]

    const_opt = bo.ConstrainedOpt(
        func=test_function,
        data_dict=data_dict,
        constraint_dicts=const_dicts,
        emulator_type="GP",
        nugget=1e-4,
        x_lower=x_lower,
        x_upper=x_upper,
        random_state=0,
        constraint_weight=1.0,
    )

    for _ in range(15):
        const_opt.run_opt(iterations=1)
        # Identify current best feasible (x0 <= 0)
        feas = const_opt.emulator.x[:, 0] <= 0.0
        if np.any(feas):
            feas_idx = np.arange(const_opt.emulator.y.size)[feas]
            idx_best_local = np.argmax(const_opt.emulator.y[feas])
            idx_best = feas_idx[idx_best_local]
            print(
                "new:", const_opt.emulator.x[-1, :], const_opt.emulator.y[-1],
                "best_feas:", const_opt.emulator.x[idx_best, :], const_opt.emulator.y[idx_best],
            )
        else:
            print("new:", const_opt.emulator.x[-1, :], const_opt.emulator.y[-1], "no feasible point yet")


if __name__ == "__main__":
    main()
