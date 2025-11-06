from mpi4py import MPI
import numpy as np
import bohydra as bo


def get_target(x):
    """
    User-provided objective function to be maximized by BO.

    Parameters
    - x: 1D array-like of shape (n_params,), the parameter vector proposed by the optimizer.

    Returns
    - float: objective value to maximize.

    Replace the body with your simulator or black-box evaluation. Include robust error handling and
    return a sentinel (e.g., np.nan or -np.inf) on failure if your coordinator supports it.
    """
    x = np.asarray(x, dtype=float)
    # Example placeholder: negative quadratic (max at 0)
    return float(-(x ** 2).sum())


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()

    if commsize < 2:
        if rank == 0:
            print("This template requires at least 2 MPI ranks (1 coordinator + >=1 worker).")
        return

    if rank == 0:
        # Define bounds explicitly: list of [low, high] for each parameter
        n_params = 8
        bounds = [[-5.0, 5.0]] * n_params
        bo.bo_coordinator(
            n_total=1000,
            n_init=50,
            n_params=n_params,
            bounds=bounds,
        )
    else:
        bo.bo_worker(get_target)


if __name__ == "__main__":
    main()
