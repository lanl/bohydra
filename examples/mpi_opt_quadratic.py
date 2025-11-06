from mpi4py import MPI
import numpy as np
import bohydra as bo


def get_target(x, _job_id=None):
    """
    2D negative quadratic for maximization: f(x) = -(x0^2 + x1^2).
    Global maximum at x = (0, 0) with value 0.
    """
    x = np.asarray(x, dtype=float)
    return -(x[0] ** 2 + x[1] ** 2)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()

    if commsize < 2:
        if rank == 0:
            print("This example requires at least 2 MPI ranks (1 coordinator + >=1 worker).")
        return

    if rank == 0:
        # If supported by your coordinator, set a seed for reproducibility
        bo.bo_coordinator(
            n_total=100,
            n_init=5,
            n_params=2,
            bounds=[[-5.0, 5.0], [-5.0, 5.0]],
            # random_state=0,  # uncomment if supported
        )
    else:
        bo.bo_worker(get_target)


if __name__ == "__main__":
    main()
