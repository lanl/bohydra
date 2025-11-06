from mpi4py import MPI
import numpy as np
import bohydra as bo


def get_target(x, _job_id=None):
    """
    2D Rosenbrock function (negated for maximization).
    Standard domain: [-5, 10] x [-5, 10]. Global minimum at (1, 1) with value 0;
    we negate to maximize.
    """
    x = np.asarray(x, dtype=float)
    xi = x[:-1]
    xn = x[1:]
    y = np.sum(100.0 * (xn - xi ** 2) ** 2 + (xi - 1.0) ** 2)
    return -y


MESSAGE = "This example requires at least 2 MPI ranks (1 coordinator + >=1 worker)."


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()

    if commsize < 2:
        if rank == 0:
            print(MESSAGE)
        return

    if rank == 0:
        bo.bo_coordinator(
            n_total=100,
            n_init=10,
            n_params=2,
            bounds=[[-5.0, 10.0], [-5.0, 10.0]],
            # random_state=0,  # uncomment if supported for reproducibility
        )
    else:
        bo.bo_worker(get_target)


if __name__ == "__main__":
    main()
