from mpi4py import MPI
import numpy as np
import multifidelity_opt as mf


def get_target(x, _job_id=None):
    """
    2D Six-Hump Camel function (maximization form by negating the standard minimization target).
    Implementation from Surjanovic and Bingham Function Library
    Domain: x0 in [-3, 3], x1 in [-2, 2].
    """
    x = np.asarray(x, dtype=float)
    y = (4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3.0) * x[0] ** 2
    y += x[0] * x[1]
    y += (-4 + 4 * x[1] ** 2) * x[1] ** 2
    return -y  # negate to maximize


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    commsize = comm.Get_size()

    if commsize < 2:
        if rank == 0:
            print(
                "This example requires at least 2 MPI ranks (1 coordinator + >=1 worker)."
            )
        return

    if rank == 0:
        # If supported, pass random_state/seed to coordinator to make initial design deterministic
        mf.bo_coordinator(
            n_total=100,
            n_init=10,
            n_params=2,
            bounds=[[-3.0, 3.0], [-2.0, 2.0]],
            # random_state=0,  # uncomment if supported
        )
    else:
        mf.bo_worker(get_target)


if __name__ == "__main__":
    main()
