"""
multifidelity_opt: Multifidelity Bayesian optimization with serial and MPI-enabled optimizers.

Primary public API:
- Emulators: EmuGP, EmuMF, initialize_emulator
- Optimizers: Opt (single-fidelity), OptMF (multi-fidelity), ConstrainedOpt
- Utilities: running_max
- MPI (optional): bo_worker, bo_coordinator

Backwards-compatibility aliases:
- gp_emu -> EmuGP, mf_emu -> EmuMF
- gp_opt/OptGP -> Opt, mf_opt -> OptMF
"""

from importlib import metadata as _metadata

# Version: try to read from package metadata; fallback if missing (e.g., during editable installs)
try:
    __version__ = _metadata.version("multifidelity_opt")
except Exception:
    __version__ = "0.0.0"

# Core public API
from .emulators import EmuGP, EmuMF, initialize_emulator
from .optimizers import Opt, OptMF, ConstrainedOpt
from .utils import running_max

# Optional MPI API: provide helpful stubs if mpi4py or mpi_optimizers is unavailable

def _missing_mpi(name: str, err: Exception):
    def _raise(*args, **kwargs):
        raise ImportError(
            f"{name} requires mpi4py and an MPI runtime. Install mpi4py and run under mpirun/mpiexec. "
            f"Original import error: {err}"
        )

    return _raise


try:
    from .mpi_optimizers import bo_worker, bo_coordinator
except Exception as _mpi_err:
    bo_worker = _missing_mpi("bo_worker", _mpi_err)
    bo_coordinator = _missing_mpi("bo_coordinator", _mpi_err)

# Backwards-compatibility aliases (consider deprecating in a future release)
_gp_emu = EmuGP  # internal to avoid mypy shadowing complaints in some setups
mf_emu = EmuMF
# Maintain external names expected by users
gp_emu = _gp_emu

_gp_opt = Opt
mf_opt = OptMF
OptGP = _gp_opt
gp_opt = _gp_opt

__all__ = [
    # Version
    "__version__",
    # Emulators
    "EmuGP",
    "EmuMF",
    "initialize_emulator",
    # Optimizers
    "Opt",
    "OptMF",
    "ConstrainedOpt",
    # Utilities
    "running_max",
    # MPI (optional)
    "bo_worker",
    "bo_coordinator",
    # Aliases
    "gp_emu",
    "mf_emu",
    "gp_opt",
    "OptGP",
    "mf_opt",
]
