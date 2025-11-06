import numpy as np

from scipy.optimize import minimize
from scipy.stats import norm
from .emulators import EmuGP, EmuMF, EmuMFOld, initialize_emulator
from .utils import log_h


default_nugget = 1.0e-4
_EPS = 1.0e-12


class Opt:
    """Single-fidelity Bayesian optimization using GP-based emulators.

    Parameters
    - func: Callable mapping x -> y for objective evaluations.
    - data_dict: For GP: keys {"x", "y", (optional) "nugget"}.
    - x_lower, x_upper: Bounds arrays; if None, default to min/max of data_dict["x"].
    - emulator_type: "GP" or "MFGP" (for IECI computations against MF emulator).
    - nugget: Added to data_dict if not present and emulator_type uses GP.
    - random_state: Seed for reproducible random restarts and subsampling.
    """

    def __init__(
        self,
        func,
        data_dict,
        x_lower=None,
        x_upper=None,
        emulator_type="GP",
        nugget=default_nugget,
        random_state=None,
    ):
        self.func = func
        self.data = data_dict
        self.emu_type = emulator_type
        self.rng = np.random.default_rng(random_state)

        if "nugget" not in list(data_dict.keys()) and emulator_type in ["GP", "MFGP"]:
            data_dict["nugget"] = nugget

        self.emulator = initialize_emulator(emulator_type, data_dict)

        if x_lower is None:
            self.x_lower = np.min(data_dict["x"], 0)
        else:
            self.x_lower = x_lower

        if x_upper is None:
            self.x_upper = np.max(data_dict["x"], 0)
        else:
            self.x_upper = x_upper

    def _best_y(self):
        y_I = getattr(self.emulator, "y_I", None)
        return np.max(y_I) if y_I is not None else np.max(self.emulator.y)

    def logei(self, x_test, explore_discount=1.0):
        """Log of Expected Improvement for numerical stability.
        Note: explore_discount is ignored here; use ei() if you need explicit scaling.
        """
        f_max = self._best_y()
        test_mean, test_sd = self.emulator.predict(x_test[None, :])
        test_sd = np.maximum(test_sd, _EPS)
        imp = test_mean - f_max
        z = imp / test_sd
        return log_h(z) + np.log(test_sd)

    def ei(self, x_test, explore_discount=1.0):
        normrv = norm()
        f_max = self._best_y()
        test_mean, test_sd = self.emulator.predict(x_test[None, :])
        test_sd = np.maximum(test_sd, _EPS)
        imp = test_mean - f_max
        z = imp / test_sd
        return imp * (normrv.cdf(z)) + explore_discount * test_sd * normrv.pdf(z)

    def ieci(self, x_test, explore_discount, x_reference):
        normrv = norm()
        test_mean = self.emulator.predict(x_test[None, :])[0]
        pm = self.emulator.predict(x_reference)[0]
        if self.emu_type == "GP":
            tmp_gp = EmuGP(
                x=np.vstack([self.emulator.x, x_test]),
                y=np.hstack([self.emulator.y, test_mean]),
                nugget=self.emulator.nugget,
            )
            tmp_gp.beta = self.emulator.beta
            tmp_gp.marginal_variance = self.emulator.marginal_variance
        elif self.emu_type == "MFGP":
            tmp_gp = EmuMF(
                x_low=self.emulator.x_low,
                y_low=self.emulator.y_low,
                x=np.vstack([self.emulator.x, x_test]),
                y=np.hstack([self.emulator.y, test_mean]),
                nugget=self.emulator.nugget,
            )
            tmp_gp.beta_l = self.emulator.beta_l
            tmp_gp.beta_h = self.emulator.beta_h
            tmp_gp.marginal_variance_l = self.emulator.marginal_variance_l
            tmp_gp.marginal_variance_h = self.emulator.marginal_variance_h
        else:
            raise ValueError(f'IECI not implemented for "{self.emu_type}"')
        pm, psd = tmp_gp.predict(x_reference)
        psd = np.maximum(psd, _EPS)
        f_max = np.max(pm)
        imp = pm - f_max
        z = imp / psd
        return np.mean(imp * (normrv.cdf(z)) + explore_discount * psd * normrv.pdf(z))

    def find_candidate(self, explore_discount=1.0):
        n_feats = self.emulator.x.shape[1]
        bnds = [(self.x_lower[i], self.x_upper[i]) for i in range(n_feats)]
        ei_opts = [
            minimize(
                lambda x: -self.logei(x),
                self.rng.uniform(self.x_lower, self.x_upper),
                method="L-BFGS-B",
                tol=1e-8,
                bounds=bnds,
            )
            for _ in range(10)
        ]
        best_case = np.argmin([x["fun"] for x in ei_opts])
        return ei_opts[best_case]["x"]

    def find_candidate_ieci(self, x_reference, subsample_ref, explore_discount):
        nref = int(np.floor(x_reference.shape[0] * subsample_ref))
        self.refi = self.rng.choice(np.arange(x_reference.shape[0]), nref, replace=False)
        self.iecis = np.zeros(nref)

        for ii in range(nref):
            self.iecis[ii] = self.ieci(
                x_reference[self.refi[ii], :], explore_discount, x_reference
            )
        best_ind = self.refi[np.argmin(self.iecis)]
        return x_reference[best_ind, :]

    def run_opt(self, iterations=1, explore_discount=1.0):
        for _ in range(iterations):
            new_x = self.find_candidate(explore_discount)
            new_y = self.func(new_x)

            self.data["x"] = np.vstack([self.emulator.x, new_x])
            self.data["y"] = np.hstack([self.emulator.y, new_y])
            self.emulator = initialize_emulator(self.emu_type, self.data)

    def run_opt_ieci(self, x_reference, iterations=1, subsample_ref=1.0, explore_discount=1.0):
        for _ in range(iterations):
            new_x = self.find_candidate_ieci(x_reference, subsample_ref, explore_discount)
            new_y = self.func(new_x)

            self.data["x"] = np.vstack([self.emulator.x, new_x])
            self.data["y"] = np.hstack([self.emulator.y, new_y])
            self.emulator = initialize_emulator(self.emu_type, self.data)


class OptMF:
    """Multi-fidelity Bayesian optimization orchestrator (two fidelities).

    Parameters
    - func_low, func_high: Callables for low/high fidelity evaluations.
    - data_dict: For MFGPOld: keys {"x_low", "y_low", "x", "y", (optional) "nugget"}.
    - emulator_type: Must be "MFGPOld".
    - Bounds: If None, derived from x_low range.
    - random_state: Seed for reproducibility.
    """

    def __init__(
        self,
        func_low,
        func_high,
        data_dict,
        emulator_type="MFGPOld",
        x_lower=None,
        x_upper=None,
        nugget=default_nugget,
        random_state=None,
    ):
        if emulator_type not in ["MFGPOld"]:
            raise ValueError(
                "Emulator " + emulator_type + " not implemented for multifidelity optimization"
            )

        self.func_low = func_low
        self.func_high = func_high
        self.data = data_dict
        self.emu_type = emulator_type
        self.rng = np.random.default_rng(random_state)

        if "nugget" not in list(data_dict.keys()) and emulator_type in ["EmuMFOld"]:
            data_dict["nugget"] = nugget

        self.emulator = initialize_emulator(emulator_type, data_dict)

        self.iecis_low = None
        self.iecis_high = None

        self.evaluated_fidelities = []

        # Below assumes that the low and high fidelities share the same parameter space.
        if x_lower is None:
            self.x_lower = np.min(data_dict["x_low"], 0)
        else:
            self.x_lower = x_lower

        if x_upper is None:
            self.x_upper = np.max(data_dict["x_low"], 0)
        else:
            self.x_upper = x_upper

    def ieci_high(self, x_test, explore_discount, x_reference):
        normrv = norm()
        test_mean = self.emulator.predict(x_test[None, :])[0]
        pm = self.emulator.predict(x_reference)[0]
        tmp_gp = EmuMFOld(
            x_low=self.emulator.x_low,
            y_low=self.emulator.y_low,
            x=np.vstack([self.emulator.x, x_test]),
            y=np.hstack([self.emulator.y, test_mean]),
            nugget=self.emulator.nugget,
        )
        tmp_gp.beta_l = self.emulator.beta_l
        tmp_gp.beta_h = self.emulator.beta_h
        tmp_gp.marginal_variance_l = self.emulator.marginal_variance_l
        tmp_gp.marginal_variance_h = self.emulator.marginal_variance_h
        pm, psd = tmp_gp.predict(x_reference)
        psd = np.maximum(psd, _EPS)
        f_max = np.max(pm)
        imp = pm - f_max
        z = imp / psd
        return np.mean(imp * (normrv.cdf(z)) + explore_discount * psd * normrv.pdf(z))

    def ieci_low(self, x_test, explore_discount, x_reference):
        normrv = norm()
        test_mean = self.emulator.predict(x_test[None, :])[0]
        pm = self.emulator.predict(x_reference)[0]
        tmp_gp = EmuMFOld(
            x_low=np.vstack([self.emulator.x_low, x_test]),
            y_low=np.hstack([self.emulator.y_low, test_mean]),
            x=self.emulator.x,
            y=self.emulator.y,
            nugget=self.emulator.nugget,
        )
        tmp_gp.beta_l = self.emulator.beta_l
        tmp_gp.beta_h = self.emulator.beta_h
        tmp_gp.marginal_variance_l = self.emulator.marginal_variance_l
        tmp_gp.marginal_variance_h = self.emulator.marginal_variance_h
        pm, psd = tmp_gp.predict(x_reference)
        psd = np.maximum(psd, _EPS)
        f_max = np.max(pm)
        imp = pm - f_max
        z = imp / psd
        return np.mean(imp * (normrv.cdf(z)) + explore_discount * psd * normrv.pdf(z))

    def find_candidate(self, x_reference, cost_ratio=1.0, subsample_ref=1.0, explore_discount=1.0):
        nref = int(np.floor(x_reference.shape[0] * subsample_ref))
        self.refi = self.rng.choice(np.arange(x_reference.shape[0]), nref, replace=False)

        self.iecis_low = np.zeros(nref)
        self.iecis_high = np.zeros(nref)
        for ii in range(nref):
            self.iecis_low[ii] = self.ieci_low(
                x_reference[self.refi[ii], :], explore_discount, x_reference
            )
            self.iecis_high[ii] = self.ieci_high(
                x_reference[self.refi[ii], :], explore_discount, x_reference
            )
        # offset = np.min([self.iecis_low.min(), self.iecis_high.min()])
        best_low_ind = np.argmin(self.iecis_low)
        best_high_ind = np.argmin(self.iecis_high)
        best_low_val = np.min(self.iecis_low)
        best_high_val = np.min(self.iecis_high)
        # self.iecis_high -= offset
        # self.iecis_low -= offset
        if best_high_val / (best_low_val + _EPS) <= cost_ratio:
            ret_type = "high"
            best_ind = self.refi[best_high_ind]
        else:
            ret_type = "low"
            best_ind = self.refi[best_low_ind]
        return (x_reference[best_ind, :], ret_type)

    def run_opt(self, x_reference, iterations=1, cost_ratio=1, subsample_ref=1.0, explore_discount=1.0):
        for _ in range(iterations):
            new_x, new_type = self.find_candidate(
                x_reference, cost_ratio, subsample_ref, explore_discount
            )

            if new_type == "low":
                new_y = self.func_low(new_x)
                self.data["x_low"] = np.vstack([self.emulator.x_low, new_x])
                self.data["y_low"] = np.hstack([self.emulator.y_low, new_y])
                self.emulator = initialize_emulator(self.emu_type, self.data)
                self.evaluated_fidelities.append("low")
            else:
                new_y = self.func_high(new_x)
                self.data["x"] = np.vstack([self.emulator.x, new_x])
                self.data["y"] = np.hstack([self.emulator.y, new_y])
                self.emulator = initialize_emulator(self.emu_type, self.data)
                self.evaluated_fidelities.append("high")


class ConstrainedOpt(Opt):
    def __init__(
        self,
        func,
        data_dict,
        constraint_dicts,
        x_lower=None,
        x_upper=None,
        emulator_type="GP",
        nugget=default_nugget,
        constraint_weight=1,
        random_state=None,
    ):
        super().__init__(
            func, data_dict, x_lower, x_upper, emulator_type, nugget, random_state
        )
        self.constraints = constraint_dicts
        self.c_weight = constraint_weight
        if any([x["sign"] not in ["lessThan", "greaterThan"] for x in self.constraints]):
            raise ValueError(
                "At least one sign in the constraints is not lessThan or greaterThan"
            )
        # Ensure nugget present for constraint emulators if needed
        c_dicts = []
        for cdict in constraint_dicts:
            cdict = dict(cdict)
            if "nugget" not in cdict and emulator_type in ["GP", "MFGP"]:
                cdict["nugget"] = nugget
            c_dicts.append(cdict)
        self.cemus = [initialize_emulator(emulator_type, cdict) for cdict in c_dicts]

    def constraint_logprob(self, x):
        test_predictions = [emu.predict(x[None, :]) for emu in self.cemus]
        ps = []
        for (m, s), thresh in zip(test_predictions, self.constraints):
            s = np.maximum(s, _EPS)
            z = (thresh["value"] - m) / s
            if thresh["sign"] == "lessThan":
                ps.append(norm().logcdf(z))
            else:
                ps.append(norm().logcdf(-z))
        return np.sum(ps)

    def find_candidate(self, explore_discount=1.0):
        n_feats = self.emulator.x.shape[1]
        bnds = [(self.x_lower[i], self.x_upper[i]) for i in range(n_feats)]
        ei_opts = [
            minimize(
                lambda x: -(self.logei(x) + self.c_weight * self.constraint_logprob(x)),
                self.rng.uniform(self.x_lower, self.x_upper),
                method="L-BFGS-B",
                tol=1e-4,
                bounds=bnds,
            )
            for _ in range(10)
        ]
        best_case = np.argmin([x["fun"] for x in ei_opts])
        # LBFGS can have some problems with optimizing this - probably because the probability
        # flattens the optimization surface a bit, particularly with simple to emulate constraints.
        # Might be a place for improvement but works on my test case for now. Will work on
        # improving it going forward.
        return ei_opts[best_case]["x"]

    def find_candidate_ieci(self, x_reference, subsample_ref, explore_discount):
        nref = int(np.floor(x_reference.shape[0] * subsample_ref))
        self.refi = self.rng.choice(np.arange(x_reference.shape[0]), nref, replace=False)
        self.iecis = np.zeros(nref)

        for ii in range(nref):
            base = self.ieci(x_reference[self.refi[ii], :], explore_discount, x_reference)
            self.iecis[ii] = np.log(base) + self.c_weight * self.constraint_logprob(
                x_reference[self.refi[ii], :]
            )
        best_ind = self.refi[np.argmin(self.iecis)]
        return x_reference[best_ind, :]
