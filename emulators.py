import numpy as np

from .covariances import get_obs_cov, get_cross_cov
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def initialize_emulator(emulator_type, data_dict):
    """Factory to initialize and fit an emulator.

    emulator_type:
        - "GP": requires keys {"x", "y", "nugget"}
        - "MFGP": if "prior_emu" provided, uses it; otherwise requires
          {"x_low", "y_low", "x", "y", "nugget"} and builds the low GP first.
        - "MFGPOld": legacy multi-fidelity emulator, requires
          {"x_low", "y_low", "x", "y", "nugget"}.
    """
    if emulator_type == "GP":
        emulator = EmuGP(x=data_dict["x"], y=data_dict["y"], nugget=data_dict["nugget"])
        emulator.fit()
    elif emulator_type == "MFGP":
        if "prior_emu" in data_dict.keys():
            emulator = EmuMF(
                x=data_dict["x"],
                y=data_dict["y"],
                prior_emu=data_dict["prior_emu"],
                nugget=data_dict["nugget"],
            )
            emulator.fit()
        else:
            low_emu = EmuGP(x=data_dict["x_low"], y=data_dict["y_low"], nugget=data_dict["nugget"])
            low_emu.fit()
            emulator = EmuMF(
                x=data_dict["x"],
                y=data_dict["y"],
                prior_emu=low_emu,
                nugget=data_dict["nugget"],
            )
            emulator.fit()
    elif emulator_type == "MFGPOld":
        emulator = EmuMFOld(
            x_low=data_dict["x_low"],
            x=data_dict["x"],
            y_low=data_dict["y_low"],
            y=data_dict["y"],
            nugget=data_dict["nugget"],
        )
        emulator.fit()
    else:
        raise ValueError(
            'Emulator "' + emulator_type + '" not implemented. Check documentation for all available emulators.'
        )
    return emulator


class EmuGP:
    def gp_nll(self, params, x, y):
        xs = self.x_scaler.transform(x)
        ys = self.y_scaler.transform(y[:, None]).flatten()
        n = x.shape[0]
        p = x.shape[1]

        beta = np.exp(params[: p])
        marginal_variance = np.exp(params[p])

        obs_cov = get_obs_cov(xs, beta, marginal_variance) + self.nugget * np.eye(n)

        # Gaussian prior on log-lengthscales (params[:p])
        ell = params[:p]
        prior_mean, prior_scale = (np.sqrt(2) + np.log(p) / 2.0, np.sqrt(3.0))
        prior_nll = np.sum(
            np.log(prior_scale)
            + 0.5 * np.log(2.0 * np.pi)
            + (ell - prior_mean) ** 2 / (2.0 * prior_scale ** 2)
        )

        try:
            cho = cho_factor(obs_cov, lower=True, check_finite=False)
        except Exception as e:
            # fallback: add a small jitter to ensure PD
            jitter = 1e-12
            for _ in range(5):
                try:
                    cho = cho_factor(obs_cov + jitter * np.eye(n), lower=True, check_finite=False)
                    break
                except Exception:
                    jitter *= 10
            else:
                raise e
        alpha = cho_solve(cho, ys, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
        return 0.5 * logdet + 0.5 * ys.T @ alpha + prior_nll

    def fit(self, n_optimization_restarts=11, random_state=None):
        n_params = self.p + 1
        bnds = [(np.log(0.01), np.log(100)) for _ in range(n_params)]
        rng = np.random.default_rng(random_state)
        gp_list = []
        init = np.concatenate([0.5 * rng.standard_normal(self.p), 0.5 * rng.standard_normal(1)])
        gp_list.append(
            minimize(
                self.gp_nll,
                init,
                method="L-BFGS-B",
                tol=1e-8,
                args=(self.x, self.y),
                bounds=bnds,
            )
        )
        for _ in range(n_optimization_restarts - 1):
            init = np.concatenate([0.5 * rng.standard_normal(self.p), 0.5 * rng.standard_normal(1)])
            gp_list.append(
                minimize(
                    self.gp_nll,
                    init,
                    method="L-BFGS-B",
                    tol=1e-8,
                    args=(self.x, self.y),
                    bounds=bnds,
                )
            )
        gp_opts = gp_list[np.argmin([x["fun"] for x in gp_list])]

        self.beta = np.exp(gp_opts["x"][: self.p])
        self.marginal_variance = np.exp(gp_opts["x"][self.p])
        self._obs_cov = (
            get_obs_cov(self.xs, self.beta, self.marginal_variance) + self.nugget * np.eye(self.n)
        )
        self._cho = cho_factor(self._obs_cov, lower=True, check_finite=False)
        self._alpha = cho_solve(self._cho, self.ys, check_finite=False)

    def predict(self, x_new, return_full_cov=False):
        xs_new = self.x_scaler.transform(x_new)
        n_pred = xs_new.shape[0]
        if return_full_cov:
            pred_cov = get_obs_cov(xs_new, self.beta, self.marginal_variance) + self.nugget * np.eye(n_pred)
            cross_cov = get_cross_cov(xs_new, self.xs, self.beta, self.marginal_variance)

            pred_mean = cross_cov @ self._alpha
            if self._cho_I is None:
                invK_cross = cho_solve(self._cho, cross_cov.T, check_finite=False)
                pred_var = pred_cov - cross_cov @ invK_cross
            else:
                cross_cov_I = get_cross_cov(xs_new, self.xs_I, self.beta, self.marginal_variance)
                invK_I_cross = cho_solve(self._cho_I, cross_cov_I.T, check_finite=False)
                pred_var = pred_cov - cross_cov_I @ invK_I_cross

            pred_mean = self.y_scaler.inverse_transform(pred_mean[:, None]).flatten()
            scale = float(self.y_scaler.scale_)
            pred_cov = scale * pred_var * scale
            return [pred_mean, pred_cov]
        else:
            ret_means = np.zeros(n_pred)
            ret_sds = np.zeros(n_pred)
            for ii in range(n_pred):
                pred_cov = (
                    get_obs_cov(xs_new[[ii], :], self.beta, self.marginal_variance) + self.nugget * np.eye(1)
                )
                cross_cov = get_cross_cov(xs_new[[ii], :], self.xs, self.beta, self.marginal_variance)

                pred_mean = cross_cov @ self._alpha
                if self._cho_I is None:
                    invK_cross = cho_solve(self._cho, cross_cov.T, check_finite=False)
                    pred_var = pred_cov - cross_cov @ invK_cross
                else:
                    cross_cov_I = get_cross_cov(xs_new[[ii], :], self.xs_I, self.beta, self.marginal_variance)
                    invK_I_cross = cho_solve(self._cho_I, cross_cov_I.T, check_finite=False)
                    pred_var = pred_cov - cross_cov_I @ invK_I_cross

                ret_means[ii] = self.y_scaler.inverse_transform(pred_mean[:, None]).flatten().item()
                ret_sds[ii] = float(self.y_scaler.scale_) * np.sqrt(float(np.diag(pred_var)))
            return [ret_means, ret_sds]

    def add_impute_data(self, x, y):
        if self._obs_cov is None:
            print("Need to fit GP before the imputed data can be handled.")
            return 0
        self.x_I = x
        self.y_I = y
        self.n_I = x.shape[0]
        self.p_I = x.shape[1]
        self.xs_I = self.x_scaler.transform(self.x_I)
        self.ys_I = self.y_scaler.transform(self.y_I[:, None]).flatten()

        self._obs_cov_I = (
            get_obs_cov(self.xs_I, self.beta, self.marginal_variance) + self.nugget * np.eye(self.n_I)
        )
        self._cho_I = cho_factor(self._obs_cov_I, lower=True, check_finite=False)

    def __init__(self, x, y, nugget=1.0e-6):
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.p = x.shape[1]

        self.nugget = nugget
        self.x_scaler = MinMaxScaler().fit(self.x)
        self.xs = self.x_scaler.transform(self.x)

        self.y_scaler = StandardScaler().fit(self.y[:, None])
        self.ys = self.y_scaler.transform(self.y[:, None]).flatten()

        self._obs_cov = None
        self._cho = None
        self._alpha = None
        self._cho_I = None
        self.y_I = None


class EmuMF(EmuGP):
    def gp_nll(self, params, x, y):
        xs = self.x_scaler.transform(x)
        ys = self.y_scaler.transform(y[:, None]).flatten()
        n = x.shape[0]
        p = x.shape[1]

        beta = np.exp(params[:p])
        marginal_variance = np.exp(params[p])

        obs_cov = self.cov_s_low + get_obs_cov(xs, beta, marginal_variance) + self.nugget * np.eye(n)

        # Gaussian prior on log-lengthscales
        ell = params[:p]
        ls_prior_mean, ls_prior_scale = (np.sqrt(2) + np.log(p) / 2.0, np.sqrt(3.0))
        prior_nll = np.sum(
            np.log(ls_prior_scale)
            + 0.5 * np.log(2.0 * np.pi)
            + (ell - ls_prior_mean) ** 2 / (2.0 * ls_prior_scale ** 2)
        )

        try:
            cho = cho_factor(obs_cov, lower=True, check_finite=False)
        except Exception as e:
            jitter = 1e-12
            for _ in range(5):
                try:
                    cho = cho_factor(obs_cov + jitter * np.eye(n), lower=True, check_finite=False)
                    break
                except Exception:
                    jitter *= 10
            else:
                raise e
        rhs = ys - self.mu_s_low
        alpha = cho_solve(cho, rhs, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(cho[0])))
        return 0.5 * logdet + 0.5 * rhs.T @ alpha + prior_nll

    def fit(self, n_optimization_restarts=11, random_state=None):
        n_params = self.p + 1
        bnds = [(np.log(0.01), np.log(100)) for _ in range(n_params)]
        rng = np.random.default_rng(random_state)
        gp_list = []
        init = np.concatenate([0.5 * rng.standard_normal(self.p), 0.5 * rng.standard_normal(1)])
        gp_list.append(
            minimize(
                self.gp_nll,
                init,
                method="L-BFGS-B",
                tol=1e-8,
                args=(self.x, self.y),
                bounds=bnds,
            )
        )
        for _ in range(n_optimization_restarts - 1):
            init = np.concatenate([0.5 * rng.standard_normal(self.p), 0.5 * rng.standard_normal(1)])
            gp_list.append(
                minimize(
                    self.gp_nll,
                    init,
                    method="L-BFGS-B",
                    tol=1e-8,
                    args=(self.x, self.y),
                    bounds=bnds,
                )
            )
        gp_opts = gp_list[np.argmin([x["fun"] for x in gp_list])]

        self.beta = np.exp(gp_opts["x"][: self.p])
        self.marginal_variance = np.exp(gp_opts["x"][self.p])
        self._obs_cov = (
            self.cov_s_low + get_obs_cov(self.xs, self.beta, self.marginal_variance) + self.nugget * np.eye(self.n)
        )
        self._cho = cho_factor(self._obs_cov, lower=True, check_finite=False)
        rhs = self.ys - self.mu_s_low
        self._alpha = cho_solve(self._cho, rhs, check_finite=False)

    def predict(self, x_new, return_full_cov=False):
        xs_new = self.x_scaler.transform(x_new)
        n_pred = xs_new.shape[0]

        if return_full_cov:
            prior_mean, prior_cov = self.prior_emulator.predict(x_new, return_full_cov=True)

            pred_cov = get_obs_cov(xs_new, self.beta, self.marginal_variance) + self.nugget * np.eye(n_pred)
            cross_cov = get_cross_cov(xs_new, self.xs, self.beta, self.marginal_variance)

            pred_mean = cross_cov @ self._alpha
            if self._cho_I is None:
                invK_cross = cho_solve(self._cho, cross_cov.T, check_finite=False)
                pred_var = pred_cov - cross_cov @ invK_cross
            else:
                cross_cov_I = get_cross_cov(xs_new, self.xs_I, self.beta, self.marginal_variance)
                invK_I_cross = cho_solve(self._cho_I, cross_cov_I.T, check_finite=False)
                pred_var = pred_cov - cross_cov_I @ invK_I_cross

            scale = float(self.y_scaler.scale_)
            pred_mean = scale * pred_mean
            pred_cov = (scale * pred_var * scale)
            return [pred_mean + prior_mean, pred_cov + prior_cov]
        else:
            prior_mean, prior_sds = self.prior_emulator.predict(x_new, return_full_cov=False)

            ret_means = np.zeros(n_pred)
            ret_sds = np.zeros(n_pred)
            for ii in range(n_pred):
                pred_cov = (
                    get_obs_cov(xs_new[[ii], :], self.beta, self.marginal_variance) + self.nugget * np.eye(1)
                )
                cross_cov = get_cross_cov(xs_new[[ii], :], self.xs, self.beta, self.marginal_variance)

                pred_mean = cross_cov @ self._alpha
                if self._cho_I is None:
                    invK_cross = cho_solve(self._cho, cross_cov.T, check_finite=False)
                    pred_var = pred_cov - cross_cov @ invK_cross
                else:
                    cross_cov_I = get_cross_cov(xs_new[[ii], :], self.xs_I, self.beta, self.marginal_variance)
                    invK_I_cross = cho_solve(self._cho_I, cross_cov_I.T, check_finite=False)
                    pred_var = pred_cov - cross_cov_I @ invK_I_cross

                ret_means[ii] = float(self.y_scaler.scale_) * float(pred_mean)
                ret_sds[ii] = float(self.y_scaler.scale_) * np.sqrt(float(np.diag(pred_var)))
            return [ret_means + prior_mean, np.sqrt(ret_sds ** 2 + prior_sds ** 2)]

    def add_impute_data(self, x, y):
        if self._obs_cov is None:
            print("Need to fit GP before the imputed data can be handled.")
            return 0
        self.x_I = x
        self.y_I = y
        self.n_I = x.shape[0]
        self.p_I = x.shape[1]

        self.xs_I = self.x_scaler.transform(self.x_I)
        self.ys_I = self.y_scaler.transform(self.y_I[:, None]).flatten()

        _, prior_cov = self.prior_emulator.predict(self.x_I, return_full_cov=True)
        scale = float(self.y_scaler.scale_)
        cov_s_low = (1.0 / scale) * prior_cov * (1.0 / scale)
        self._obs_cov_I = cov_s_low + get_obs_cov(self.xs_I, self.beta, self.marginal_variance) + self.nugget * np.eye(self.n_I)
        self._cho_I = cho_factor(self._obs_cov_I, lower=True, check_finite=False)

    def __init__(self, x, y, prior_emu, nugget=1.0e-6):
        super().__init__(x, y, nugget)
        self.prior_emulator = prior_emu
        prior_mean, prior_cov = self.prior_emulator.predict(self.x, return_full_cov=True)
        self.mu_s_low = self.y_scaler.transform(prior_mean[:, None]).flatten()
        scale = float(self.y_scaler.scale_)
        self.cov_s_low = (1.0 / scale) * prior_cov * (1.0 / scale)


class EmuMFOld:
    def mf_nll(self, params, x_l, y_l, x_h, y_h):
        xs_l = self.x_scaler.transform(x_l)
        xs_h = self.x_scaler.transform(x_h)
        xs = np.vstack([xs_h, xs_l])

        ys_l = self.y_scaler.transform(y_l[:, None]).flatten()
        ys_h = self.y_scaler.transform(y_h[:, None]).flatten()
        ys = np.hstack([ys_h, ys_l])

        parm_l = np.exp(params[(self.p_h + 1) :])
        parm_h = np.exp(params[: (self.p_h + 1)])

        beta_l = parm_l[: self.p_l]
        beta_h = parm_h[: self.p_h]

        marginal_variance_l = parm_l[self.p_l]
        marginal_variance_h = parm_h[self.p_h]

        obs_cov = get_obs_cov(xs, beta_l, marginal_variance_l) + self.nugget * np.eye(xs.shape[0])
        obs_cov_h = get_obs_cov(xs_h, beta_h, marginal_variance_h)
        obs_cov[: self.n_h, : self.n_h] += obs_cov_h
        return -(
            -0.5 * np.linalg.slogdet(obs_cov)[1] - 0.5 * ys.T @ np.linalg.inv(obs_cov) @ ys
        )

    def fit(self, n_optimization_restarts=26):
        n_params = self.p_l + self.p_h + 2
        bnds = [(np.log(0.01), np.log(100)) for _ in range(n_params)]
        gp_list = []
        gp_list.append(
            minimize(
                self.mf_nll,
                np.concatenate(
                    [
                        0.5 * np.random.randn(self.p_h),
                        0.5 * np.random.randn(1),
                        0.5 * np.random.randn(self.p_l),
                        0.5 * np.random.randn(1),
                    ]
                ),
                method="L-BFGS-B",
                tol=1e-8,
                args=(self.x_low, self.y_low, self.x, self.y),
                bounds=bnds,
            )
        )
        for _ in range(n_optimization_restarts - 1):
            gp_list.append(
                minimize(
                    self.mf_nll,
                    np.concatenate(
                        [
                            0.5 * np.random.randn(self.p_h),
                            0.5 * np.random.randn(1),
                            0.5 * np.random.randn(self.p_l),
                            0.5 * np.random.randn(1),
                        ]
                    ),
                    method="L-BFGS-B",
                    tol=1e-8,
                    args=(self.x_low, self.y_low, self.x, self.y),
                    bounds=bnds,
                )
            )
        gp_opts = gp_list[np.argmin([x["fun"] for x in gp_list])]
        parm_h = np.exp(gp_opts["x"][: (self.p_h + 1)])
        parm_l = np.exp(gp_opts["x"][(self.p_h + 1) :])

        self.beta_l = parm_l[: self.p_l]
        self.beta_h = parm_h[: self.p_h]

        self.marginal_variance_l = parm_l[self.p_l]
        self.marginal_variance_h = parm_h[self.p_h]

    def predict(self, x_new):
        xs_new = self.x_scaler.transform(x_new)
        n_pred = xs_new.shape[0]
        obs_cov = get_obs_cov(self.xs, self.beta_l, self.marginal_variance_l) + self.nugget * np.eye(self.n)
        obs_cov_h = get_obs_cov(self.xs_high, self.beta_h, self.marginal_variance_h)
        pred_cov = get_obs_cov(xs_new, self.beta_l, self.marginal_variance_l) + self.nugget * np.eye(n_pred)
        pred_cov += get_obs_cov(xs_new, self.beta_h, self.marginal_variance_h)
        obs_cov[: self.n_h, : self.n_h] += obs_cov_h

        cross_cov = get_cross_cov(xs_new, self.xs, self.beta_l, self.marginal_variance_l)
        cross_cov_h = get_cross_cov(xs_new, self.xs_high, self.beta_h, self.marginal_variance_h)
        cross_cov[:, : self.n_h] += cross_cov_h

        inv_obs_cov = np.linalg.inv(obs_cov)

        pred_mean = cross_cov @ (inv_obs_cov @ self.ys)
        pred_var = pred_cov - cross_cov @ (inv_obs_cov @ cross_cov.T)

        pred_mean = self.y_scaler.inverse_transform(pred_mean[:, None]).flatten()
        return [pred_mean, float(self.y_scaler.scale_) * np.sqrt(np.diag(pred_var))]

    def predict_low(self, x_new):
        xs_new = self.x_scaler.transform(x_new)
        n_pred = xs_new.shape[0]
        obs_cov = get_obs_cov(self.xs_low, self.beta_l, self.marginal_variance_l) + self.nugget * np.eye(self.n_l)
        pred_cov = get_obs_cov(xs_new, self.beta_l, self.marginal_variance_l) + self.nugget * np.eye(n_pred)

        cross_cov = get_cross_cov(xs_new, self.xs_low, self.beta_l, self.marginal_variance_l)
        inv_obs_cov = np.linalg.inv(obs_cov)

        pred_mean = cross_cov @ (inv_obs_cov @ self.ys_low)
        pred_var = pred_cov - cross_cov @ (inv_obs_cov @ cross_cov.T)
        pred_mean = self.y_scaler.inverse_transform(pred_mean[:, None]).flatten()
        return [pred_mean, float(self.y_scaler.scale_) * np.sqrt(np.diag(pred_var))]

    def __init__(self, x_low, y_low, x, y, nugget=1.0e-6):
        self.x_low = x_low
        self.y_low = y_low
        self.x = x
        self.y = y

        self.n_l = x_low.shape[0]
        self.n_h = x.shape[0]
        self.p_l = x_low.shape[1]
        self.p_h = x.shape[1]

        self.nugget = nugget

        self.n = self.n_l + self.n_h
        self.x_all = np.vstack([x, x_low])
        self.y_all = np.hstack([y, y_low])

        self.x_scaler = MinMaxScaler().fit(self.x_all)
        self.xs = self.x_scaler.transform(self.x_all)
        self.xs_low = self.x_scaler.transform(self.x_low)
        self.xs_high = self.x_scaler.transform(self.x)

        self.y_scaler = StandardScaler().fit(self.y_low[:, None])
        self.ys = self.y_scaler.transform(self.y_all[:, None]).flatten()
        self.ys_low = self.y_scaler.transform(self.y_low[:, None]).flatten()
        self.ys_high = self.y_scaler.transform(self.y[:, None]).flatten()
        self.y_I = None
