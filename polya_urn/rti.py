from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from polya_urn.cluster import clustering
from polya_urn.kill import kill

rng = np.random.default_rng()


@dataclass
class RTI:
    """Keeps track of info for nested sampling."""

    nlive: int = 1000
    cluster: np.array = np.array([0] * nlive)
    logLs: np.array = -rng.uniform(-10, 10, size=nlive) ** 2 / 2 - np.log(2 * np.pi) / 2
    logZ: np.float64 = -np.inf
    logZ_p: np.array = np.array([logZ])
    logX_p: np.array = np.array([0.0])
    logX_p_bar: np.array = np.array(logX_p)
    logX_p_X_q_bar: np.array = np.array([2 * logX_p])
    logZ_bar: np.float64 = logZ
    logZ2_bar: np.float64 = 2 * logZ
    logZ_p_bar: np.array = np.array([logZ_bar])
    logZ2_p_bar: np.array = np.array([logZ2_bar])
    logZ_X_p_bar: np.array = np.array([-np.inf])
    logZ_p_X_p_bar: np.array = np.array([-np.inf])

    @property
    def logX(self):
        return logsumexp(self.logX_p)

    @property
    def sigma_X_p(self):
        return np.sqrt(
            np.exp(np.diag(self.logX_p_X_q_bar)) - np.exp(self.logX_p_bar * 2)
        )

    @property
    def sigma_Z_p(self):
        return np.sqrt(np.exp(self.logZ2_p_bar) - np.exp(self.logZ_p_bar * 2))

    @property
    def sigma_Z(self):
        return np.sqrt(np.exp(self.logZ2_bar) - np.exp(self.logZ_bar * 2))

    @property
    def ns(self):
        return np.array(
            [sum([q == r for r in self.cluster]) for q in np.unique(self.cluster)]
        )

    def kill(self):
        self = kill(self)

    def clustering(self):
        self = clustering(self)
