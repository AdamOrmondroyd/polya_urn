from dataclasses import dataclass
from typing import List
import numpy as np

rng = np.random.default_rng()


@dataclass
class RTI:
    """Keeps track of info for nested sampling."""

    nlive: int = 1000
    cluster: np.array = np.array([0] * nlive)
    logLs: np.array = rng.uniform(-10, 10, size=nlive) ** 2 / 2 - np.log(2 * np.pi) / 2
    logZ: np.float64 = -np.inf
    logZ_p: List[float] = np.array([logZ])
    logX: np.float64 = 0.0
    logX_p: np.array = np.array([logX])
    logX_p_bar: np.array = np.array([logX])
    logX_p_X_q_bar: np.array = np.array([[2 * logX]])
    logZ_bar: np.float64 = logZ
    logZ2_bar: np.float64 = 2 * logZ
    logZ_p_bar: np.float64 = np.array([logZ_bar])
    logZ2_p_bar: np.array = np.array([logZ2_bar])
    logZ_X_p_bar: np.array = np.array([-np.inf])
    logZ_p_X_p_bar: np.array = np.array([-np.inf])
