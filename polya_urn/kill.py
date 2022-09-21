"""
Killing live points and propagating errors
"""

import numpy as np
from scipy.special import logsumexp

rng = np.random.default_rng()
## update errors at kill
log2 = np.log(2)


def kill_errors(
    idx,
    rti,
):
    p = rti.cluster[idx]
    n_p = sum([q == p for q in rti.cluster])
    logL_dead = rti.logLs[idx]

    # C4
    rti.logZ_bar = logsumexp(
        [rti.logZ_bar, rti.logX_p_bar[p] + logL_dead - np.log(n_p + 1)]
    )
    # C5
    rti.logZ_p_bar[p] = logsumexp(
        [rti.logZ_p_bar[p], rti.logX_p_bar[p] + logL_dead - np.log(n_p + 1)]
    )
    # C14
    rti.logZ2_bar = logsumexp(
        [
            rti.logZ2_bar,
            log2 + rti.logZ_p_X_p_bar[p] + logL_dead - np.log(n_p + 1),
            log2
            + rti.logX_p_X_q_bar[p, p]
            + 2 * logL_dead
            - np.log((n_p + 1) * (n_p + 2)),
        ]
    )
    # C15
    rti.logZ2_p_bar[p] = logsumexp(
        [
            rti.logZ2_p_bar[p],
            log2 + rti.logZ_p_X_p_bar[p] + logL_dead - np.log(n_p + 1),
            log2
            + rti.logX_p_X_q_bar[p, p]
            + 2 * logL_dead
            - np.log((n_p + 1) * (n_p + 2)),
        ]
    )
    # C16
    rti.logZ_X_p_bar[p] = logsumexp(
        [
            np.log(n_p / (n_p + 1)) + rti.logZ_X_p_bar[p],
            rti.logX_p_X_q_bar[p, p]
            + logL_dead
            + np.log(n_p / ((n_p + 1) * (n_p + 2))),
        ]
    )
    # C17
    for q in np.unique(rti.cluster):
        if q != p:
            rti.logZ_X_p_bar[q] = logsumexp(
                [
                    rti.logZ_X_p_bar[q],
                    rti.logX_p_X_q_bar[p, q] + logL_dead - np.log(n_p + 1),
                ]
            )
    # C18
    rti.logZ_p_X_p_bar[p] = logsumexp(
        [
            np.log(n_p / (n_p + 1)) + rti.logZ_p_X_p_bar[p],
            np.log(n_p / ((n_p + 1) * (n_p + 2)))
            + rti.logX_p_X_q_bar[p, p]
            + logL_dead,
        ]
    )
    # C6
    rti.logX_p_bar[p] += np.log(n_p / (n_p + 1))
    # C19
    rti.logX_p_X_q_bar[p, p] += np.log(n_p / (n_p + 2))
    # C20
    for q in np.unique(rti.cluster):
        if q != p:
            rti.logX_p_X_q_bar[p, q] += np.log(n_p / (n_p + 1))
            rti.logX_p_X_q_bar[q, p] = rti.logX_p_X_q_bar[p, q]

    return rti


## kill lowest liklihood point
def kill(rti):
    idx = np.argmin(rti.logLs)
    p = rti.cluster[idx]
    t = rng.power(sum([q == p for q in rti.cluster]))
    rti.logZ = logsumexp([rti.logZ, np.log(1 - t) + rti.logX_p[p] + rti.logLs[idx]])
    rti.logZ_p[p] = logsumexp(
        [rti.logZ_p[p], np.log(1 - t) + rti.logX_p[p] + rti.logLs[idx]]
    )
    rti.logX_p[p] += np.log(t)

    rti = kill_errors(idx, rti)

    ## create new live point bugject to constraint

    new_L = -np.inf
    while new_L <= rti.logLs[idx]:
        new_L = -rng.uniform(-10, 10) ** 2 / 2 - np.log(2 * np.pi) / 2
    rti.logLs[idx] = new_L

    ## this live point needs to be assigned a cluster proportional to volume
    rti.cluster[idx] = rng.choice(
        np.arange(max(rti.cluster) + 1),
        p=[x / np.exp(logsumexp(rti.logX_p)) for x in np.exp(rti.logX_p)],
    )

    return rti
