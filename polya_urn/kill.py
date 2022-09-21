"""
Killing live points and propagating errors
"""

import numpy as np
from scipy.special import logsumexp

rng = np.random.default_rng()
## update errors at kill
log2 = np.log(2)


def kill_errors(
    p,
    n_p,
    cluster,
    logL_dead,
    logX_p_bar,
    logX_p_X_q_bar,
    logZ_bar,
    logZ2_bar,
    logZ_p_bar,
    logZ2_p_bar,
    logZ_X_p_bar,
    logZ_p_X_p_bar,
):
    # C4
    logZ_bar = logsumexp([logZ_bar, logX_p_bar[p] + logL_dead - np.log(n_p + 1)])
    # C5
    logZ_p_bar[p] = logsumexp(
        [logZ_p_bar[p], logX_p_bar[p] + logL_dead - np.log(n_p + 1)]
    )
    # C14
    logZ2_bar = logsumexp(
        [
            logZ2_bar,
            log2 + logZ_p_X_p_bar[p] + logL_dead - np.log(n_p + 1),
            log2 + logX_p_X_q_bar[p, p] + 2 * logL_dead - np.log((n_p + 1) * (n_p + 2)),
        ]
    )
    # C15
    logZ2_p_bar[p] = logsumexp(
        [
            logZ2_p_bar[p],
            log2 + logZ_p_X_p_bar[p] + logL_dead - np.log(n_p + 1),
            log2 + logX_p_X_q_bar[p, p] + 2 * logL_dead - np.log((n_p + 1) * (n_p + 2)),
        ]
    )
    # C16
    logZ_X_p_bar[p] = logsumexp(
        [
            np.log(n_p / (n_p + 1)) + logZ_X_p_bar[p],
            logX_p_X_q_bar[p, p] + logL_dead + np.log(n_p / ((n_p + 1) * (n_p + 2))),
        ]
    )
    # C17
    for q in np.unique(cluster):
        if q != p:
            logZ_X_p_bar[q] = logsumexp(
                [logZ_X_p_bar[q], logX_p_X_q_bar[p, q] + logL_dead - np.log(n_p + 1)]
            )
    # C18
    logZ_p_X_p_bar[p] = logsumexp(
        [
            np.log(n_p / (n_p + 1)) + logZ_p_X_p_bar[p],
            np.log(n_p / ((n_p + 1) * (n_p + 2))) + logX_p_X_q_bar[p, p] + logL_dead,
        ]
    )
    # C6
    logX_p_bar[p] += np.log(n_p / (n_p + 1))
    # C19
    logX_p_X_q_bar[p, p] += np.log(n_p / (n_p + 2))
    # C20
    for q in np.unique(cluster):
        if q != p:
            logX_p_X_q_bar[p, q] += np.log(n_p / (n_p + 1))
            logX_p_X_q_bar[q, p] = logX_p_X_q_bar[p, q]

    return (
        logX_p_bar,
        logX_p_X_q_bar,
        logZ_bar,
        logZ2_bar,
        logZ_p_bar,
        logZ2_p_bar,
        logZ_X_p_bar,
        logZ_p_X_p_bar,
    )


## kill lowest liklihood point
def kill(
    cluster,
    logLs,
    logX_p,
    logZ,
    logZ_p,
    logX_p_bar,
    logX_p_X_q_bar,
    logZ_bar,
    logZ_p_bar,
    logZ2_bar,
    logZ2_p_bar,
    logZ_X_p_bar,
    logZ_p_X_p_bar,
):
    idx = np.argmin(logLs)
    p = cluster[idx]
    t = rng.power(sum([q == p for q in cluster]))
    logZ = logsumexp([logZ, np.log(1 - t) + logX_p[p] + logLs[idx]])
    logZ_p[p] = logsumexp([logZ_p[p], np.log(1 - t) + logX_p[p] + logLs[idx]])
    logX_p[p] += np.log(t)

    n_p = sum([q == p for q in cluster])
    (
        logX_p_bar,
        logX_p_X_q_bar,
        logZ_bar,
        logZ2_bar,
        logZ_p_bar,
        logZ2_p_bar,
        logZ_X_p_bar,
        logZ_p_X_p_bar,
    ) = kill_errors(
        p,
        n_p,
        cluster,
        logLs[idx],
        logX_p_bar,
        logX_p_X_q_bar,
        logZ_bar,
        logZ_p_bar,
        logZ2_bar,
        logZ2_p_bar,
        logZ_X_p_bar,
        logZ_p_X_p_bar,
    )

    ## create new live point bugject to constraint

    new_L = -np.inf
    while new_L <= logLs[idx]:
        new_L = -rng.uniform(-10, 10) ** 2 / 2 - np.log(2 * np.pi) / 2
    logLs[idx] = new_L

    ## this live point needs to be assigned a cluster proportional to volume
    cluster[idx] = rng.choice(
        np.arange(max(cluster) + 1),
        p=[x / np.exp(logsumexp(logX_p)) for x in np.exp(logX_p)],
    )

    ns = [sum([q == r for r in cluster]) for q in np.unique(cluster)]
    return (
        cluster,
        logLs,
        logX_p,
        logZ,
        logZ_p,
        logX_p_bar,
        logX_p_X_q_bar,
        logZ_bar,
        logZ2_bar,
        logZ_p_bar,
        logZ2_p_bar,
        logZ_X_p_bar,
        logZ_p_X_p_bar,
        ns,
    )
