"""
Killing live points and propagating errors
"""

import numpy as np

rng = np.random.default_rng()
## update errors at kill
def kill_errors(
    p,
    n_p,
    cluster,
    L_p,
    X_p_bar,
    X_p_X_q_bar,
    Z_bar,
    Z2_bar,
    Z_p_bar,
    Z2_p_bar,
    Z_X_p_bar,
    Z_p_X_p_bar,
):
    Z_bar += X_p_bar[p] * L_p[p] / (n_p + 1)
    Z_p_bar[p] += X_p_bar[p] * L_p[p] / (n_p + 1)
    X_p_bar[p] *= n_p * X_p_bar[p] / (n_p + 1)
    Z2_bar += 2 * Z_X_p_bar[p] * L_p[p] / (n_p + 1) + X_p_X_q_bar[p, p] * L_p[
        p
    ] ** 2 / ((n_p + 1) * (n_p + 2))
    Z2_p_bar[p] += 2 * Z_p_X_p_bar[p] * L_p[p] / (n_p + 1) + 2 * X_p_X_q_bar[
        p, p
    ] * L_p[p] ** 2 / ((n_p + 1) * (n_p + 2))
    Z_X_p_bar[p] = n_p * Z_X_p_bar[p] / (n_p + 1) + n_p * X_p_X_q_bar[p, p] * L_p[
        p
    ] ** 2 / ((n_p + 1) * (n_p + 2))
    for q in np.unique(cluster):
        if q != p:
            Z_X_p_bar[q] += X_p_X_q_bar[p, q] * L_p[p] / (n_p + 1)

    Z_p_X_p_bar[p] = n_p * Z_p_X_p_bar[p] / (n_p + 1) + n_p * X_p_X_q_bar[p, p] * L_p[
        p
    ] / ((n_p + 1) * (n_p + 2))
    X_p_X_q_bar[p, p] *= n_p / (n_p + 2)
    for q in np.unique(cluster):
        if q != p:
            X_p_X_q_bar[p, q] *= n_p / (n_p + 1)
            X_p_X_q_bar[q, p] = X_p_X_q_bar[p, q]

    return (
        X_p_bar,
        X_p_X_q_bar,
        Z_bar,
        Z2_bar,
        Z_p_bar,
        Z2_p_bar,
        Z_X_p_bar,
        Z_p_X_p_bar,
    )


## kill lowest liklihood point
def kill(
    cluster,
    L_p,
    X_p,
    Z,
    Z_p,
    X_p_bar,
    X_p_X_q_bar,
    Z_bar,
    Z_p_bar,
    Z2_bar,
    Z2_p_bar,
    Z_X_p_bar,
    Z_p_X_p_bar,
):
    idx = np.argmin(L_p)
    p = cluster[idx]
    t = rng.power(sum([q == p for q in cluster]))
    Z = Z + (1 - t) * X_p[p] * L_p[idx]
    Z_p[p] = Z_p[p] + (1 - t) * X_p[p] * L_p[idx]
    X_p[p] = t * X_p[p]

    n_p = sum([q == p for q in cluster])
    (
        X_p_bar,
        X_p_X_q_bar,
        Z_bar,
        Z2_bar,
        Z_p_bar,
        Z2_p_bar,
        Z_X_p_bar,
        Z_p_X_p_bar,
    ) = kill_errors(
        p,
        n_p,
        cluster,
        L_p,
        X_p_bar,
        X_p_X_q_bar,
        Z_bar,
        Z_p_bar,
        Z2_bar,
        Z2_p_bar,
        Z_X_p_bar,
        Z_p_X_p_bar,
    )

    ## create new live point bugject to constraint

    new_L = 0
    while new_L <= L_p[idx]:
        new_L = np.exp(-rng.uniform(-10, 10) ** 2 / 2) / np.sqrt(2 * np.pi) * 20
    L_p[idx] = new_L

    ## this live point needs to be assigned a cluster proportional to volume
    cluster[idx] = rng.choice(
        np.arange(max(cluster) + 1), p=[x / sum(X_p) for x in X_p]
    )

    ns = [sum([q == r for r in cluster]) for q in np.unique(cluster)]
    return (
        cluster,
        L_p,
        X_p,
        Z,
        Z_p,
        X_p_bar,
        X_p_X_q_bar,
        Z_bar,
        Z2_bar,
        Z_p_bar,
        Z2_p_bar,
        Z_X_p_bar,
        Z_p_X_p_bar,
        ns,
    )
