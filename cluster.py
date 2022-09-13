import numpy as np

rng = np.random.default_rng()


def cluster_errors(
    p,
    n_p,
    n_new,
    cluster,
    X_p_bar,
    X_p_X_q_bar,
    Z_p_bar,
    Z2_p_bar,
    Z_X_p_bar,
    Z_p_X_p_bar,
):
    n = n_p + n_new
    X_p_bar.append(X_p_bar[p] * n_new / n)
    X_p_bar[p] *= n_p / n
    X_p_X_q_bar_new = np.zeros((X_p_X_q_bar.shape[0] + 1, X_p_X_q_bar.shape[1] + 1))
    X_p_X_q_bar_new[:-1, :-1] = X_p_X_q_bar
    X_p_X_q_bar = X_p_X_q_bar_new
    X_p_X_q_bar[p, -1] = n_p * n_new / (n * (n + 1)) * X_p_X_q_bar[p, p]
    X_p_X_q_bar[-1, p] = X_p_X_q_bar[p, -1]
    X_p_X_q_bar[p, p] *= n_p * (n_p + 1) / (n * (n + 1))
    X_p_X_q_bar[-1, -1] *= n_new * (n_new + 1) / (n * (n + 1))

    Z_X_p_bar.append(Z_X_p_bar[p] * n_new / n)
    Z_X_p_bar[p] *= n_p / n
    Z_p_X_p_bar.append(Z_p_X_p_bar[p] * n_new / n)
    Z_p_X_p_bar[p] *= n_p / n
    Z_X_p_bar.append(Z_X_p_bar[p] * n_new / n)

    for q in np.unique(cluster)[:-1]:  # omit new cluster
        if q != p:
            X_p_X_q_bar[q, -1] = X_p_X_q_bar[q, p] * n_new / n
            X_p_X_q_bar[-1, q] = X_p_X_q_bar[q, -1]
            X_p_X_q_bar[q, p] *= n_p / n
            X_p_X_q_bar[p, q] = X_p_X_q_bar[q, p]

    Z_p_bar.append(Z_p_bar[p] * n_new / n)
    Z_p_bar[p] *= n_p / n
    Z_p_X_p_bar.append(Z_p_X_p_bar[p] * n_new * (n_new + 1) / (n * (n + 1)))
    Z_p_X_p_bar[p] *= n_p * (n_p + 1) / (n * (n + 1))
    Z2_p_bar.append(Z2_p_bar[p] * n_new * (n_new + 1) / (n * (n + 1)))
    Z2_p_bar[p] *= n_p * (n_p + 1) / (n * (n + 1))
    return X_p_bar, X_p_X_q_bar, Z_p_bar, Z2_p_bar, Z_X_p_bar, Z_p_X_p_bar


## dividing a cluster - let's start with a cluster splitting in two


def clustering(
    cluster,
    X_p,
    Z_p,
    X_p_bar,
    X_p_X_q_bar,
    Z_p_bar,
    Z2_p_bar,
    Z_X_p_bar,
    Z_p_X_p_bar,
):
    ## choose a cluster at random
    p = int(rng.random() * max(cluster))
    ## index of new cluster for safe keeping
    new_cluster_idx = max(cluster) + 1
    ## split points into the two clusters 50:50
    for i in range(len(cluster)):
        if p == cluster[i]:
            if rng.random() >= 0.5:
                # assign to the new cluster
                cluster[i] = new_cluster_idx

    n_p = sum([q == p for q in cluster])
    n_new = sum([q == new_cluster_idx for q in cluster])
    X_0, X_1 = rng.dirichlet([n_p, n_new]) * X_p[p]
    X_p[p] = X_0
    X_p.append(X_1)
    Z_p.append(Z_p[p] * n_new / (n_p + n_new))
    Z_p[p] *= n_p / (n_p + n_new)

    X_p_bar, X_p_X_q_bar, Z_p_bar, Z2_p_bar, Z_X_p_bar, Z_p_X_p_bar = cluster_errors(
        p,
        n_p,
        n_new,
        cluster,
        X_p_bar,
        X_p_X_q_bar,
        Z_p_bar,
        Z2_p_bar,
        Z_X_p_bar,
        Z_p_X_p_bar,
    )
    return (
        cluster,
        X_p,
        Z_p,
        X_p_bar,
        X_p_X_q_bar,
        Z_p_bar,
        Z2_p_bar,
        Z_X_p_bar,
        Z_p_X_p_bar,
    )
