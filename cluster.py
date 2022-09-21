import numpy as np

rng = np.random.default_rng()


def cluster_errors(
    p,
    n_p,
    n_new,
    cluster,
    logX_p_bar,
    logX_p_X_q_bar,
    logZ_p_bar,
    logZ2_p_bar,
    logZ_X_p_bar,
    logZ_p_X_p_bar,
):
    n = n_p + n_new
    # C23
    logX_p_bar.append(logX_p_bar[p] + np.log(n_new / n))
    logX_p_bar[p] += np.log(n_p / n)
    # C24/25
    logX_p_X_q_bar_temp = np.zeros(
        (logX_p_X_q_bar.shape[0] + 1, logX_p_X_q_bar.shape[1] + 1)
    )
    logX_p_X_q_bar_temp[:-1, :-1] = logX_p_X_q_bar
    logX_p_X_q_bar = logX_p_X_q_bar_temp
    logX_p_X_q_bar[p, -1] = np.log(n_p * n_new / (n * (n + 1))) + logX_p_X_q_bar[p, p]
    logX_p_X_q_bar[-1, p] = logX_p_X_q_bar[p, -1]
    logX_p_X_q_bar[-1, -1] = logX_p_X_q_bar[p, p] + np.log(
        n_new * (n_new + 1) / (n * (n + 1))
    )

    #### LOOK CAREFULLY AT THIS!!!!!!!!!!!!!!

    logZ_X_p_bar.append(logZ_X_p_bar[p] + np.log(n_new / n))
    logZ_X_p_bar[p] += np.log(n_p / n)
    # logZ_p_X_p_bar.append(logZ_p_X_p_bar[p] + np.log(n_new / n))
    # logZ_p_X_p_bar[p] += np.log(n_p / n)

    for q in np.unique(cluster)[:-1]:  # omit new cluster
        if q != p:  # and omit broken cluster
            print("don't think I should ever see this with only two clusters")
            logX_p_X_q_bar[q, -1] = logX_p_X_q_bar[q, p] + np.log(n_new / n)
            logX_p_X_q_bar[-1, q] = logX_p_X_q_bar[q, -1]
            logX_p_X_q_bar[q, p] += np.log(n_p / n)
            logX_p_X_q_bar[p, q] = logX_p_X_q_bar[q, p]

    # I think everything which depends on the original XpXp is done
    logX_p_X_q_bar[p, p] += np.log(n_p * (n_p + 1) / (n * (n + 1)))

    logZ_p_bar.append(logZ_p_bar[p] + np.log(n_new / n))
    logZ_p_bar[p] += np.log(n_p / n)
    logZ_p_X_p_bar.append(
        logZ_p_X_p_bar[p] + np.log(n_new * (n_new + 1) / (n * (n + 1)))
    )
    logZ_p_X_p_bar[p] += np.log(n_p * (n_p + 1) / (n * (n + 1)))
    logZ2_p_bar.append(logZ2_p_bar[p] + np.log(n_new * (n_new + 1) / (n * (n + 1))))
    logZ2_p_bar[p] += np.log(n_p * (n_p + 1) / (n * (n + 1)))
    return (
        logX_p_bar,
        logX_p_X_q_bar,
        logZ_p_bar,
        logZ2_p_bar,
        logZ_X_p_bar,
        logZ_p_X_p_bar,
    )


## dividing a cluster - let's start with a cluster splitting in two


def clustering(
    cluster,
    logX_p,
    logZ_p,
    logX_p_bar,
    logX_p_X_q_bar,
    logZ_p_bar,
    logZ2_p_bar,
    logZ_X_p_bar,
    logZ_p_X_p_bar,
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
    # generate fraction of volume to go in each cluster
    frac_p, frac_new = rng.dirichlet([n_p, n_new])
    logX_p.append(np.log(frac_new) + logX_p[p])
    logX_p[p] = np.log(frac_p) + logX_p[p]
    logZ_p.append(logZ_p[p] + np.log(n_new / (n_p + n_new)))
    logZ_p[p] += np.log(n_p / (n_p + n_new))

    (
        logX_p_bar,
        logX_p_X_q_bar,
        logZ_p_bar,
        logZ2_p_bar,
        logZ_X_p_bar,
        logZ_p_X_p_bar,
    ) = cluster_errors(
        p,
        n_p,
        n_new,
        cluster,
        logX_p_bar,
        logX_p_X_q_bar,
        logZ_p_bar,
        logZ2_p_bar,
        logZ_X_p_bar,
        logZ_p_X_p_bar,
    )
    return (
        cluster,
        logX_p,
        logZ_p,
        logX_p_bar,
        logX_p_X_q_bar,
        logZ_p_bar,
        logZ2_p_bar,
        logZ_X_p_bar,
        logZ_p_X_p_bar,
    )
