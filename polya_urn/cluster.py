import numpy as np

rng = np.random.default_rng()


def cluster_errors(
    p,
    n_p,
    n_new,
    rti,
):
    n = n_p + n_new
    # C23
    rti.logX_p_bar = np.append(rti.logX_p_bar, rti.logX_p_bar[p] + np.log(n_new / n))
    rti.logX_p_bar[p] += np.log(n_p / n)
    # C24/25
    logX_p_X_q_bar_temp = np.zeros(
        (rti.logX_p_X_q_bar.shape[0] + 1, rti.logX_p_X_q_bar.shape[1] + 1)
    )
    logX_p_X_q_bar_temp[:-1, :-1] = rti.logX_p_X_q_bar
    rti.logX_p_X_q_bar = logX_p_X_q_bar_temp
    rti.logX_p_X_q_bar[p, -1] = (
        np.log(n_p * n_new / (n * (n + 1))) + rti.logX_p_X_q_bar[p, p]
    )
    rti.logX_p_X_q_bar[-1, p] = rti.logX_p_X_q_bar[p, -1]
    rti.logX_p_X_q_bar[-1, -1] = rti.logX_p_X_q_bar[p, p] + np.log(
        n_new * (n_new + 1) / (n * (n + 1))
    )

    #### LOOK CAREFULLY AT THIS!!!!!!!!!!!!!!

    rti.logZ_X_p_bar = np.append(
        rti.logZ_X_p_bar, rti.logZ_X_p_bar[p] + np.log(n_new / n)
    )
    rti.logZ_X_p_bar[p] += np.log(n_p / n)
    # logZ_p_X_p_bar.append(logZ_p_X_p_bar[p] + np.log(n_new / n))
    # logZ_p_X_p_bar[p] += np.log(n_p / n)

    for q in np.unique(rti.cluster)[:-1]:  # omit new cluster
        if q != p:  # and omit broken cluster
            print("don't think I should ever see this with only two clusters")
            rti.logX_p_X_q_bar[q, -1] = rti.logX_p_X_q_bar[q, p] + np.log(n_new / n)
            rti.logX_p_X_q_bar[-1, q] = rti.logX_p_X_q_bar[q, -1]
            rti.logX_p_X_q_bar[q, p] += np.log(n_p / n)
            rti.logX_p_X_q_bar[p, q] = rti.logX_p_X_q_bar[q, p]

    # I think everything which depends on the original XpXp is done
    rti.logX_p_X_q_bar[p, p] += np.log(n_p * (n_p + 1) / (n * (n + 1)))

    rti.logZ_p_bar = np.append(rti.logZ_p_bar, rti.logZ_p_bar[p] + np.log(n_new / n))
    rti.logZ_p_bar[p] += np.log(n_p / n)
    rti.logZ_p_X_p_bar = np.append(
        rti.logZ_p_X_p_bar,
        rti.logZ_p_X_p_bar[p] + np.log(n_new * (n_new + 1) / (n * (n + 1))),
    )
    rti.logZ_p_X_p_bar[p] += np.log(n_p * (n_p + 1) / (n * (n + 1)))
    rti.logZ2_p_bar = np.append(
        rti.logZ2_p_bar,
        rti.logZ2_p_bar[p] + np.log(n_new * (n_new + 1) / (n * (n + 1))),
    )
    rti.logZ2_p_bar[p] += np.log(n_p * (n_p + 1) / (n * (n + 1)))
    return rti


## dividing a cluster - let's start with a cluster splitting in two


def clustering(rti):
    ## choose a cluster at random
    p = int(rng.random() * max(rti.cluster))
    ## index of new cluster for safe keeping
    new_cluster_idx = max(rti.cluster) + 1
    ## split points into the two clusters 50:50
    for i in range(len(rti.cluster)):
        if p == rti.cluster[i]:
            if rng.random() >= 0.5:
                # assign to the new cluster
                rti.cluster[i] = new_cluster_idx

    n_p = sum([q == p for q in rti.cluster])
    n_new = sum([q == new_cluster_idx for q in rti.cluster])
    # generate fraction of volume to go in each cluster
    frac_p, frac_new = rng.dirichlet([n_p, n_new])
    rti.logX_p = np.append(rti.logX_p, np.log(frac_new) + rti.logX_p[p])
    rti.logX_p[p] = np.log(frac_p) + rti.logX_p[p]
    rti.logZ_p = np.append(rti.logZ_p, rti.logZ_p[p] + np.log(n_new / (n_p + n_new)))
    rti.logZ_p[p] += np.log(n_p / (n_p + n_new))

    rti = cluster_errors(p, n_p, n_new, rti)
    return rti
