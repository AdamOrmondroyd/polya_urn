"""
Doing cluster splitting independently of PolyChord.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()


# %%
## kill lowest liklihood point
def kill(cluster, Ls, X_p, Z, Z_p):
    idx = np.argmin(Ls)
    p = cluster[idx]
    t = rng.power(sum([q == p for q in cluster]))
    Z = Z + (1 - t) * X_p[p] * Ls[idx]
    Z_p[p] = Z_p[p] + (1 - t) * X_p[p] * Ls[idx]
    X_p[p] = t * X_p[p]

    ## create new live point bugject to constraint

    new_L = 0
    while new_L <= Ls[idx]:
        new_L = np.exp(-rng.uniform(-10, 10) ** 2 / 2) / np.sqrt(2 * np.pi) * 20
    Ls[idx] = new_L

    ## this live point needs to be assigned a cluster proportional to volume
    cluster[idx] = rng.choice(
        np.arange(max(cluster) + 1), p=[x / sum(X_p) for x in X_p]
    )

    ns = [sum([q == r for r in cluster]) for q in np.unique(cluster)]
    return cluster, Ls, X_p, Z, Z_p, ns


## dividing a cluster - let's start with a cluster splitting in two


def clustering(cluster, X_p, Z_p):
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
    return cluster, X_p, Z_p


# %%
## initialise
def initialise():
    nlive = 1000
    cluster = [0] * nlive
    Ls = list(
        np.exp(-rng.uniform(-10, 10, size=nlive) ** 2 / 2) / np.sqrt(2 * np.pi) * 20
    )
    Z = 0
    Z_p = [Z]
    X = 1
    X_p = [X]
    return (
        nlive,
        cluster,
        Ls,
        X,
        X_p,
        Z,
        Z_p,
    )


# %%
def simulation():
    (
        nlive,
        cluster,
        Ls,
        X,
        X_p,
        Z,
        Z_p,
    ) = initialise()

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.flatten()

    kills_between_clusterings = 10000
    num_clusterings = 1

    colors = ["k", "c", "m"]
    for ii in range(num_clusterings):
        print(ii)
        for i in range(kills_between_clusterings + 1):
            print(i)
            cluster, Ls, X_p, Z, Z_p, ns = kill(cluster, Ls, X_p, Z, Z_p)

            # plot every 10
            if 0 == i % 10:
                print(ns)
                print(X_p)
                print(Z_p)
                for n, x, z, color in zip(ns, X_p, Z_p, colors):
                    print("here")
                    print(n)
                    print(x)
                    print(z)
                    ax[0].scatter(
                        [ii * kills_between_clusterings + i],
                        [n],
                        marker="+",
                        color=color,
                        s=0.1,
                    )
                    ax[1].scatter(
                        [ii * kills_between_clusterings + i],
                        [x],
                        marker="+",
                        color=color,
                        s=0.1,
                    )

                    ax[2].scatter(
                        [ii * kills_between_clusterings + i],
                        [z],
                        marker="+",
                        color=color,
                        s=0.1,
                    )

                ax[3].scatter([ii * kills_between_clusterings + i], Z, color="k", s=0.1)

        # don't cluster at the end
        if ii < num_clusterings - 1:
            cluster, X_p, Z_p = clustering(cluster, X_p, Z_p)
            ax[0].vlines((ii + 1) * kills_between_clusterings, 0, nlive)
            ax[-1].vlines((ii + 1) * kills_between_clusterings, 0, 1)
        for a, title in zip(ax, ["n_p", "X_p", "Z_p", "Z"]):
            if "Z" == title:
                title += f" = {sum(Z_p)}"
            a.set(title=title)
        ax[1].set(yscale="log")

    fig.tight_layout()
    return fig, ax


# %%
# do 20 simulations
for i in range(1):
    fig, ax = simulation()
    fig.savefig(f"simulation_{i}.png", dpi=1200)
    plt.close()
    # plt.show()

# %%
