"""
Doing cluster splitting independently of PolyChord.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from cluster import clustering
from kill import kill


rng = np.random.default_rng()


# %%
## initialise
def initialise():
    nlive = 1000
    cluster = [0] * nlive
    logLs = list(-rng.uniform(-10, 10, size=nlive) ** 2 / 2 - np.log(2 * np.pi) / 2)
    logZ = -np.inf
    logZ_p = [logZ]
    logX = 0.0
    logX_p = [logX]
    logX_p_bar = [logX]
    logX_p_X_q_bar = np.array([[2 * logX]])
    logZ_bar = logZ
    logZ2_bar = 2 * logZ
    logZ_p_bar = [logZ]
    logZ2_p_bar = [2 * logZ]
    logZ_X_p_bar = [-np.inf]
    logZ_p_X_p_bar = [-np.inf]
    return (
        nlive,
        cluster,
        logLs,
        logX,
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
    )


# %%
def simulation(kills_between_clusterings):
    (
        nlive,
        cluster,
        logLs,
        logX,
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
    ) = initialise()

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.flatten()

    colors = ["k", "c", "m"]

    for ii, kills in enumerate(kills_between_clusterings):
        print(ii)
        for i in range(kills):
            print(i)
            (
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
            ) = kill(
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
            )

            # plot every 10
            if 0 == i % 100:
                print(ns)
                print(logX_p)
                print(logZ_p)
                for n, x, x_bar, x2_bar, z, z_bar, z2_bar, color in zip(
                    ns,
                    logX_p,
                    logX_p_bar,
                    np.diag(logX_p_X_q_bar),
                    logZ_p,
                    logZ_p_bar,
                    logZ2_p_bar,
                    colors,
                ):
                    print(n)
                    print(x)
                    print(z)
                    sigma_x = np.sqrt(np.exp(x2_bar) - np.exp(x_bar) ** 2)
                    sigma_z = np.sqrt(np.exp(z2_bar) - np.exp(z_bar) ** 2)
                    ax[0].scatter(
                        [sum(kills_between_clusterings[:ii]) + i],
                        [n],
                        marker="+",
                        color=color,
                        s=0.1,
                    )
                    ax[1].errorbar(
                        [sum(kills_between_clusterings[:ii]) + i],
                        [np.exp(x - logsumexp(logX_p))],
                        yerr=sigma_x / np.exp(logsumexp(logX_p)),
                        marker="+",
                        color=color,
                    )

                    ax[2].errorbar(
                        [sum(kills_between_clusterings[:ii]) + i],
                        [np.exp(z)],
                        yerr=sigma_z,
                        marker="+",
                        color=color,
                    )

                ax[3].errorbar(
                    [sum(kills_between_clusterings[:ii]) + i],
                    [np.exp(logZ)],
                    yerr=np.sqrt(np.exp(logZ2_bar) - np.exp(logZ_bar) ** 2),
                    marker="+",
                    color="k",
                )

            # don't cluster at the end
            print(
                f"Z = {np.exp(logZ)} ± {np.sqrt(np.exp(logZ2_bar) - np.exp(logZ_bar)**2)}"
            )
            # print Z for each cluster
            for iii, (Zi, Z2i_bar, Zi_bar2) in enumerate(
                zip(np.exp(logZ_p), np.exp(logZ2_p_bar), np.exp(logZ_p_bar) ** 2)
            ):
                print(f"Z{iii} = {Zi} ± {np.sqrt(Z2i_bar - Zi_bar2)}")

        if ii < len(kills_between_clusterings) - 1:
            print("clustering")
            (
                cluster,
                logX_p,
                logZ_p,
                logX_p_bar,
                logX_p_X_q_bar,
                logZ_p_bar,
                logZ2_p_bar,
                logZ_X_p_bar,
                logZ_p_X_p_bar,
            ) = clustering(
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
            print(
                f"Z = {np.exp(logZ)} ± {np.sqrt(np.exp(logZ2_bar) - np.exp(logZ_bar)**2)}"
            )
            for iii, (Zi, Z2i_bar, Zi_bar2) in enumerate(
                zip(np.exp(logZ_p), np.exp(logZ2_p_bar), np.exp(logZ_p_bar) ** 2)
            ):
                print(f"Z{iii} = {Zi} ± {np.sqrt(Z2i_bar - Zi_bar2)}")
            ax[0].vlines(sum(kills_between_clusterings[: ii + 1]), 0, nlive)
    print(f"Z = {np.exp(logZ)} ± {np.sqrt(np.exp(logZ2_bar) - np.exp(logZ_bar)**2)}")

    for a, title in zip(ax, ["n_p", "X_p/X", "Z_p", "Z"]):
        if "Z" == title:
            title += f" = {np.exp(logZ):.2E} ± {np.sqrt(np.exp(logZ2_bar) - np.exp(logZ_bar)**2):.2E}"
        a.set(title=title)

    print(f"Z = {np.exp(logZ)} ± {np.sqrt(np.exp(logZ2_bar) - np.exp(logZ_bar)**2)}")

    assert np.isclose(logZ, logsumexp(logZ_p))
    fig.tight_layout()
    return fig, ax


# %%
# do 20 simulations
for i in range(1):
    # fig, ax = simulation(kills_between_clusterings=[1000, 100])
    fig, ax = simulation(kills_between_clusterings=[1000, 9000])
    # fig, ax = simulation(kills_between_clusterings=[2, 2, 100])
    # fig, ax = simulation(kills_between_clusterings=[10000])
    fig.savefig(f"simulation_{i}.png", dpi=1200)
    plt.close()
    # plt.show()

# %%
