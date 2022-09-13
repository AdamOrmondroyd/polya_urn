"""
Doing cluster splitting independently of PolyChord.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from cluster import clustering
from kill import kill


rng = np.random.default_rng()


# %%
## initialise
def initialise():
    nlive = 1000
    cluster = [0] * nlive
    L_p = list(
        np.exp(-rng.uniform(-10, 10, size=nlive) ** 2 / 2) / np.sqrt(2 * np.pi) * 20
    )
    Z = 0
    Z_p = [Z]
    X = 1
    X_p = [X]
    X_p_bar = [X]
    X_p_X_q_bar = np.array([[X**2]])
    Z_bar = 0
    Z2_bar = 0
    Z_p_bar = [0]
    Z2_p_bar = [0]
    Z_X_p_bar = [0]
    Z_p_X_p_bar = [0]
    return (
        nlive,
        cluster,
        L_p,
        X,
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
    )


# %%
def simulation():
    (
        nlive,
        cluster,
        L_p,
        X,
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
    ) = initialise()

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.flatten()

    kills_between_clusterings = [1000, 9000]

    colors = ["k", "c", "m"]

    for ii, kills in enumerate(kills_between_clusterings):
        print(ii)
        for i in range(kills):
            print(i)
            (
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
            ) = kill(
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
            )

            # plot every 10
            if 0 == i % 10:
                print(ns)
                print(X_p)
                print(Z_p)
                for n, x, x_bar, x2_bar, z, z_bar, z2_bar, color in zip(
                    ns,
                    X_p,
                    X_p_bar,
                    np.diag(X_p_X_q_bar),
                    Z_p,
                    Z_p_bar,
                    Z2_p_bar,
                    colors,
                ):
                    print(n)
                    print(x)
                    print(z)
                    sigma_x = np.sqrt(x2_bar - x_bar**2)
                    sigma_z = np.sqrt(z2_bar - z_bar**2)
                    ax[0].scatter(
                        [sum(kills_between_clusterings[:ii]) + i],
                        [n],
                        marker="+",
                        color=color,
                        s=0.1,
                    )
                    ax[1].errorbar(
                        [sum(kills_between_clusterings[:ii]) + i],
                        [x / np.sum(X_p)],
                        yerr=sigma_x / np.sum(X_p),
                        marker="+",
                        color=color,
                        # s=0.1,
                    )

                    ax[2].errorbar(
                        [sum(kills_between_clusterings[:ii]) + i],
                        [z],
                        yerr=sigma_z,
                        marker="+",
                        color=color,
                        # s=0.1,
                    )

                ax[3].errorbar(
                    [sum(kills_between_clusterings[:ii]) + i],
                    [Z],
                    yerr=np.sqrt(Z2_bar - Z_bar**2),
                    color="k",
                    # markersize=0.1,
                )

        # don't cluster at the end
        if ii < len(kills_between_clusterings) - 1:
            (
                cluster,
                X_p,
                Z_p,
                X_p_bar,
                X_p_X_q_bar,
                Z_p_bar,
                Z2_p_bar,
                Z_X_p_bar,
                Z_p_X_p_bar,
            ) = clustering(
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
            ax[0].vlines(sum(kills_between_clusterings[: ii + 1]), 0, nlive)
        for a, title in zip(ax, ["n_p", "X_p/X", "Z_p", "Z"]):
            if "Z" == title:
                title += f" = {Z:.2E} ± {np.sqrt(Z2_bar - Z_bar**2):.2E}"
            a.set(title=title)
        # ax[1].set(yscale="log")
    print(f"Z = {Z} ± {np.sqrt(Z2_bar-Z_bar**2)}")

    assert np.isclose(Z, sum(Z_p))
    fig.tight_layout()
    return fig, ax


# %%
# do 20 simulations
for i in range(20):
    fig, ax = simulation()
    fig.savefig(f"simulation_{i}.png", dpi=1200)
    plt.close()
    # plt.show()

# %%
