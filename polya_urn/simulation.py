"""
Doing cluster splitting independently of PolyChord.
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from polya_urn.rti import RTI

rng = np.random.default_rng()

# %%
def simulation(kills_between_clusterings):

    rti = RTI()
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    ax = ax.flatten()

    colors = ["k", "c", "m"]

    for iii, kills in enumerate(kills_between_clusterings):
        print(iii)
        for ii in range(kills):
            print(ii)

            # kill lowest likelihood point
            rti.kill()

            # plot every 10
            if 0 == ii % 100:
                print(rti.ns)

                for i, (n, x, sigma_x, z, sigma_z, color) in enumerate(
                    zip(
                        rti.ns,
                        rti.logX_p_bar,
                        rti.sigma_X_p,
                        rti.logZ_p_bar,
                        rti.sigma_Z_p,
                        colors,
                    )
                ):
                    print(f"X{i} = {np.exp(x)} ± {sigma_x}")
                    print(f"Z{i} = {np.exp(z)} ± {sigma_z}")

                    ax[0].scatter(
                        [sum(kills_between_clusterings[:iii]) + ii],
                        [n],
                        marker="+",
                        color=color,
                        s=0.1,
                    )
                    ax[1].errorbar(
                        [sum(kills_between_clusterings[:iii]) + ii],
                        [np.exp(x - rti.logX)],
                        yerr=sigma_x / rti.logX,
                        marker="+",
                        color=color,
                    )

                    ax[2].errorbar(
                        [sum(kills_between_clusterings[:iii]) + ii],
                        [np.exp(z)],
                        yerr=sigma_z,
                        marker="+",
                        color=color,
                    )

                ax[3].errorbar(
                    [sum(kills_between_clusterings[:iii]) + ii],
                    [np.exp(rti.logZ_bar)],
                    yerr=rti.sigma_Z,
                    marker="+",
                    color="k",
                )

            print(f"Z = {np.exp(rti.logZ_bar)} ± {rti.sigma_Z}")

        # don't cluster at the end
        if iii < len(kills_between_clusterings) - 1:
            print("clustering")
            rti.clustering()
            print(
                f"Z = {np.exp(rti.logZ_bar)} ± {np.sqrt(np.exp(rti.logZ2_bar) - np.exp(rti.logZ_bar)**2)}"
            )
            for iii, (Zi, sigma_Z_i) in enumerate(
                zip(np.exp(rti.logZ_p), rti.sigma_Z_p)
            ):
                print(f"Z{iii} = {Zi} ± {sigma_Z_i}")
            ax[0].vlines(sum(kills_between_clusterings[: iii + 1]), 0, rti.nlive)

    for a, title in zip(ax, ["n_p", "X_p/X", "Z_p", "Z"]):
        if "Z" == title:
            title += f" = {np.exp(rti.logZ_bar):.2E} ± {rti.sigma_Z:.2E}"
        a.set(title=title)

    print(
        f"Z = {np.exp(rti.logZ_bar)} ± {np.sqrt(np.exp(rti.logZ2_bar) - np.exp(rti.logZ_bar)**2)}"
    )
    print(f"logZ = {rti.logZ_bar} ± {rti.logZ2_bar - 2 * rti.logZ_bar}")

    assert np.isclose(rti.logZ, logsumexp(rti.logZ_p))
    fig.tight_layout()
    return fig, ax
