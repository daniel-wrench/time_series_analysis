# Testing fbm code

# from Kea.Kea.simulator import fbm as fbm
# from Kea.Kea.statistics.spectra import per_spectra as ps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import src.sf_funcs as sf
import random as random
from matplotlib.patches import Rectangle


def get_lag_vals_list(df):
    lag_vals_wide = pd.DataFrame(df["sq_diffs"].tolist(), index=df.index)
    lag_vals_wide.reset_index(inplace=True)  # Make the index a column
    lag_vals_wide.rename(columns={"index": "lag"}, inplace=True)
    lag_vals = pd.melt(
        lag_vals_wide, id_vars=["lag"], var_name="index", value_name="sq_diffs"
    )
    return lag_vals


# Make plt params use latex
plt.rcParams.update(
    {
        "text.usetex": True,
        "mathtext.fontset": "stix",  # Set the font to use for math
        "font.family": "serif",  # Set the default font family
        "font.size": 11,
    }
)

##################################

# from fbm import FBM

# Two ways of creating fBm

# 1. Specifying Hurst parameter, H

# fbm = FBM(n=N, hurst=H, length=L, method="daviesharte")
# X = fbm.fbm()

# 2. Specifying power-law/s, alphas

# N = 10000
# D = 1
# L = N
# dk = 2.0 * np.pi / L
# BREAK = 100.0 * dk

random.seed(42)
# x = fbm.create_fbm(grid_dims=(1000,), phys_dims=(1000,), alphas=(-1,-5/3), breaks=(100,), )
# x_periodic = fbm.create_fbm(
#     grid_dims=[4 * N for _ in range(D)],
#     phys_dims=[L for _ in range(D)],
#     alphas=(-1.0, -5.0 / 3.0),
#     gfunc="smooth_pow",
#     func_kwargs={"breaks": (BREAK,), "delta": 0.25},
# )

# timestamp = "20160101"
# fbm_field = pd.Series(
#     x_periodic[:N], name="fbm", index=pd.date_range(timestamp, periods=N, freq="S")
# )

# fbm_field.to_pickle("data/processed/wind/fbm_field_" + timestamp + ".pkl")

##################################

# dataset_name = "PSP\ solar\ wind"


##################################

# # Remove slashes from dataset name for saving
# dataset_name_save = dataset_name.replace("\\", "")

# plt.plot(x_periodic, label="Periodic fBM field")
# plt.plot(x, label="$\\mathbf{%s}$" % dataset_name)
# plt.legend()
# # plt.savefig(f"plots/{dataset_name_save}_time_series.png")
# plt.show()

# # Compute the spectra
# for data in [x_periodic, x]:
#     # for data in [x]:
#     spectrum = ps.modal_spectrum(ar1=data, phys_dims=[L for _ in range(D)])
#     plt.plot(spectrum[0][0], spectrum[1])
# # plt.plot(np.arange(len(x_subset)), np.arange(len(x_subset)) ** (-5/3), color="black", linestyle="--")
# plt.axvline(BREAK, color="black", linestyle="--", label="Break frequency")
# plt.semilogx()
# plt.semilogy()
# plt.xlim(spectrum[0][0].min(), spectrum[0][0].max())
# plt.ylim(1e-5, 1e3)
# plt.legend()
# plt.title("$\\mathbf{%s}$: Power spectral density" % dataset_name)
# # plt.savefig(f"plots/{dataset_name_save}_psd.png")
# plt.show()

##########################################################


timestamp = "20160101"

wind_raw = pd.read_pickle("data/processed/wind/mfi/" + timestamp + ".pkl")
wind_resampled = wind_raw["Bx"].resample("0.1S").mean()
wind = wind_resampled[10000:20000]

psp_raw = pd.read_pickle("data/processed/psp/psp_fld_l2_mag_rtn_201811.pkl")
psp_resampled = psp_raw["B_R"].resample("0.1S").mean()
psp = psp_resampled[:10000]

fbm = pd.read_pickle("data/processed/wind/fbm_field_" + timestamp + ".pkl")

data_dict = {
    "x": [psp, wind, fbm],
    "dt": [0.1, 0.1, 1],  # S
    "dataset_name": [
        "Solar\ wind\ from\ PSP\ spacecraft",
        "Solar\ wind\ from\ Wind\ spacecraft",
        "Multi-fractal\ fBm",
    ],
    "dataset_brief": [
        f"psp_B_R_{timestamp}",
        f"wind_Bx_{timestamp}",
        f"_fbm_{timestamp}",
    ],
}


# Loop over dictionary
for x, dataset_name, dataset_brief, dt in zip(
    data_dict["x"],
    data_dict["dataset_name"],
    data_dict["dataset_brief"],
    data_dict["dt"],
):
    print(f"Processing {dataset_name}...")

    # Compute the structure function
    lags = np.arange(1, 0.25 * len(x))
    lag_times = lags * dt
    powers = [0.5, 2]

    good_output = sf.compute_sf(pd.DataFrame(x), lags, powers, retain_increments=True)

    good_output["ch_numerator"] = good_output["0.5_mean"] ** 4
    good_output["ch"] = good_output["ch_numerator"] / (
        0.457 + (0.494 / good_output["n"]) + (0.045 / good_output["n"] ** 2)
    )
    # Final part of original correction is so small as to be neglibile, hence why often left out
    # Down-weights outliers - recommended to use both
    good_output["dowd"] = (good_output["mapd"] ** 2) * 2.198

    # lag_vals = get_lag_vals_list(good_output)

    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.scatter(lag_vals["lag"], lag_vals["sq_diffs"], alpha=0.005, s=0.5, c="black")
    ax.plot(lag_times, good_output["classical"], label="Classical", lw=3)
    ax.plot(lag_times, good_output["ch"], label="Cressie-Hawkins", lw=1.5)
    ax.plot(lag_times, good_output["dowd"], label="Dowd", lw=1.5)
    ax.axhline(y=2 * np.var(x), color="pink", linestyle=":", label=r"$2\sigma^2$")
    # Plot a powerlaw
    ax.plot(
        lag_times,
        good_output["classical"].min() * 10 * good_output["lag"] ** (2 / 3),
        color="purple",
        linestyle="--",
        label=r"$\tau^{2/3}$ power-law",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(good_output["classical"].min() / 2, good_output["classical"].max() * 5)
    # ax.set_xlim(lag_times[0] * 0.9, lag_times[-1] * 1.1)
    ax.set_title("$\\mathbf{%s}$: Various estimators of SF" % dataset_name)
    ax.set_xlabel(r"Lag ($\tau$)")
    ax.set_ylabel(r"$S_{2}(\tau)$")

    lags_to_examine = [5, 50, 500]
    # Draw boxes at values on x-axis
    for lag in lags_to_examine:
        lag = lag * dt
        rect = Rectangle(
            (lag - 0.05 * lag, ax.get_ylim()[0]),
            lag / 10,
            ax.get_ylim()[1] - ax.get_ylim()[0],
            fill=False,
            edgecolor="red",
            linestyle="--",
            alpha=0.8,
        )  # change color and linestyle as needed
        ax.add_patch(rect)

    plt.legend(loc="upper left")
    plt.savefig(f"plots/{dataset_brief}_sf_estimators.png")

    # Calculate the increments of x and plot the histogram
    fig, ax = plt.subplots(
        3,
        len(lags_to_examine),
        figsize=(9, 9),
        gridspec_kw={"hspace": 0.3, "wspace": 0.05},
    )
    fig.suptitle(
        r"$\mathbf{%s}$: $\tau$-scattergrams and PDFs of selected lags" % dataset_name,
        fontsize=14,
    )

    x_series = pd.Series(x)  # to use the diff method, very different from np.diff

    for i, lag in enumerate(lags_to_examine):
        x_series_lagged = x_series.shift(lag)
        increments = x_series - x_series_lagged  # same as x_series.diff(lag)
        ax[0, i].set_title(r"$\mathbf{\tau={%s}}$" % lag, fontsize=20, color="red")
        ax[0, i].scatter(x_series, x_series_lagged, s=1, alpha=0.2)
        ax[0, i].annotate(
            f"$r$ ={x_series.corr(x_series_lagged):.2f}",
            (0.05, 0.9),
            xycoords="axes fraction",
            fontsize=9,
        )

        ax[1, i].hist(increments, bins=50, alpha=0.5)

        # Annotate the values of the first four moments
        ax[1, i].annotate(
            rf"$\mu$={increments.mean():.2f}",
            (0.05, 0.9),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="white",
                alpha=0.5,
                pad=0.2,
            ),
        )
        ax[1, i].annotate(
            rf"$\sigma^2$={increments.var():.2f}",
            (0.05, 0.8),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="white",
                alpha=0.5,
                pad=0.2,
            ),
        )
        ax[1, i].annotate(
            f"Skew={increments.skew():.2f}",
            (0.05, 0.7),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="white",
                alpha=0.5,
                pad=0.2,
            ),
        )
        ax[1, i].annotate(
            f"Kurt={(increments.kurtosis()+3):.2f}",  # Add 3 to get actual kurtosis
            (0.05, 0.6),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="white",
                alpha=0.5,
                pad=0.2,
            ),
        )

        ax[2, i].hist((increments**2), bins=50, color="black", alpha=0.5)

        # Annotate the values of the first four moments
        ax[2, i].annotate(
            rf"$\mu$={(increments**2).mean():.2f}",
            (0.6, 0.9),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="white",
                alpha=0.5,
                pad=0.2,
            ),
        )
        ax[2, i].annotate(
            rf"$\sigma^2$={(increments**2).var():.2f}",
            (0.6, 0.8),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="white",
                alpha=0.5,
                pad=0.2,
            ),
        )
        ax[2, i].annotate(
            f"Skew={(increments**2).skew():.2f}",
            (0.6, 0.7),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="white",
                alpha=0.5,
                pad=0.2,
            ),
        )
        ax[2, i].annotate(
            f"Kurt={((increments**2).kurtosis()+3):.2f}",  # Add 3 to get actual kurtosis
            (0.6, 0.6),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="white",
                alpha=0.5,
                pad=0.2,
            ),
        )

        ax[0, i].set_xlabel(r"$x(t)$")
        ax[0, i].set_ylabel(r"$x(t+\tau)$")
        ax[0, i].set_xlim([x_series.min(), x_series.max()])
        ax[0, i].set_ylim([x_series.min(), x_series.max()])

        ax[1, i].set_xlabel(r"$x(t)-x(t+\tau)$")
        ax[2, i].set_xlabel(r"$(x(t)-x(t+\tau))^2$")
        ax[1, i].set_ylabel("Frequency")
        ax[2, i].set_ylabel("Frequency")

        # Add vertical lines corresponding to the mean and median
        ax[1, i].axvline(increments.mean(), color="black", linestyle="-", label="Mean")
        ax[1, i].axvline(
            increments.median(), color="black", linestyle="--", label="Median"
        )

        ax[1, i].set_yscale("log")
        # ax[2, i].set_yscale("log")
        # ax[2, i].set_xscale("log")
        ax[1, i].set_ylim(1, 1e5)

        # ax[1, i].set_xlim(-3, 3)
        # Overlay an "equivalent Gaussian" (same mean and variance)
        x_vals = np.linspace(increments.min(), increments.max(), 100)
        y_vals = (
            1
            / np.sqrt(2 * np.pi * increments.var())  # type: ignore
            * np.exp(-0.5 * (x_vals - increments.mean()) ** 2 / increments.var())  # type: ignore
        )
        ax[1, i].plot(
            x_vals,
            y_vals * len(increments) * 0.1,
            color="black",
            linestyle=":",
            label=r"$N(\mu, \sigma^2)$",
        )
        ax[1, 0].legend(loc="upper right", fontsize=9, edgecolor="white")

        if i != 0:
            ax[0, i].set_yticklabels([])
            ax[1, i].set_yticklabels([])
            ax[2, i].set_yticklabels([])
            ax[0, i].set_ylabel("")
            ax[1, i].set_ylabel("")
            ax[2, i].set_ylabel("")

    plt.savefig(f"plots/{dataset_brief}_stats_lags.png")

print("Done!")
