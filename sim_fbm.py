# Testing fbm code

from pyrsistent import b
from Kea.Kea.simulator import fbm as fbm
from Kea.Kea.statistics.spectra import per_spectra as ps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sf_funcs as sf
# from fbm import FBM

# Two ways of creating fBm

# 1. Specifying Hurst parameter, H

# fbm = FBM(n=N, hurst=H, length=L, method="daviesharte")
# X = fbm.fbm()

# 2. Specifying power-law/s, alphas

N = 1000
D = 1
L = N
dk = 2.0 * np.pi / L
BREAK = 100.0 * dk

# x = fbm.create_fbm(grid_dims=(1000,), phys_dims=(1000,), alphas=(-1,-5/3), breaks=(100,), )
x_periodic = fbm.create_fbm(
    grid_dims=[4 * N for _ in range(D)],
    phys_dims=[L for _ in range(D)],
    alphas=(-1.0, -5.0 / 3.0),
    gfunc="smooth_pow",
    func_kwargs={"breaks": (BREAK,), "delta": 0.25},
)

dataset_name = "Multi-fractal\ fBm"
x = x_periodic[:N]

# dataset_name = "PSP\ solar\ wind"

# df_raw_full = pd.read_pickle("data/processed/psp/psp_fld_l2_mag_rtn_201811.pkl")
# df_raw = df_raw_full["B_R"]
# df_raw.head()
# x = df_raw.values[:1000]

# Remove slashes from dataset name for saving
dataset_name_save = dataset_name.replace("\\", "")

plt.plot(x_periodic, label="Periodic fBM field")
plt.plot(x, label=rf"$\bf{dataset_name}$")
plt.legend()
plt.savefig(f"plots/{dataset_name_save}_time_series.png")
plt.show()

# Compute the spectra
for data in [x_periodic, x]:
    # for data in [x]:
    spectrum = ps.modal_spectrum(ar1=data, phys_dims=[L for _ in range(D)])
    plt.plot(spectrum[0][0], spectrum[1])
# plt.plot(np.arange(len(x_subset)), np.arange(len(x_subset)) ** (-5/3), color="black", linestyle="--")
plt.axvline(BREAK, color="black", linestyle="--", label="Break frequency")
plt.semilogx()
plt.semilogy()
plt.xlim(spectrum[0][0].min(), spectrum[0][0].max())
plt.ylim(1e-5, 1e3)
plt.legend()
plt.title(rf"$\bf{dataset_name}:$ Power spectral density")
plt.savefig(f"plots/{dataset_name_save}_psd.png")
plt.show()

# Compute the structure function
lags = np.arange(1, len(x))
powers = [0.5, 2]

good_output = sf.compute_sf(pd.DataFrame(x), lags, powers, retain_increments=True)
good_output["ch_numerator"] = good_output["0.5_mean"] ** 4
good_output["ch"] = good_output["ch_numerator"] / (
    0.457 + (0.494 / good_output["n"]) + (0.045 / good_output["n"] ** 2)
)
# Final part of original correction is so small as to be neglibile, hence why often left out
# Down-weights outliers - recommended to use both
good_output["dowd"] = (good_output["mapd"] ** 2) * 2.198

# plt.plot(good_output["ch_numerator"], label="CH numerator")
plt.plot(good_output["ch"], label="Cressie-Hawkins")
plt.plot(good_output["dowd"], label="Dowd")
plt.plot(good_output["sosf"], label="Method-of-moments (standard)")
plt.axhline(y=2 * np.var(x), color="black", linestyle=":", label=r"$2\sigma^2$")
# Plot a powerlaw
plt.plot(
    good_output["n"],
    0.2 * good_output["n"] ** (2 / 3),
    color="black",
    linestyle="--",
    label=r"$\tau^{2/3}$ power-law",
)
plt.semilogx()
plt.semilogy()
# plt.ylim(0.08, 20)
plt.title(rf"$\bf{dataset_name}:$ Various estimators of SF")
plt.xlabel(r"Lag $\tau$")
plt.ylabel(r"$S_{2}(\tau)$")
plt.legend()
plt.savefig(f"plots/{dataset_name_save}_sf_estimators.png")
plt.show()

# Calculate the increments of x and plot the histogram
fig, ax = plt.subplots(
    2,
    4,
    figsize=(14, 7),
    tight_layout=True,
    gridspec_kw={"hspace": 0.3, "wspace": 0.05},
)
fig.suptitle(
    rf"$\bf{dataset_name}:$ $\tau$-scattergrams, PDFs and first 4 moments of increments $x(t)-x(t+\tau)$ at various lags $\tau$",
    fontsize=16,
)

x_series = pd.Series(x)  # to use the diff method, very different from np.diff

for i, lag in enumerate([1, 5, 50, 500]):
    x_series_lagged = x_series.shift(lag)
    increments = x_series - x_series_lagged  # same as x_series.diff(lag)
    ax[0, i].set_title(rf"$\tau={lag}$", fontsize=13)
    ax[0, i].scatter(x_series, x_series_lagged, s=1)
    ax[0, i].annotate(
        f"$r$ ={x_series.corr(x_series_lagged):.2f}",
        (0.05, 0.9),
        xycoords="axes fraction",
        fontsize=9,
    )

    ax[1, i].hist(increments, bins=30)
    # Add the mean, variance, and kurtosis as annotations
    ax[1, i].annotate(
        rf"$\mu$={increments.mean():.2f}",
        (0.05, 0.9),
        xycoords="axes fraction",
        fontsize=9,
    )
    ax[1, i].annotate(
        rf"$\sigma^2$={increments.var():.2f}",
        (0.05, 0.8),
        xycoords="axes fraction",
        fontsize=9,
    )
    ax[1, i].annotate(
        f"Skew={increments.skew():.2f}",
        (0.05, 0.7),
        xycoords="axes fraction",
        fontsize=9,
    )
    ax[1, i].annotate(
        f"Kurt={(increments.kurtosis()+3):.2f}",  # Add 3 to get actual kurtosis
        (0.05, 0.6),
        xycoords="axes fraction",
        fontsize=9,
        bbox=dict(
            boxstyle="round", facecolor="white", edgecolor="white", alpha=0.5, pad=0.2
        ),
    )

    ax[0, i].set_xlabel(r"$x(t)$")
    ax[0, i].set_ylabel(r"$x(t+\tau)$")
    ax[0, i].set_xlim([x_series.min(), x_series.max()])
    ax[0, i].set_ylim([x_series.min(), x_series.max()])

    ax[1, i].set_xlabel(r"$x(t)-x(t+\tau)$")
    ax[1, i].set_ylabel("Count")

    # Add vertical lines corresponding to the mean and median
    ax[1, i].axvline(increments.mean(), color="black", linestyle="-", label="Mean")
    ax[1, i].axvline(increments.median(), color="black", linestyle="--", label="Median")
    ax[1, i].set_yscale("log")
    ax[1, i].set_ylim(1, 1e3)
    # Overlay a Gaussian with the corresponding mean and variance onto each histogram
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
        ax[0, i].set_ylabel("")
        ax[1, i].set_ylabel("")
plt.savefig(f"plots/{dataset_name_save}_lag_stats.png")
plt.show()
