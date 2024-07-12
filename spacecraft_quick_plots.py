# Quick side-by-side plot of PSP, Wind, and Voyager data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import src.utils as utils  # copied directly from Reynolds project, normalize() added


# Read in data and calculate correlation scales
with open("data\processed\psp\Bx_raw_20160101.pkl", "rb") as f:
    psp = pd.read_pickle(f)

p_time_lags_lr, p_r_vec_lr = utils.compute_nd_acf(
    [psp],
    nlags=1000,
)

psp_tce_lags = utils.compute_outer_scale_exp_trick(
    p_time_lags_lr, p_r_vec_lr, plot=False
)
psp_tce_s = psp_tce_lags * psp.index.freq.delta.total_seconds()

with open("data\processed\wind\Bx_raw_20160101.pkl", "rb") as f:
    wind = pd.read_pickle(f)

w_time_lags_lr, w_r_vec_lr = utils.compute_nd_acf(
    [wind],
    nlags=1000,
)

w_tce_lags = utils.compute_outer_scale_exp_trick(w_time_lags_lr, w_r_vec_lr, plot=False)
w_tce_s = w_tce_lags * wind.index.freq.delta.total_seconds()

voyager = pd.read_csv(
    "data/processed/voyager1_48s.csv", parse_dates=["Time"], index_col="Time"
)
# Set frequency to 48s
voyager.index.freq = pd.Timedelta("48s")

v_time_lags_lr, v_r_vec_lr = utils.compute_nd_acf(
    [voyager["BR"]],
    nlags=1000,
)

v_tce_lags = utils.compute_outer_scale_exp_trick(v_time_lags_lr, v_r_vec_lr, plot=False)
v_tce_s = v_tce_lags * voyager.index.freq.delta.total_seconds()

# Calculate correlation scales and save, for each dataset:


fig, axs = plt.subplots(3, 1, figsize=(5, 8), constrained_layout=True)

axs[0].plot(psp["B_R"], label="PSP")
axs[0].set_title("PSP (0.3au)")
axs[0].set_ylabel("$B_R$ (nT)")
# Annotate with the mean and variance
axs[0].annotate(
    f"Mean: {np.mean(psp.B_R):.2f}\nVar: {np.var(psp.B_R):.2f}",
    xy=(0.8, 0.8),
    xycoords="axes fraction",
)

axs[1].plot(wind["Bx"], label="Wind")
axs[1].set_title("Wind (1au)")
axs[1].set_ylabel("$B_x$ (nT)")
axs[1].annotate(
    f"Mean: {np.mean(wind.Bx):.2f}\nVar: {np.var(wind.Bx):.2f}",
    xy=(0.8, 0.8),
    xycoords="axes fraction",
)

axs[2].plot(voyager["BR"], label="Voyager")
axs[2].set_title("Voyager (48au)")
axs[2].set_ylabel("$B_R$ (nT)")
axs[2].annotate(
    f"Mean: {np.mean(voyager.BR):.3f}\nVar: {np.var(voyager.BR):.4f}",
    xy=(0.8, 0.8),
    xycoords="axes fraction",
)
# Rotate the x-axis labels
# axs[0].tick_params(axis='x', rotation=45)
axs[1].tick_params(axis="x", rotation=45)

for ax in axs:
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )

    # ax_inset = inset_axes(ax, width="30%", height="30%", loc="lower left")
    # Remove the axis labels


# Plot the correlation functions for each dataset in the inset
# and annotate with the correlation scales

ax_inset = inset_axes(axs[0], width="30%", height="30%", loc="lower left")
ax_inset.plot(p_time_lags_lr, p_r_vec_lr, label="PSP", color="black")
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.yaxis.set_label_position("right")
ax_inset.set_title("$R(\\tau)$")

ax_inset.annotate(
    f"$\lambda_c$ = {psp_tce_s:.3g}s",
    xy=(0.2, 0.7),
    xycoords="axes fraction",
)

ax_inset = inset_axes(axs[1], width="30%", height="30%", loc="lower left")
ax_inset.plot(w_time_lags_lr, w_r_vec_lr, label="Wind", color="black")
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.yaxis.set_label_position("right")
ax_inset.set_title("$R(\\tau)$")

ax_inset.annotate(
    f"$\lambda_c$ = {w_tce_s:.3g}s",
    xy=(0.2, 0.7),
    xycoords="axes fraction",
)

ax_inset = inset_axes(axs[2], width="30%", height="30%", loc="lower left")
ax_inset.plot(v_time_lags_lr, v_r_vec_lr, label="Voyager", color="black")
ax_inset.set_xticks([])
ax_inset.set_yticks([])
ax_inset.yaxis.set_label_position("right")
ax_inset.set_title("$R(\\tau)$")

ax_inset.annotate(
    f"$\lambda_c$ = {v_tce_s:.3g}s",
    xy=(0.2, 0.7),
    xycoords="axes fraction",
)

# Save figure
plt.savefig("plots/background/spacecraft_comparision.png")
