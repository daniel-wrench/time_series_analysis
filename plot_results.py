import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils as utils  # copied directly from Reynolds project, normalize() added
import sf_funcs as sf
import pickle

# Because Rāpoi can't handle latex apparently
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "stix",  # Set the font to use for math
        "font.family": "serif",  # Set the default font family
        # "font.size": 10,
    }
)

input_path = "data/processed/"
save_dir = "plots/"
missing_measure = "missing_prop"
n_bins = 18
input_ind = 2
n = 4

print("Reading in processed data files, merging...")
# List all pickle files in the folder
pickle_files = [file for file in os.listdir(input_path) if file.endswith(".pkl")]

good_inputs_list = []
good_outputs_list = []
all_bad_inputs_list = []
all_bad_outputs_list = []
all_interp_inputs_list = []
all_interp_outputs_list = []

# Read in all pickle files in the directory
for file in pickle_files:
    with open(os.path.join(input_path, file), "rb") as file:
        list_of_list_of_dfs = pickle.load(file)

        good_inputs_list += list_of_list_of_dfs[0]
        good_outputs_list += list_of_list_of_dfs[1]
        all_bad_inputs_list += list_of_list_of_dfs[2]
        all_bad_outputs_list += list_of_list_of_dfs[3]
        all_interp_inputs_list += list_of_list_of_dfs[4]
        all_interp_outputs_list += list_of_list_of_dfs[5]


print(
    f"... = {len(all_interp_outputs_list[0])} versions of {len(all_interp_outputs_list)} inputs"
)
print("Now plotting figures.")
# Check results, for a given clean input

sf.plot_sample(
    good_inputs_list,
    good_outputs_list,
    all_bad_inputs_list,
    all_bad_outputs_list,
    "C0",
    input_ind,
    n,
    False,
)
plt.savefig(save_dir + f"psp_missing_effect_int_{input_ind}.png")
plt.clf()

sf.plot_sample(
    good_inputs_list,
    good_outputs_list,
    all_interp_inputs_list,
    all_interp_outputs_list,
    "purple",
    input_ind,
    n,
    False,
)
plt.savefig(save_dir + f"psp_missing_effect_int_{input_ind}_lint.png")
plt.clf()


# Do holistic analysis of errors

# Concatenate the list of lists of dataframes to a single dataframe for error analysis
bad_outputs_df = pd.concat(
    [pd.concat(lst, keys=range(len(lst))) for lst in all_bad_outputs_list],
    keys=range(len(all_bad_outputs_list)),
)

# Renaming MultiIndex levels
bad_outputs_df.index.names = ["Original interval", "Interval version", "Lag"]

interp_outputs_df = pd.concat(
    [pd.concat(lst, keys=range(len(lst))) for lst in all_interp_outputs_list],
    keys=range(len(all_interp_outputs_list)),
)

# Renaming MultiIndex levels
interp_outputs_df.index.names = ["Original interval", "Interval version", "Lag"]

# View trends as fn of OVERALL missing amount
sf.plot_error_trend_line(other_outputs_df=bad_outputs_df)
plt.savefig(save_dir + "psp_missing_effect_holistic.png")
plt.clf()

sf.plot_error_trend_line(
    other_outputs_df=interp_outputs_df,
    title="SF estimation error vs. lag and global sparsity (LINT)",
)
plt.savefig(save_dir + "psp_missing_effect_holistic_lint.png")
plt.clf()


def plot_average_errors(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    stats = df.groupby("Lag")["error_percent"].describe()
    plt.plot(stats["mean"], lw=3, label="Mean % error")
    plt.plot(stats["50%"], lw=3, label="Median % error")
    plt.semilogx()
    plt.legend()


plot_average_errors(bad_outputs_df)
plt.title("Average errors by lag in naive SFs")
plt.savefig(save_dir + "psp_missing_effect_holistic_avg.png")
plt.clf()

sf.plot_error_trend_scatter(bad_outputs_df, interp_outputs_df)
plt.savefig(save_dir + "psp_missing_effect_holistic_scatter.png")
plt.clf()

# Check any cases of particularly large errors in lint dataset
print("\nLargest errors in LINT SFs:\n")
print(interp_outputs_df.sort_values("error_percent", ascending=False).head(5))

# Create empirical correction factor using heatmap of errors

# Compute heatmap of average error as fn of lag and missing prop at that lag
# (logarithmic spacing for lags)
heatmap_bin_vals_log, heatmap_bin_edges_log, lookup_table_log = (
    sf.create_heatmap_lookup(interp_outputs_df, missing_measure, n_bins, log=True)
)

fig, ax = plt.subplots(figsize=(7, 5))
plt.pcolormesh(
    heatmap_bin_edges_log[0],
    heatmap_bin_edges_log[1],
    heatmap_bin_vals_log.T,
    cmap="bwr",
)
plt.colorbar(label="MPE")
plt.clim(-100, 100)
plt.xlabel("Lag")
plt.ylabel("Missing proportion")
plt.title("Distribution of missing proportion and lag")
ax.set_facecolor("black")
ax.set_xscale("log")
plt.savefig(save_dir + f"psp_correction_heatmap_{n_bins}_bins.png")
plt.clf()

fig, ax = plt.subplots(figsize=(7, 5))
hb = ax.hist2d(
    interp_outputs_df["lag"],
    interp_outputs_df[missing_measure],
    bins=n_bins,
    cmap="copper",
    range=[[0, interp_outputs_df.lag.max()], [0, 1]],
)
plt.colorbar(hb[3], ax=ax, label="Counts")
hb[3].set_clim(0, hb[0].max())
plt.xlabel("Lag")
plt.ylabel("Missing proportion")
plt.title("Distribution of missing proportion and lag (linear bins)")
plt.savefig(save_dir + "psp_heatmap_sample_size.png")
plt.clf()

# Now in 3D
# (logarithmic spacing for lags and power)
heatmap_bin_vals_3d, heatmap_bin_edges_3d, lookup_table_3d = (
    sf.create_heatmap_lookup_3D(interp_outputs_df, missing_measure, n_bins, True)
)

fig, axs = plt.subplots(n, 2, figsize=(10, 4 * n))

# Before plotting, sort the n outputs by missing proportion
other_outputs_plot = all_interp_outputs_list[input_ind][:n]
sparsities = [df["missing_prop_overall"].values[0] for df in other_outputs_plot]
sorted_lists = zip(*sorted(zip(sparsities, other_outputs_plot)))
sparsities_ordered, other_outputs_plot = sorted_lists

for i, df_to_plot in enumerate(other_outputs_plot):
    print(f"\n2D: Correcting interval {i}:")
    sf_corrected = sf.compute_scaling(
        df_to_plot,
        missing_measure,
        lookup_table_log,
    )
    print(f"\n3D: Correcting interval {i}:")
    sf_corrected_3d = sf.compute_scaling_3d(
        df_to_plot,
        missing_measure,
        lookup_table_3d,
    )

    sf_corrected["error_corrected"] = (
        sf_corrected["sosf_corrected"] - good_outputs_list[input_ind]["sosf"]
    )
    sf_corrected["error_percent_corrected"] = (
        sf_corrected["error_corrected"] / good_outputs_list[input_ind]["sosf"] * 100
    )
    mape_bad = sf_corrected["error_percent"].abs().mean()
    mape_corrected = sf_corrected["error_percent_corrected"].abs().mean()

    sf_corrected_3d["error_corrected_3d"] = (
        sf_corrected_3d["sosf_corrected_3d"] - good_outputs_list[input_ind]["sosf"]
    )
    sf_corrected_3d["error_percent_corrected_3d"] = (
        sf_corrected_3d["error_corrected_3d"]
        / good_outputs_list[input_ind]["sosf"]
        * 100
    )
    mape_corrected_3d = sf_corrected_3d["error_percent_corrected_3d"].abs().mean()

    # mape_corrected_sm = sf_corrected["error_percent_corrected_sm"].abs().mean()

    axs[i, 1].annotate(
        "MAPE = {:.2f}".format(mape_bad),
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.05, 0.9),
        textcoords="axes fraction",
        transform=axs[i, 1].transAxes,
        c="red",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )

    axs[i, 1].annotate(
        "MAPE = {:.2f}".format(mape_corrected),
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.05, 0.8),
        textcoords="axes fraction",
        transform=axs[i, 1].transAxes,
        c="blue",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )

    axs[i, 1].annotate(
        "MAPE = {:.2f}".format(mape_corrected_3d),
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.05, 0.7),
        textcoords="axes fraction",
        transform=axs[i, 1].transAxes,
        c="purple",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
    )

    axs[i, 1].plot(good_outputs_list[input_ind]["sosf"], c="green", label="True")
    axs[i, 1].plot(df_to_plot["sosf"], c="red", label="Interp.")
    # axs[i, 1].plot(
    #     sf_corrected["sosf"] * sf_corrected["scaling"], c="blue", label="Corrected Bad"
    # )
    axs[i, 1].plot(sf_corrected["sosf_corrected"], c="blue", label="Corrected Interp.")
    #    axs[i, 1].plot(
    #        sf_corrected["sosf_corrected_smoothed"],
    #        c="orange",
    #        label="Corrected Bad Smoothed",
    #    )
    axs[i, 1].plot(
        sf_corrected_3d["sosf_corrected_3d"],
        c="purple",
        label="Corrected Interp. (3D)",
    )
    axs[i, 1].semilogx()
    axs[i, 1].semilogy()
    axs[i, 1].legend(loc="lower right")
    axs[i, 1].set_title(
        f"Input version {i+1} with {np.round(sf_corrected['missing_prop_overall'].values[0]*100, 2)}% missing"
    )

    # if log is True:
    c = axs[i, 0].pcolormesh(
        heatmap_bin_edges_log[0],
        heatmap_bin_edges_log[1],
        heatmap_bin_vals_log.T,
        cmap="bwr",
    )
    # fig.colorbar(c, ax=axs[i, 0], label="MPE")
    c.set_clim(-100, 100)
    c.set_facecolor("black")
    # axs[i, 0].set_xlabel("Lag")
    axs[i, 0].plot(
        df_to_plot["lag"],
        df_to_plot[missing_measure],
    )

    axs[i, 0].set_ylabel("Missing proportion (overall)")
    axs[i, 0].set_title("Correction factor extraction")
    axs[i, 0].set_xscale("log")

    # axs[i, 0] = sf.plot_heatmap(
    #     heatmap_bin_vals_log,
    #     heatmap_bin_edges_log,
    #     missing_measure=missing_measure,
    #     log=True,
    #     overlay_x=df_to_plot["lag"],
    #     overlay_y=df_to_plot[missing_measure],
    #     subplot=axs[i, 0],
    # )

plt.savefig(save_dir + f"psp_corrected_{n_bins}_bins_CHECK.png")
plt.clf()

# Plotting 3D heatmaps

fig, ax = plt.subplots(1, n_bins, figsize=(n_bins * 3, 3), tight_layout=True)
# Remove spacing between subplots
plt.subplots_adjust(wspace=0.2)
for i in range(n_bins):
    c = ax[i].pcolormesh(
        heatmap_bin_edges_3d[0],
        heatmap_bin_edges_3d[1],
        heatmap_bin_vals_3d[:, :, i],
        cmap="bwr",
    )
    # plt.colorbar(label="MPE")
    c.set_clim(-100, 100)
    plt.xlabel("Lag")
    plt.ylabel("Missing proportion")
    plt.title("Distribution of missing proportion and lag")
    ax[i].set_facecolor("black")
    ax[i].semilogx()
    ax[i].set_title(
        f"Power bin {i+1}/{n_bins}".format(np.round(heatmap_bin_edges_3d[2][i], 2))
    )
    ax[i].set_xlabel("Lag")
    # Remove y-axis labels for all but the first plot
    if i > 0:
        ax[i].set_yticklabels([])
        ax[i].set_ylabel("")
plt.savefig(save_dir + "psp_heatmap_3d_power.png")
plt.clf()

fig, ax = plt.subplots(1, n_bins, figsize=(n_bins * 3, 3), tight_layout=True)
# Remove spacing between subplots
plt.subplots_adjust(wspace=0.2)
for i in range(n_bins):
    c = ax[i].pcolormesh(
        heatmap_bin_edges_3d[1],
        heatmap_bin_edges_3d[2],
        heatmap_bin_vals_3d[i, :, :],
        cmap="bwr",
    )
    # plt.colorbar(label="MPE")
    c.set_clim(-100, 100)
    plt.xlabel("Missing prop")
    plt.ylabel("Power")
    plt.title("Distribution of missing proportion and lag")
    ax[i].set_facecolor("black")
    ax[i].semilogy()
    ax[i].set_ylim(6e-1, 105)
    ax[i].set_title(
        f"Lag bin {i+1}/{n_bins}".format(np.round(heatmap_bin_edges_3d[2][i], 2))
    )
    ax[i].set_xlabel("Missing prop")
    # Remove y-axis labels for all but the first plot
    if i > 0:
        ax[i].set_yticklabels([])
        ax[i].set_ylabel("")
plt.savefig(save_dir + "psp_heatmap_3d_lag.png")
plt.clf()

fig, ax = plt.subplots(1, n_bins, figsize=(n_bins * 3, 3), tight_layout=True)
# Remove spacing between subplots
plt.subplots_adjust(wspace=0.2)
for i in range(n_bins):
    c = ax[i].pcolormesh(
        heatmap_bin_edges_3d[0],
        heatmap_bin_edges_3d[2],
        heatmap_bin_vals_3d[:, i, :],
        cmap="bwr",
    )
    # plt.colorbar(label="MPE")
    c.set_clim(-100, 100)
    plt.title("Distribution of missing proportion and lag")
    ax[i].set_facecolor("black")
    ax[i].semilogx()
    ax[i].semilogy()
    ax[i].set_title(
        f"Missing prop bin {i+1}/{n_bins}".format(
            np.round(heatmap_bin_edges_3d[2][i], 2)
        )
    )
    ax[i].set_xlabel("Lag")
    ax[i].set_ylabel("Power")
    # Remove y-axis labels for all but the first plot
    if i > 0:
        ax[i].set_yticklabels([])
        ax[i].set_ylabel("")
plt.savefig(save_dir + "psp_heatmap_3d_missing.png")
plt.clf()
