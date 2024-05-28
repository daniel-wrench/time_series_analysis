import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.sf_funcs as sf
import pickle
from sklearn.model_selection import train_test_split

from datetime import datetime

# Print the current time
print("Current time:", datetime.now(), "\n")

# Because Rāpoi can't handle latex apparently
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "stix",  # Set the font to use for math
        "font.family": "serif",  # Set the default font family
        "font.size": 10,
    }
)

input_path = "data/processed/"
save_dir = "plots/sim_results_local/"
# input_path = "/nfs/scratch/wrenchdani/time_series_analysis/data/processed_small/"
# save_dir = "plots/plots_small/"

missing_measure = "missing_prop"
n_ints_to_plot = 2
n_versions_to_plot = 2  # Number of version of each interval to plot

print("Reading in processed data files, merging...")
# List all pickle files in the folder
# pickle_files = [file for file in os.listdir(input_path) if file.endswith(".pkl")][:10]
pickle_files = ["sfs_psp_core_0.pkl"]
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

# Perform random train-test split

(
    good_inputs_train,
    good_inputs_test,
    good_outputs_train,
    good_outputs_test,
    bad_inputs_train,
    bad_inputs_test,
    bad_outputs_train,
    bad_outputs_test,
    interp_inputs_train,
    interp_inputs_test,
    interp_outputs_train,
    interp_outputs_test,
) = train_test_split(
    good_inputs_list,
    good_outputs_list,
    all_bad_inputs_list,
    all_bad_outputs_list,
    all_interp_inputs_list,
    all_interp_outputs_list,
    test_size=0.1,
    random_state=42,
)

times_to_gap = len(bad_inputs_train[0])

print(
    f"Number of training interval: {len(good_inputs_train)} x {times_to_gap} = {len(good_inputs_train)*len(bad_inputs_train[0])}"
)
print(
    f"Number of test intervals: {len(good_inputs_test)} x {times_to_gap} = {len(good_inputs_test)*len(bad_inputs_test[0])}"
)


print("Now plotting figures.")
# Check results, for a given clean input
for input_ind in range(n_ints_to_plot):
    sf.plot_sample(
        good_inputs_test,
        good_outputs_test,
        bad_inputs_test,
        bad_outputs_test,
        input_ind,
        n_versions_to_plot,
        False,
        "SF estimation subject to missing data: naive",
    )
    plt.savefig(save_dir + f"sf_i_{input_ind}_naive.png")
    plt.close()

    sf.plot_sample(
        good_inputs_test,
        good_outputs_test,
        interp_inputs_test,
        interp_outputs_test,
        input_ind,
        n_versions_to_plot,
        False,
        "SF estimation subject to missing data: linear interpolation",
    )
    plt.savefig(save_dir + f"sf_i_{input_ind}_lint.png")
    plt.close()


# Do holistic analysis of errors


# Concatenate the list of lists of dataframes to a single dataframe for error analysis
def concat_dfs(lst_of_list_of_dfs):
    merged_df = pd.concat(
        [pd.concat(lst, keys=range(len(lst))) for lst in lst_of_list_of_dfs],
        keys=range(len(lst_of_list_of_dfs)),
    )
    # Renaming MultiIndex levels
    merged_df.index.names = ["Original interval", "Interval version", "Lag"]
    return merged_df


bad_outputs_train_df = concat_dfs(bad_outputs_train)
interp_outputs_train_df = concat_dfs(interp_outputs_train)
bad_outputs_test_df = concat_dfs(bad_outputs_test)
interp_outputs_test_df = concat_dfs(interp_outputs_test)

# View trends as fn of OVERALL missing amount
for estimator in ["classical", "ch", "dowd"]:
    print(
        "Mean MAPE of uncorrected bad SFs in test set ({0}) = {1:.2f}".format(
            estimator, bad_outputs_test_df[f"{estimator}_error_percent"].abs().mean()
        )
    )
    print(
        "Mean MAPE of uncorrected interpolated SFs in test set ({0}) = {1:.2f}".format(
            estimator, interp_outputs_test_df[f"{estimator}_error_percent"].abs().mean()
        )
    )

    sf.plot_error_trend_line(
        other_outputs_df=bad_outputs_train_df,
        estimator=estimator,
        title=f"SF estimation ({estimator}, naive) error vs. lag and global sparsity",
        y_axis_log=False,
    )
    plt.savefig(save_dir + f"error_lag_{estimator}_naive.png")
    plt.close()

    sf.plot_error_trend_line(
        other_outputs_df=interp_outputs_train_df,
        estimator=estimator,
        title=f"SF estimation ({estimator}, lint) error vs. lag and global sparsity",
    )
    plt.savefig(save_dir + f"error_lag_{estimator}_lint.png")
    plt.close()

    sf.plot_error_trend_line(
        other_outputs_df=bad_outputs_train_df,
        estimator=estimator,
        title=f"SF estimation ({estimator}, naive) error vs. lag and global sparsity",
        y_axis_log=True,
    )
    plt.savefig(save_dir + f"error_lag_{estimator}_naive_log.png")
    plt.close()

    sf.plot_error_trend_line(
        other_outputs_df=interp_outputs_train_df,
        estimator=estimator,
        title=f"SF estimation ({estimator}, lint) error vs. lag and global sparsity",
        y_axis_log=True,
    )
    plt.savefig(save_dir + f"error_lag_{estimator}_lint_log.png")
    plt.close()


def plot_average_errors(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    stats = df.groupby("Lag")["classical_error_percent"].describe()
    plt.plot(stats["mean"], lw=3, label="Mean % error")
    plt.plot(stats["50%"], lw=3, label="Median % error")
    plt.semilogx()
    plt.legend()


plot_average_errors(interp_outputs_train_df)
plt.title("Average errors by lag in naive SFs")
plt.savefig(save_dir + "error_lag_classical_avg.png")
plt.close()

sf.plot_error_trend_scatter(bad_outputs_train_df, interp_outputs_train_df)
plt.savefig(save_dir + "error_lag_classical_avg.png")
plt.close()

# Check any cases of particularly large errors in lint dataset
# print("\nLargest errors in LINT SFs:\n")
# print(
#     interp_outputs_train_df.sort_values("classical_error_percent", ascending=False).head(5)
# )

# Create empirical correction factor using heatmap of errors

# Compute heatmap of average error as fn of lag and missing prop at that lag
# (logarithmic spacing for lags)

for n_bins in [15]:
    # First with no interpolation
    print(f"Calculating 2D heatmap with {n_bins} bins")
    heatmap_bin_vals_log_bad, heatmap_bin_edges_log_bad, lookup_table_log_bad = (
        sf.create_heatmap_lookup(
            bad_outputs_train_df, missing_measure, n_bins, log=True
        )
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.pcolormesh(
        heatmap_bin_edges_log_bad[0],
        heatmap_bin_edges_log_bad[1],
        heatmap_bin_vals_log_bad.T,
        cmap="bwr",
    )
    plt.colorbar(label="MPE")
    plt.clim(-100, 100)
    plt.xlabel("Lag")
    plt.ylabel("Missing proportion")
    plt.title("Distribution of missing proportion and lag (NO LINT)")
    ax.set_facecolor("black")
    ax.set_xscale("log")
    plt.savefig(save_dir + f"error_heatmap_b_{n_bins}_2d_naive.png")
    plt.close()

    # Now with linear interpolation
    print(f"Calculating 3D heatmap with {n_bins} bins")
    heatmap_bin_vals, heatmap_bin_edges, lookup_table = sf.create_heatmap_lookup(
        interp_outputs_train_df, missing_measure, n_bins, log=True
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    plt.pcolormesh(
        heatmap_bin_edges[0],
        heatmap_bin_edges[1],
        heatmap_bin_vals.T,
        cmap="bwr",
    )
    plt.colorbar(label="MPE")
    plt.clim(-100, 100)
    plt.xlabel("Lag")
    plt.ylabel("Missing proportion")
    plt.title("Distribution of missing proportion and lag")
    ax.set_facecolor("black")
    ax.set_xscale("log")
    plt.savefig(save_dir + f"error_heatmap_b_{n_bins}_2d.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 5))
    hb = ax.hist2d(
        interp_outputs_train_df["lag"],
        interp_outputs_train_df[missing_measure],
        bins=n_bins,
        cmap="copper",
        range=[[0, interp_outputs_train_df.lag.max()], [0, 1]],
    )
    plt.colorbar(hb[3], ax=ax, label="Counts")
    hb[3].set_clim(0, hb[0].max())
    plt.xlabel("Lag")
    plt.ylabel("Missing proportion")
    plt.title("Distribution of missing proportion and lag (linear bins)")
    plt.savefig(save_dir + f"error_heatmap_b_{n_bins}_2d_counts.png")
    plt.close()

    # Now in 3D
    # (logarithmic spacing for lags and power)
    heatmap_bin_vals_3d, heatmap_bin_edges_3d, lookup_table_3d = (
        sf.create_heatmap_lookup_3D(
            interp_outputs_train_df, missing_measure, n_bins, True
        )
    )

    # Apply 2D and 3D scaling to test set, report avg errors
    print(f"Correcting test set intervals using 2D error heatmap with {n_bins} bins")
    test_set_corrected = sf.compute_scaling(
        interp_outputs_test_df, missing_measure, lookup_table
    )
    print(f"Correcting test set intervals using 3D error heatmap with {n_bins} bins")
    # Overriding with the version including the 3D scaling (as well as 2d) to save memory
    test_set_corrected = sf.compute_scaling_3d(
        interp_outputs_test_df, missing_measure, lookup_table_3d
    )

    # Calculate corrected test set error
    error_percents_2d = []
    error_percents_3d = []

    for i, df in enumerate(good_outputs_test):
        for j in range(times_to_gap):
            key = (i, j)  # Checking bug issue here
            if key not in test_set_corrected.index:
                print(f"\nKey {key} not found in the MultiIndex")
            else:
                error_2d = (
                    test_set_corrected.loc[(i, j), "classical_corrected"]
                    - df["classical"]
                )
                error_percent_2d = error_2d / df["classical"] * 100
                error_percents_2d.append(error_percent_2d)

                error_3d = (
                    test_set_corrected.loc[(i, j), "classical_corrected_3d"]
                    - df["classical"]
                )
                error_percent_3d = error_3d / df["classical"] * 100
                error_percents_3d.append(error_percent_3d)

    print(
        "Mean MAPE of corrected interpolated intervals test set (classical, 2D, {0} bins) = {1:.2f}".format(
            n_bins, np.mean(np.abs(error_percents_2d))
        )
    )
    print(
        "Mean MAPE of corrected interpolated intervals test set (classical, 3D, {0} bins) = {1:.2f}".format(
            n_bins, np.mean(np.abs(error_percents_3d))
        )
    )

    for input_ind in range(n_ints_to_plot):
        fig, axs = plt.subplots(
            n_versions_to_plot, 2, figsize=(10, 4 * n_versions_to_plot)
        )

        # Get the relevent indices in order of sparsity for better plot aesthetics
        versions_ordered_sparsity = (
            interp_outputs_test_df.loc[input_ind, : n_versions_to_plot - 1, :]
            .sort_values("missing_prop_overall")
            .index.get_level_values(1)
            .unique()
            .values
        )

        for i, int_version in enumerate(versions_ordered_sparsity):
            mape_bad = (
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_error_percent"
                ]
                .abs()
                .mean()
            )
            mape_corrected = (
                error_percents_2d[input_ind * times_to_gap + int_version].abs().mean()
            )
            mape_corrected_3d = (
                error_percents_3d[input_ind * times_to_gap + int_version].abs().mean()
            )

            axs[i, 1].annotate(
                "MAPE = {:.2f}".format(mape_bad),
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(0.05, 0.9),
                textcoords="axes fraction",
                transform=axs[i, 1].transAxes,
                c="black",
                bbox=dict(facecolor="white", edgecolor="white", boxstyle="round"),
            )

            axs[i, 1].annotate(
                "MAPE = {:.2f}".format(mape_corrected),
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(0.05, 0.8),
                textcoords="axes fraction",
                transform=axs[i, 1].transAxes,
                c="blue",
                bbox=dict(facecolor="white", edgecolor="white", boxstyle="round"),
            )

            axs[i, 1].annotate(
                "MAPE = {:.2f}".format(mape_corrected_3d),
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(0.05, 0.7),
                textcoords="axes fraction",
                transform=axs[i, 1].transAxes,
                c="purple",
                bbox=dict(facecolor="white", edgecolor="white", boxstyle="round"),
            )

            axs[i, 1].plot(
                good_outputs_test[input_ind]["classical"],
                color="black",
                label="True (classical)",
                lw=3,
                alpha=0.5,
            )
            axs[i, 1].plot(
                interp_outputs_test_df.loc[input_ind, int_version]["classical"],
                color="black",
                lw=1,
                label="Interp. (classical)",
            )
            # axs[i, 1].plot(
            #     sf_corrected["classical"] * sf_corrected["scaling"], c="blue", label="Corrected Bad"
            # )
            axs[i, 1].plot(
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_corrected"
                ],
                c="blue",
                lw=1.2,
                ls=":",
                label="Interp. corr. (2D) $\pm$2SE",
            )
            axs[i, 1].fill_between(
                interp_outputs_test_df.loc[input_ind, int_version]["lag"],
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_corrected_lower"
                ],
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_corrected_upper"
                ],
                color="blue",
                alpha=0.2,
            )
            #    axs[i, 1].plot(
            #        sf_corrected["classical_corrected_smoothed"],
            #        c="orange",
            #        label="Corrected Bad Smoothed",
            #    )
            axs[i, 1].plot(
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_corrected_3d"
                ],
                c="purple",
                ls=":",
                lw=1.2,
                label="Interp. corrected. (3D)",
            )
            axs[i, 1].semilogx()
            axs[i, 1].semilogy()
            axs[i, 1].set_ylim(1e-2, 1e1)
            axs[i, 1].legend(loc="lower right")

            # if log is True:
            c = axs[i, 0].pcolormesh(
                heatmap_bin_edges[0],
                heatmap_bin_edges[1] * 100,  # convert to % Missing
                heatmap_bin_vals.T,
                cmap="bwr",
            )
            # fig.colorbar(c, ax=axs[i, 0], label="MPE")
            c.set_clim(-100, 100)
            c.set_facecolor("black")
            # axs[i, 0].set_xlabel("Lag")
            axs[i, 0].plot(
                interp_outputs_test_df.loc[input_ind, int_version]["lag"],
                interp_outputs_test_df.loc[input_ind, int_version][missing_measure]
                * 100,
                c="black",
            )

            axs[i, 0].set_xscale("log")
            axs[i, 0].set_ylim(0, 100)
            # axs[i, 0] = sf.plot_heatmap(
            #     heatmap_bin_vals_log,
            #     heatmap_bin_edges_log,
            #     missing_measure=missing_measure,
            #     log=True,
            #     overlay_x=df_to_plot["lag"],
            #     overlay_y=df_to_plot[missing_measure],
            #     subplot=axs[i, 0],
            # )
        axs[0, 1].set_title("Applying correction factor")
        axs[0, 0].set_title("Extracting correction factor")
        axs[n_versions_to_plot - 1, 0].set_xlabel("Lag ($\\tau$)")
        axs[n_versions_to_plot - 1, 1].set_xlabel("Lag ($\\tau$)")
        axs[0, 0].set_ylabel("% pairs missing")
        fig.suptitle(
            "Applying correction factor to interpolated SFs in test set", size=16
        )
        plt.savefig(
            save_dir + f"sf_i_{input_ind}_classical_lint_corrected_b_{n_bins}_2d.png"
        )
        plt.close()

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
plt.savefig(save_dir + f"error_heatmap_b_{n_bins}_3d_power.png")
plt.close()

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
    ax[i].set_xlabel("Missing prop")
    ax[i].set_ylabel("Power")
    plt.title("Distribution of missing proportion and lag")
    ax[i].set_facecolor("black")
    ax[i].semilogy()
    ax[i].set_title(
        f"Lag bin {i+1}/{n_bins}".format(np.round(heatmap_bin_edges_3d[2][i], 2))
    )
    ax[i].set_xlabel("Missing prop")
    # Remove y-axis labels for all but the first plot
    if i > 0:
        ax[i].set_yticklabels([])
        ax[i].set_ylabel("")
plt.savefig(save_dir + f"error_heatmap_b_{n_bins}_3d_lag.png")
plt.close()

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
plt.savefig(save_dir + f"error_heatmap_b_{n_bins}_3d_missing.png")
plt.close()

print("Done!")
print("\nCurrent time:", datetime.now())

##################################

# Plot bar plots of test set errors

# LATEST: 50 x 1040 intervals

# import seaborn as sns

# # Create the DataFrame
# data = {
#     "Estimator": [
#         "Classical",
#         "Classical",
#         "CH",
#         "CH",
#         "Dowd",
#         "Dowd",
#         "Classical",
#         "Classical",
#         "Classical",
#         "Classical",
#         "Classical",
#         "Classical",
#     ],
#     "Corrected": [
#         "No",
#         "No",
#         "No",
#         "No",
#         "No",
#         "No",
#         "Yes",
#         "Yes",
#         "Yes",
#         "Yes",
#         "Yes",
#         "Yes",
#     ],
#     "Gap handling method": [
#         "No handling",
#         "LINT",
#         "No handling",
#         "LINT",
#         "No handling",
#         "LINT",
#         "LINT",
#         "LINT",
#         "LINT",
#         "LINT",
#         "LINT",
#         "LINT",
#     ],
#     "Number of bins": [
#         np.nan,
#         np.nan,
#         np.nan,
#         np.nan,
#         np.nan,
#         np.nan,
#         15,
#         20,
#         25,
#         15,
#         20,
#         25,
#     ],
#     "Dimensions": [
#         np.nan,
#         np.nan,
#         np.nan,
#         np.nan,
#         np.nan,
#         np.nan,
#         "2D",
#         "2D",
#         "2D",
#         "3D",
#         "3D",
#         "3D",
#     ],
#     "MAPE on test set": [
#         21.4,
#         14.0,
#         25.7,
#         16.6,
#         36.4,
#         22.3,
#         12.3, BELOW TO BE UPDATED
#         12.2,
#         12.2,
#         10.7,
#         10.6,
#         10.6,
#     ],
# }

# df = pd.DataFrame(data)

# Set the plot size
# plt.figure(figsize=(6, 4))
# plt.tight_layout()

# # Create the bar plot
# sns.barplot(
#     data=df[df.Corrected == "No"],
#     x="Gap handling method",
#     y="MAPE on test set",
#     hue="Estimator",
# )

# # Add title and labels
# plt.title("MAPE on Test Set by Estimator and Gap Handling Method")
# plt.xlabel("Gap handling method")
# plt.ylabel("MAPE on Test Set")

# # Return the y-lims of the plot
# ymin, ymax = plt.ylim()

# # Show the plot
# plt.show()

# # Set the plot size
# plt.figure(figsize=(6, 4))
# plt.tight_layout()
# # Create the bar plot
# sns.barplot(
#     data=df[df.Corrected == "Yes"],
#     x="Dimensions",
#     y="MAPE on test set",
#     hue="Number of bins",
#     palette=sns.color_palette("Blues", 3),
# )
# # Make the color palette a set of 3 blues


# # Add title and labels
# plt.title("MAPE on Test Set using Classical LINT + Correction Factor")
# plt.xlabel("Dimensionality of correction factor")
# plt.ylabel("MAPE on Test Set")
# plt.ylim(ymin, ymax)  # Set the y-lims to be the same as the previous plot

# # Show the plot
# plt.show()
