import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.sf_funcs as sf
import pickle
from sklearn.model_selection import train_test_split
from matplotlib.gridspec import GridSpec


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

input_path = "data/processed/external/"
save_dir = "plots/temp/"
# input_path = "/nfs/scratch/wrenchdani/time_series_analysis/data/processed_small/"
# save_dir = "plots/plots_small/"

missing_measure = "missing_prop"
n_ints_to_plot = 2
n_versions_to_plot = 2  # Number of version of each interval to plot

print("Reading in processed data files, merging...")
# List all pickle files in the folder
# pickle_files = [file for file in os.listdir(input_path) if file.endswith(".pkl")][:10]
pickle_files = ["sfs_wind_core_0.pkl"]
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
    good_inputs_test,
    good_outputs_test,
    bad_inputs_test,
    bad_outputs_test,
    interp_inputs_test,
    interp_outputs_test,
) = (
    good_inputs_list,
    good_outputs_list,
    all_bad_inputs_list,
    all_bad_outputs_list,
    all_interp_inputs_list,
    all_interp_outputs_list,
)

times_to_gap = len(bad_inputs_test[0])

# print(
#     f"Number of training interval: {len(good_inputs_train)} x {times_to_gap} = {len(good_inputs_train)*len(bad_inputs_train[0])}"
# )
print(
    f"Number of Wind test intervals: {len(good_inputs_test)} x {times_to_gap} = {len(good_inputs_test)*len(bad_inputs_test[0])}"
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
        ["classical"],
        "SF estimation subject to missing data: naive",
    )
    plt.savefig(save_dir + f"sf_i_{input_ind}_naive_wind.png")
    plt.close()

    sf.plot_sample(
        good_inputs_test,
        good_outputs_test,
        interp_inputs_test,
        interp_outputs_test,
        input_ind,
        n_versions_to_plot,
        False,
        ["classical"],
        "SF estimation subject to missing data: linear interpolation",
    )
    plt.savefig(save_dir + f"sf_i_{input_ind}_lint_wind.png")
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


bad_outputs_test_df = concat_dfs(bad_outputs_test)
interp_outputs_test_df = concat_dfs(interp_outputs_test)

# View trends as fn of OVERALL missing amount
for estimator in ["classical", "ch", "dowd"]:
    print(
        "Mean MAPE of uncorrected bad SFs in external test set ({0}) = {1:.2f}".format(
            estimator, bad_outputs_test_df[f"{estimator}_error_percent"].abs().mean()
        )
    )
    print(
        "Mean MAPE of uncorrected interpolated SFs in external test set ({0}) = {1:.2f}".format(
            estimator, interp_outputs_test_df[f"{estimator}_error_percent"].abs().mean()
        )
    )


def plot_average_errors(df):
    fig, ax = plt.subplots(figsize=(6, 3))
    stats = df.groupby("Lag")["classical_error_percent"].describe()
    plt.plot(stats["mean"], lw=3, label="Mean % error")
    plt.plot(stats["50%"], lw=3, label="Median % error")
    plt.semilogx()
    plt.legend()


# Annotate each heatmap trace with info
def annotate_curve(ax, x, y, text, offset_scaling=(0.3, 0.1)):
    # Find the index of y value closest to the median value
    idx = np.argmin(np.abs(y - np.percentile(y, 20)))

    # Coordinates of the point of maximum y value
    x_max = x[idx]
    y_max = y[idx]

    # Convert offset from axes fraction to data coordinates
    x_text = 10 ** (offset_scaling[0] * np.log10(x_max))  # Log-axis
    y_text = y_max + offset_scaling[1] * (ax.get_ylim()[1] - ax.get_ylim()[0])

    # Annotate with the text, adjusting the position with xytext_offset
    ax.annotate(
        text,
        xy=(x_max, y_max - 1),
        xytext=(x_text, y_text),
        # xycoords="axes fraction",
        # textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
        bbox=dict(facecolor="white", edgecolor="white", boxstyle="round", alpha=0.7),
    )


for n_bins in [15]:
    # Import the lookup tables and 2D error heatmap
    lookup_table = pd.read_csv(save_dir + f"lookup_table_2d_b_{n_bins}.csv")
    lookup_table_3d = pd.read_csv(save_dir + f"lookup_table_3d_b_{n_bins}.csv")

    with open(save_dir + f"error_heatmap_b_{n_bins}.pkl", "rb") as file:
        heatmap_bin_vals, heatmap_bin_edges = pickle.load(file)

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
        "Mean MAPE of corrected interpolated intervals in external test set (classical, 2D, {0} bins) = {1:.2f}".format(
            n_bins, np.mean(np.abs(error_percents_2d))
        )
    )
    print(
        "Mean MAPE of corrected interpolated intervals in external test set (classical, 3D, {0} bins) = {1:.2f}".format(
            n_bins, np.mean(np.abs(error_percents_3d))
        )
    )

    for input_ind in range(n_ints_to_plot):
        fig = plt.figure(figsize=(12, 4))

        # Create a GridSpec layout with specified width ratios and horizontal space
        gs1 = GridSpec(1, 1, left=0.06, right=0.35)
        gs2 = GridSpec(1, 2, left=0.43, right=0.99, wspace=0)

        # Create subplots
        ax0 = fig.add_subplot(gs1[0, 0])
        ax1 = fig.add_subplot(gs2[0, 0])

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
            try:
                mape_corrected = (
                    error_percents_2d[input_ind * times_to_gap + int_version]
                    .abs()
                    .mean()
                )
                mape_corrected_3d = (
                    error_percents_3d[input_ind * times_to_gap + int_version]
                    .abs()
                    .mean()
                )

            except IndexError:
                print(
                    f"IndexError with finding error to annotate plot: input_ind={input_ind}, int_version={int_version}, times_to_gap={times_to_gap}"
                )

            if i == 0:
                ax = ax1
                ax.set_ylabel("SF")
            else:
                ax = fig.add_subplot(gs2[0, i], sharey=ax1)
                plt.setp(ax.get_yticklabels(), visible=False)

            ax.plot(
                good_outputs_test[input_ind]["classical"],
                color="black",
                label="True",
                lw=3,
                alpha=0.5,
            )
            ax.plot(
                interp_outputs_test_df.loc[input_ind, int_version]["classical"],
                color="black",
                lw=1,
                label="Interpolated ({:.2f})".format(mape_bad),
            )
            # axs[i+1].plot(
            #     sf_corrected["classical"] * sf_corrected["scaling"], c="blue", label="Corrected Bad"
            # )
            ax.plot(
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_corrected"
                ],
                c="blue",
                lw=1.2,
                ls=":",
                label="2D corrected ({:.2f})".format(mape_corrected),
            )
            ax.fill_between(
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

            ax.plot(
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_corrected_3d"
                ],
                c="purple",
                ls=":",
                lw=1.2,
                label="3D corrected ({:.2f})".format(mape_corrected_3d),
            )
            ax.fill_between(
                interp_outputs_test_df.loc[input_ind, int_version]["lag"],
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_corrected_3d_lower"
                ],
                interp_outputs_test_df.loc[input_ind, int_version][
                    "classical_corrected_3d_upper"
                ],
                color="purple",
                alpha=0.2,
            )

            ax.semilogx()
            ax.semilogy()
            ax.set_ylim(1e-2, 1e1)
            ax.legend(loc="lower right")

            # if log is True:
            c = ax0.pcolormesh(
                heatmap_bin_edges[0],
                heatmap_bin_edges[1] * 100,  # convert to % Missing
                heatmap_bin_vals.T,
                cmap="bwr",
            )
            # fig.colorbar(c, ax=ax0, label="MPE")
            c.set_clim(-100, 100)
            c.set_facecolor("black")
            # ax0.set_xlabel("Lag")
            ax0.plot(
                interp_outputs_test_df.loc[input_ind, int_version]["lag"],
                interp_outputs_test_df.loc[input_ind, int_version][missing_measure]
                * 100,
                c="black",
            )
            alphabet = "abcdefghijklmnopqrstuvwxyz"
            annotate_curve(
                ax0,
                interp_outputs_test_df.loc[input_ind, int_version]["lag"],
                interp_outputs_test_df.loc[input_ind, int_version][missing_measure]
                * 100,
                f"{alphabet[i]}",
                offset_scaling=(0.8, -0.3),
            )
            ax.annotate(
                f"{alphabet[i]}: {interp_outputs_test_df.loc[input_ind, int_version]['missing_prop_overall'].mean()*100:.1f}% missing overall",
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(0.1, 0.9),
                textcoords="axes fraction",
                transform=ax.transAxes,
                c="black",
                fontsize=12,
                bbox=dict(facecolor="lightgrey", edgecolor="white", boxstyle="round"),
            )
            ax.set_xlabel("Lag ($\\tau$)")
            # ax0 = sf.plot_heatmap(
            #     heatmap_bin_vals_log,
            #     heatmap_bin_edges_log,
            #     missing_measure=missing_measure,
            #     log=True,
            #     overlay_x=df_to_plot["lag"],
            #     overlay_y=df_to_plot[missing_measure],
            #     subplot=ax0,
            # )
        # axs[0, 1].set_title("Applying correction factor")
        # axs[0, 0].set_title("Extracting correction factor")
        # axs[n_versions_to_plot - 1, 0].set_xlabel("Lag ($\\tau$)")
        ax0.set_xlabel("Lag ($\\tau$)")
        ax0.set_ylabel("% pairs missing")
        ax0.set_xscale("log")
        ax0.set_ylim(0, 100)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(
            save_dir
            + f"sf_i_{input_ind}_classical_lint_corrected_b_{n_bins}_2d_external.png"
        )
        plt.close()

print("Done!")
print("\nCurrent time:", datetime.now())

##################################

# Plot bar plots of test set errors

# LATEST: 50 x 1040 intervals

# Create the DataFrame
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
#         25
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
#         20.1,
#         13.4,
#         25.2,
#         16.1,
#         36.6,
#         21.4,
#         # corrected results for various bins and dimensions
#         12.96,  # 15
#         12.92,  # 20
#         12.89, #25
#         11.27,  # 15
#         11.13,  # 15
#         11.07,  # 20
#     ],
# }

# df = pd.DataFrame(data)

# # Set the plot size
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
