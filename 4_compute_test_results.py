# # STEP 4: FOR ALL INTERVALS IN TEST SET: get overall test set results


import pickle
import numpy as np
import src.sf_funcs as sf
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import sys

sns.set_theme(style="whitegrid", font_scale=1.5)
# plt.rcParams.update({"font.size": 16})

# Import all corrected (test) files
spacecraft = sys.argv[1]
n_bins = 10

if spacecraft == "psp":
    input_file_list = sorted(glob.glob("data/processed/psp/test/psp_*_corrected.pkl"))
elif spacecraft == "wind":
    input_file_list = sorted(glob.glob("data/processed/wind/wi_*_corrected.pkl"))
else:
    raise ValueError("Spacecraft must be 'psp' or 'wind'")

(
    files_metadata,
    ints_metadata,
    ints,
    ints_gapped_metadata,
    ints_gapped,
    sfs,
    sfs_gapped_corrected,
) = sf.load_and_concatenate_dataframes(input_file_list)

print(
    "Successfully read in and concatenated {} files, starting with {}".format(
        len(input_file_list), input_file_list[0]
    )
)
print("Now proceeding to calculate overall test set statistics")
# Also get the 2d heatmap for final case study correction figure, by undoing the above operation

with open(f"data/processed/heatmap_2d_{n_bins}bins.pkl", "rb") as f:
    data = pickle.load(f)

heatmap_bin_vals_2d = data["heatmap_bin_vals_2d"]
heatmap_bin_edges_2d = data["heatmap_bin_edges_2d"]

# Export final overall dataframes, combined from all outputs
output_file_path = f"data/processed/test_corrected_{spacecraft}.pkl"

with open(output_file_path, "wb") as f:
    pickle.dump(
        {
            "files_metadata": files_metadata,
            "ints_metadata": ints_metadata,
            "ints": ints,
            "ints_gapped_metadata": ints_gapped_metadata,
            "ints_gapped": ints_gapped,
            "sfs": sfs,
            "sfs_gapped": sfs_gapped_corrected,
        },
        f,
    )


# Box plots


# ints_gapped_metadata.groupby("gap_handling")[["missing_percent_overall", "slope", "slope_pe", "mpe", "mape"]].agg(["mean", "median", "std", "min", "max"])

# Assuming ints_gapped_metadata is your DataFrame
# Define the list of columns to plot
columns = ["mpe", "mape", "slope_pe", "slope_ape"]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Flatten the 2D array of axes for easy iteration
axes = axes.flatten()

custom_order = ["naive", "lint", "corrected_2d", "corrected_3d"]
colors = ["indianred", "grey", "royalblue", "purple"]

# Create boxplots for each column
for col, ax in zip(columns, axes):
    data_to_plot = [
        ints_gapped_metadata[ints_gapped_metadata["gap_handling"] == method][col]
        for method in custom_order
    ]
    box = ax.boxplot(data_to_plot, patch_artist=True)

    # Set colors for the boxes

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)

    # Set colors for the median lines
    median_color = "black"
    for median in box["medians"]:
        median.set_color(median_color)
        median.set_linewidth(2)  # optional: set line width to make it more prominent

    ax.set_title(f"{col}")
    ax.set_ylabel(f"{col}")
    ax.set_xticklabels(custom_order)

# Adjust layout
plt.tight_layout()
plt.suptitle("")  # Remove the default title to avoid overlap
plt.savefig(f"plots/temp/test_{spacecraft}_boxplots.jpg", bbox_inches="tight")


# Regression lines


# Make scatterplot of mape vs. missing_percent, coloured by gap handling
palette = dict(zip(custom_order, colors))

# Plotting the MAPE vs. missing percentage
fig, ax = plt.subplots(1, 2, figsize=(18, 5))
sns.scatterplot(
    data=ints_gapped_metadata,
    x="missing_percent_overall",
    y="mape",
    hue="gap_handling",
    palette=palette,
    ax=ax[0],
)
ax[0].legend(title="Gap handling method")

# Add regression lines for each group
unique_gap_handling = ints_gapped_metadata["gap_handling"].unique()

for gap_handling_method in unique_gap_handling:
    subset = ints_gapped_metadata[
        ints_gapped_metadata["gap_handling"] == gap_handling_method
    ]
    sns.regplot(
        data=subset,
        x="missing_percent_overall",
        y="mape",
        scatter=False,
        color=palette[gap_handling_method],
        label=gap_handling_method,
        ci=None,
        ax=ax[0],
    )

sns.scatterplot(
    data=ints_gapped_metadata,
    x="missing_percent_overall",
    y="slope_ape",
    hue="gap_handling",
    palette=palette,
    ax=ax[1],
)

# Add regression lines for each group

unique_gap_handling = ints_gapped_metadata["gap_handling"].unique()

for gap_handling_method in unique_gap_handling:
    subset = ints_gapped_metadata[
        ints_gapped_metadata["gap_handling"] == gap_handling_method
    ]
    sns.regplot(
        data=subset,
        x="missing_percent_overall",
        y="slope_ape",
        scatter=False,
        color=palette[gap_handling_method],
        label=gap_handling_method,
        ci=None,
        ax=ax[1],
    )

ax[0].set_xlabel("Missing %")
ax[0].set_ylabel("SF MAPE")
ax[0].set_title("Overall SF estimation error")
ax[1].set_xlabel("Missing %")
ax[1].set_ylabel("Slope APE")
ax[1].set_title("Inertial range slope estimation error")

plt.savefig(f"plots/temp/test_{spacecraft}_scatterplots.jpg", bbox_inches="tight")


# Error trendlines


for gap_handling in sfs_gapped_corrected.gap_handling.unique():
    sf.plot_error_trend_line(
        sfs_gapped_corrected[sfs_gapped_corrected["gap_handling"] == gap_handling],
        estimator="sf_2",
        title=f"SF estimation error ({gap_handling}) vs. lag and global sparsity",
        y_axis_log=True,
    )
    plt.savefig(
        f"plots/temp/test_{spacecraft}_error_trend_{gap_handling}.jpg",
        bbox_inches="tight",
    )


# CASE STUDY PLOTS

# Pre-correction case studies

# Parameters for the 3 case study plots

print(ints_gapped_metadata)

int_index = 0  # We will be selecting the first interval for each file

for file_index_selected in range(2):
    file_index = ints_gapped_metadata["file_index"].unique()[file_index_selected]
    print(
        "Currenting making plots for file index", file_index, "and interval", int_index
    )

    fig, ax = plt.subplots(2, 3, figsize=(16, 2 * 3))

    for version in range(2):
        # ax[version, 0].plot(ints[0][int_index][0].values, c="grey")
        # Not currently plotting due to indexing issue: need to be able to index
        # on both file_index and int_index
        ax[version, 0].plot(
            ints_gapped.loc[
                (ints_gapped["file_index"] == file_index)
                & (ints_gapped["int_index"] == int_index)
                & (ints_gapped["version"] == version)
                & (ints_gapped["gap_handling"] == "lint"),
                "B",
            ].values,
            c="black",
        )

        # Put missing_percent_overall in the title
        ax[version, 0].set_title(
            f"{ints_gapped_metadata.loc[(ints_gapped_metadata['file_index']==file_index) & (ints_gapped_metadata['int_index']==int_index) & (ints_gapped_metadata['version']==version) & (ints_gapped_metadata['gap_handling']=='lint'), 'missing_percent_overall'].values[0]:.1f}% missing"
        )

        # Plot the SF
        ax[version, 1].plot(
            sfs.loc[
                (sfs["file_index"] == file_index) & (sfs["int_index"] == int_index),
                "lag",
            ],
            sfs.loc[
                (sfs["file_index"] == file_index) & (sfs["int_index"] == int_index),
                "sf_2",
            ],
            c="grey",
            label="True",
            lw=5,
        )

        ax[version, 1].plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "naive"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "naive"),
                "sf_2",
            ],
            c="indianred",
            label="Naive ({:.1f})".format(
                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == file_index)
                    & (ints_gapped_metadata["int_index"] == int_index)
                    & (ints_gapped_metadata["version"] == version)
                    & (ints_gapped_metadata["gap_handling"] == "naive"),
                    "mape",
                ].values[0]
            ),
        )

        ax[version, 1].plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "lint"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "lint"),
                "sf_2",
            ],
            c="black",
            label="LINT ({:.1f})".format(
                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == file_index)
                    & (ints_gapped_metadata["int_index"] == int_index)
                    & (ints_gapped_metadata["version"] == version)
                    & (ints_gapped_metadata["gap_handling"] == "lint"),
                    "mape",
                ].values[0]
            ),
        )

        # Plot the sf_2_pe
        ax[version, 2].plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "lint"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "lint"),
                "sf_2_pe",
            ],
            c="black",
        )
        ax[version, 2].plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "naive"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "naive"),
                "sf_2_pe",
            ],
            c="indianred",
        )

        # plot sample size n on right axis
        ax2 = ax[version, 2].twinx()
        ax2.plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "naive"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "naive"),
                "missing_percent",
            ],
            c="grey",
        )

        # Label the axes
        ax[1, 0].set_xlabel("Time")
        ax[version, 0].set_ylabel("$B$ (normalised)")
        ax[1, 1].set_xlabel("Lag ($\\tau$)")
        ax[version, 1].set_ylabel("SF")
        ax[1, 2].set_xlabel("Lag ($\\tau$)")
        ax[version, 2].set_ylabel("% error")
        ax2.set_ylabel("% missing", color="grey")
        ax2.tick_params(axis="y", colors="grey")
        ax2.set_ylim(0, 100)

        # ax[version, 2].axhline(0, c="black", linestyle="--")
        ax[version, 2].set_ylim(-100, 100)

        ax[version, 1].set_xscale("log")
        ax[version, 1].set_yscale("log")
        ax[version, 2].set_xscale("log")
        ax[version, 1].legend(fontsize=12)
        [ax[0, i].set_xticklabels([]) for i in range(3)]

    # Add titles
    ax[0, 1].set_title("Structure function estimates")
    ax[0, 2].set_title("SF % error and % pairs missing")
    plt.subplots_adjust(wspace=0.4)

    plt.savefig(
        f"plots/temp/test_{spacecraft}_case_study_gapping_{file_index}_{int_index}.jpg",
        bbox_inches="tight",
    )

    # 5e. Corrected case studies

    # Annotate each heatmap trace with info
    def annotate_curve(ax, x, y, text, offset_scaling=(0.3, 0.1)):
        # Find the index of y value closest to the median value
        idx = np.argmin(np.abs(y - np.percentile(y, 20)))

        # Coordinates of the point of maximum y value
        x_max = x.iloc[idx]
        y_max = y.iloc[idx]

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
            bbox=dict(
                facecolor="white", edgecolor="white", boxstyle="round", alpha=0.7
            ),
            fontsize=20,
        )

    fig = plt.figure(figsize=(20, 6))

    # Create a GridSpec layout with specified width ratios and horizontal space
    gs1 = GridSpec(1, 1, left=0.06, right=0.35)
    gs2 = GridSpec(1, 2, left=0.43, right=0.99, wspace=0)

    # Create subplots
    ax0 = fig.add_subplot(gs1[0, 0])
    ax1 = fig.add_subplot(gs2[0, 0])

    for version in range(2):
        if version == 0:
            ax = ax1
            ax.set_ylabel("SF")
        else:
            ax = fig.add_subplot(gs2[0, version], sharey=ax1)
            plt.setp(ax.get_yticklabels(), visible=False)

        ax.plot(
            sfs[(sfs["file_index"] == file_index) & (sfs["int_index"] == int_index)][
                "lag"
            ],
            sfs[(sfs["file_index"] == file_index) & (sfs["int_index"] == int_index)][
                "sf_2"
            ],
            color="black",
            label="True",
            lw=5,
            alpha=0.5,
        )
        ax.plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "naive"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "naive"),
                "sf_2",
            ],
            color="red",
            lw=1,
            label="Naive ({:.1f})".format(
                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == file_index)
                    & (ints_gapped_metadata["int_index"] == int_index)
                    & (ints_gapped_metadata["version"] == version)
                    & (ints_gapped_metadata["gap_handling"] == "naive"),
                    "mape",
                ].values[0]
            ),
        )
        ax.plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "lint"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "lint"),
                "sf_2",
            ],
            color="black",
            lw=1,
            label="Linear interp. ({:.1f})".format(
                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == file_index)
                    & (ints_gapped_metadata["int_index"] == int_index)
                    & (ints_gapped_metadata["version"] == version)
                    & (ints_gapped_metadata["gap_handling"] == "lint"),
                    "mape",
                ].values[0]
            ),
        )
        ax.plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
                "sf_2",
            ],
            color="blue",
            lw=1,
            label="2D corrected ({:.1f})".format(
                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == file_index)
                    & (ints_gapped_metadata["int_index"] == int_index)
                    & (ints_gapped_metadata["version"] == version)
                    & (ints_gapped_metadata["gap_handling"] == "corrected_2d"),
                    "mape",
                ].values[0]
            ),
        )
        ax.plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
                "sf_2",
            ],
            color="purple",
            lw=1,
            label="3D corrected ({:.1f})".format(
                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == file_index)
                    & (ints_gapped_metadata["int_index"] == int_index)
                    & (ints_gapped_metadata["version"] == version)
                    & (ints_gapped_metadata["gap_handling"] == "corrected_3d"),
                    "mape",
                ].values[0]
            ),
        )
        ax.fill_between(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
                "sf_2_lower",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_2d"),
                "sf_2_upper",
            ],
            color="blue",
            alpha=0.2,
        )

        missing = ints_gapped_metadata.loc[
            (ints_gapped_metadata["file_index"] == file_index)
            & (ints_gapped_metadata["int_index"] == int_index)
            & (ints_gapped_metadata["version"] == version),
            "missing_percent_overall",
        ].values

        ax.legend(loc="lower right", fontsize=16)
        ax.semilogx()
        ax.semilogy()

        # PLOTTING HEATMAP IN FIRST PANEL

        c = ax0.pcolormesh(
            heatmap_bin_edges_2d[0],
            heatmap_bin_edges_2d[1],  # convert to % Missing
            heatmap_bin_vals_2d.T,
            cmap="bwr",
        )
        # fig.colorbar(c, ax=ax0, label="MPE")
        c.set_clim(-100, 100)
        c.set_facecolor("black")
        ax0.set_xlabel("Lag ($\\tau$)")
        ax0.plot(
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "corrected_3d"),
                "missing_percent",
            ],
            c="black",
            lw=3,
        )

        # Label test intervals with letters
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        annotate_curve(
            ax0,
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "lint"),
                "lag",
            ],
            sfs_gapped_corrected.loc[
                (sfs_gapped_corrected["file_index"] == file_index)
                & (sfs_gapped_corrected["int_index"] == int_index)
                & (sfs_gapped_corrected["version"] == version)
                & (sfs_gapped_corrected["gap_handling"] == "lint"),
                "missing_percent",
            ],
            f"{alphabet[version]}",
            offset_scaling=(0.8, -0.3),
        )

        ax.annotate(
            f"{alphabet[version]}: {float(missing[0]):.1f}% missing overall",
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(0.1, 0.9),
            textcoords="axes fraction",
            transform=ax.transAxes,
            c="black",
            fontsize=18,
            bbox=dict(facecolor="lightgrey", edgecolor="white", boxstyle="round"),
        )

        ax.set_xlabel("Lag ($\\tau$)")

        print(f"\nStats for interval version {alphabet[version]}:\n")
        print(
            ints_gapped_metadata.loc[
                (ints_gapped_metadata["file_index"] == file_index)
                & (ints_gapped_metadata["int_index"] == int_index)
                & (ints_gapped_metadata["version"] == version)
            ][
                [
                    "file_index",
                    "int_index",
                    "version",
                    "missing_percent_overall",
                    "gap_handling",
                    "mape",
                    "slope_pe",
                ]
            ]
        )

    ax0.set_xscale("log")
    ax0.set_xlabel("Lag ($\\tau$)")
    ax0.set_ylabel("% pairs missing")
    ax0.set_ylim(0, 100)

    plt.savefig(
        f"plots/temp/test_{spacecraft}_case_study_correcting_{file_index}_{int_index}.jpg",
        bbox_inches="tight",
    )
