import pickle
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.5)
# Because Rāpoi can't handle latex apparently
# plt.rcParams.update(
#     {
#         "text.usetex": False,
#         "mathtext.fontset": "stix",  # Set the font to use for math
#         "font.family": "serif",  # Set the default font family
#         "font.size": 11,
#     }
# )

# Set seed for reproducibility
np.random.seed(42)


def compute_sf(
    data,
    lags,
    powers=[2],
    retain_increments=False,
    alt_estimators=True,
    pwrl_range=None,
):
    """
    Routine to compute the increments of a time series and then the mean (structure function) and standard deviation
    of the PDF of these increments, raised to the specified powers.
    Input:
            data: pd.DataFrame of data to be analysed. Must have shape (1, N) or (3, N)
            lags: The array consisting of lags, being the number of points to shift each of the series
            powers: The array consisting of the Structure orders to perform on the series for all lags
    Output:
            df: The DataFrame containing  corresponding to lags for each order in powers
    """
    # run through lags and powers so that for each power, run through each lag
    df = {}

    if data.shape[1] == 1:
        ax = data.iloc[:, 0].copy()
        for i in powers:
            array = []
            mean_array = []
            mapd_array = []
            std_array = []
            N_array = []
            for lag in lags:
                lag = int(lag)
                dax = np.abs(ax.shift(-lag) - ax)
                strct = dax.pow(i)
                array += [strct.values]
                strct_mean = strct.mean()
                if dax.isnull().sum() != len(dax):
                    # Otherwise this func will raise an error
                    median_abs_diff = np.nanmedian(dax)
                else:
                    median_abs_diff = np.nan
                mean_array += [strct_mean]
                mapd_array += [median_abs_diff]
                strct_std = strct.std()
                std_array += [strct_std]

                N = dax.notnull().sum()
                N_array += [N]

                df["lag"] = lags
                df["n"] = N_array
                df["sf_" + str(i)] = mean_array
                df["sf_" + str(i) + "_se"] = np.array(std_array) / np.sqrt(N_array)
                if retain_increments is True:
                    df["diffs_" + str(i)] = array
                    df["diffs_" + str(i) + "_sd"] = std_array

    else:
        raise ValueError(
            "This version only accepts scalar series: data must be a pd.DataFrame of shape (1, N)"
        )

    df = pd.DataFrame(df, index=lags)
    if alt_estimators is True:
        df["mapd"] = mapd_array
        df["sf_2_ch"] = df["sf_0.5"] ** 4 / (0.457 + (0.494 / df["n"]))
        df["sf_2_dowd"] = (df["mapd"] ** 2) * 2.198
    # calculate sample size as a proportion of the maximum sample size (for that lag)
    df.insert(2, "missing_percent", 100 * (1 - (df["n"] / (len(ax) - df.index))))

    if pwrl_range is not None:
        # Fit a line to the log-log plot of the structure function over the given range
        min, max = pwrl_range[0], pwrl_range[1]

        slope = np.polyfit(
            np.log(df.loc[min:max, "lag"]),
            np.log(df.loc[min:max, "sf_2"]),
            1,
        )[0]

        return df, slope

    else:
        return df


def get_lag_vals_list(df, value_name="sq_diffs"):
    lag_vals_wide = pd.DataFrame(df[value_name].tolist(), index=df.index)
    lag_vals_wide.reset_index(inplace=True)  # Make the index a column
    lag_vals_wide.rename(columns={"index": "lag"}, inplace=True)
    lag_vals = pd.melt(
        lag_vals_wide, id_vars=["lag"], var_name="index", value_name=value_name
    )
    return lag_vals


def plot_sample(
    good_input,
    good_output,
    other_inputs,
    other_outputs,
    input_ind=0,
    input_versions=3,  # Either number of versions to plot or a list of versions to plot
    linear=True,
    estimator_list=["sf_2"],  # sf_2, ch, dowd
    title="SF estimation subject to missing data",
):
    if linear is False:
        ncols = 3
    else:
        ncols = 4

    # Check if n is an integer
    if not isinstance(input_versions, int):
        n = len(input_versions)
        fig, ax = plt.subplots(
            n,
            ncols,
            figsize=(ncols * 5, n * 3),
            sharex="col",
            gridspec_kw={"hspace": 0.2},
        )
        other_inputs_plot = [other_inputs[input_ind][i] for i in input_versions]
        other_outputs_plot = [other_outputs[input_ind][i] for i in input_versions]
    else:
        n = input_versions
        fig, ax = plt.subplots(
            n,
            ncols,
            figsize=(ncols * 5, n * 3),
            sharex="col",
            gridspec_kw={"hspace": 0.2},
        )
        # Before plotting, sort the n bad inputs by missing proportion
        other_inputs_plot = other_inputs[input_ind][:n]
        other_outputs_plot = other_outputs[input_ind][:n]

    sparsities = [df["missing_percent_overall"].values[0] for df in other_outputs_plot]

    sorted_lists = zip(*sorted(zip(sparsities, other_inputs_plot)))
    sparsities_ordered, other_inputs_plot = sorted_lists

    sorted_lists = zip(*sorted(zip(sparsities, other_outputs_plot)))
    sparsities_ordered, other_outputs_plot = sorted_lists

    ax[0, 0].set_title("Interpolated time series")
    ax[0, ncols - 2].set_title("$S_2(\\tau)$")
    ax[0, ncols - 1].set_title("Estimation error")

    for i in range(n):
        missing = other_outputs_plot[i]["missing_percent_overall"].values[0]
        # missing = np.isnan(ts_plot).sum() / len(ts_plot)
        ax[i, 0].plot(good_input[input_ind].values, color="grey", lw=0.8)
        ax[i, 0].plot(other_inputs_plot[i], color="black", lw=0.8)

        # Add the missing % as an annotation in the top left
        ax[i, 0].annotate(
            f"{missing*100:.2f}% missing",
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(0.05, 0.9),
            textcoords="axes fraction",
            transform=ax[i, 0].transAxes,
            c="black",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        )

        colors = ["C0", "darkred", "olivedrab"]
        pos_y = [0.9, 0.8, 0.7]
        for est_ind, estimator in enumerate(estimator_list):
            mape = other_outputs_plot[i][estimator + "_pe"].abs().mean()

            ax[i, 1].annotate(
                "MAPE = {:.2f}".format(mape),
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(0.05, pos_y[est_ind]),
                textcoords="axes fraction",
                transform=ax[i, 1].transAxes,
                color=colors[est_ind],
                bbox=dict(
                    facecolor="white",
                    edgecolor="white",
                    boxstyle="round",
                    # linestyle=":",
                ),
            )

            ax[i, ncols - 1].plot(
                other_outputs_plot[i]["missing_prop"] * 100,
                color="black",
                label="% pairs missing",
            )
            ax[i, ncols - 1].semilogx()
            ax[i, ncols - 1].set_ylim(0, 100)

            ax2 = ax[i, ncols - 1].twinx()
            ax2.tick_params(axis="y", colors="C0")
            ax2.plot(
                other_outputs_plot[i][estimator + "_pe"],
                color=colors[est_ind],
                label="% error",
                lw=0.8,
            )

            ax2.semilogx()
            ax2.set_ylim(-100, 100)
            ax2.axhline(0, color="C0", linestyle="--")
            if i == 0:
                ax2.annotate(
                    "% error",
                    xy=(1, 1),
                    xycoords="axes fraction",
                    xytext=(0.75, 0.9),
                    textcoords="axes fraction",
                    transform=ax[i, 0].transAxes,
                    c="C0",
                    bbox=dict(
                        facecolor="white", edgecolor="grey", boxstyle="round", alpha=0.7
                    ),
                )

            # Plot scatter plot and line plot for both log-scale and linear-scale
            for j in range(ncols - 2):
                j += 1

                ax[i, j].plot(
                    good_output[input_ind]["lag"],
                    good_output[input_ind][estimator],
                    color=colors[est_ind],
                    alpha=0.3,
                    linewidth=3.5,
                    label=estimator,
                )
                ax[i, j].plot(
                    other_outputs_plot[i]["lag"],
                    other_outputs_plot[i][estimator],
                    color=colors[est_ind],
                    linewidth=0.8,
                    # ls=":",
                    # label=f": {estimator}",
                )
                suffix = ""  # for the title
                # Get lag vals
                if (
                    "sq_diffs" in other_outputs_plot[i].columns
                    and len(good_input[input_ind]) < 3000
                ):
                    other_lag_vals = get_lag_vals_list(other_outputs_plot[i])
                    ax[i, j].scatter(
                        other_lag_vals["lag"],
                        other_lag_vals["sq_diffs"],
                        alpha=0.005,
                        s=1,
                    )
                    suffix = " + squared diffs"

                # Plot "confidence region" of +- x SEs
                # if estimator == "sf_2":
                #     x = 3
                #     ax[i, j].fill_between(
                #         other_outputs_plot[i]["lag"],
                #         np.maximum(
                #             other_outputs_plot[i]["sf_2"]
                #             - x * other_outputs_plot[i]["sf_2_se"],
                #             0,
                #         ),
                #         other_outputs_plot[i]["sf_2"]
                #         + x * other_outputs_plot[i]["sf_2_se"],
                #         color="C0",
                #         alpha=0.4,
                #         label=f"$\pm$ {x} SE",
                #     )

                ax[i, j].set_ylim(5e-3, 5e0)

        if linear is True:
            ax[i, 2].semilogx()
            ax[i, 2].semilogy()
        else:
            ax[i, 1].semilogx()
            ax[i, 1].semilogy()

    ax[n - 1, 0].set_xlabel("Time")
    for i in range(1, ncols):
        ax[n - 1, i].set_xlabel("Lag ($\\tau$)")
    # Remove x-axis labels for all but the bottom row
    # for i in range(n):
    #     for j in range(ncols):
    #         if i < n:
    #             ax[i, j].set_xticklabels([])

    # ax[0, ncols - 1].axhline(0, color="black", linestyle="--")
    ax[0, ncols - 1].semilogx()
    # ax[0, 1].legend(loc="lower right", frameon=True)

    ax[0, ncols - 1].annotate(
        "% pairs missing",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.05, 0.9),
        textcoords="axes fraction",
        transform=ax[0, 2].transAxes,
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round", alpha=0.5),
    )

    if linear is True:
        ax[0, 1].set_title("$S_2(\\tau)$" + suffix)

    # Add overall title
    # fig.suptitle(title, size=16)

    # plt.show()


# Load in each pickle file psp_dataframes_0X.pkl and concatenate them
# into one big dataframe for each of the four dataframes


def load_and_concatenate_dataframes(pickle_files):
    concatenated_dataframes = {
        "files_metadata": [],
        "ints_metadata": [],
        "ints": [],
        "ints_gapped_metadata": [],
        "ints_gapped": [],
        "sfs": [],
        "sfs_gapped": [],
    }

    for file in pickle_files:
        with open(file, "rb") as f:
            data = pickle.load(f)
            for key in concatenated_dataframes.keys():
                concatenated_dataframes[key].append(data[key])

    for key in concatenated_dataframes.keys():
        if (
            key == "ints"
        ):  # Ints is a list of list of pd.Series, not a list of dataframes
            concatenated_dataframes[key] = concatenated_dataframes[key]
        else:
            concatenated_dataframes[key] = pd.concat(
                concatenated_dataframes[key], ignore_index=True
            )

    # Access the concatenated DataFrames
    files_metadata = concatenated_dataframes["files_metadata"]
    ints_metadata = concatenated_dataframes["ints_metadata"]
    ints = concatenated_dataframes["ints"]
    ints_gapped_metadata = concatenated_dataframes["ints_gapped_metadata"]
    ints_gapped = concatenated_dataframes["ints_gapped"]
    sfs = concatenated_dataframes["sfs"]
    sfs_gapped = concatenated_dataframes["sfs_gapped"]

    return (
        files_metadata,
        ints_metadata,
        ints,
        ints_gapped_metadata,
        ints_gapped,
        sfs,
        sfs_gapped,
    )


def plot_error_trend_line(
    other_outputs_df,
    estimator="sf_2",
    title="SF estimation error vs. lag and global sparsity",
    y_axis_log=False,
):
    fig, ax = plt.subplots(figsize=(8, 4), tight_layout=True)
    plt.title(title)
    # plt.plot(lag_error_mean_i, color="black", lw=3)
    plt.scatter(
        other_outputs_df["lag"],
        other_outputs_df[estimator + "_pe"],
        c=other_outputs_df["missing_percent"],
        s=0.03,
        alpha=0.4,
        cmap="plasma",
    )
    median_error = other_outputs_df.groupby("lag")[estimator + "_pe"].median()
    mean_error = other_outputs_df.groupby("lag")[estimator + "_pe"].mean()
    plt.plot(median_error, color="g", lw=2, label="Median")
    plt.plot(mean_error, color="c", lw=2, label="Mean")

    # plt.annotate(
    #     "MAPE = {0:.2f}".format(other_outputs_df[estimator + "_pe"].abs().mean()),
    #     xy=(1, 1),
    #     xycoords="axes fraction",
    #     xytext=(0.1, 0.9),
    #     textcoords="axes fraction",
    #     c="black",
    # )

    cb = plt.colorbar()
    cb.set_label("% missing overall")
    # Change range of color bar
    plt.hlines(0, 1, other_outputs_df.lag.max(), color="black", linestyle="--")
    plt.clim(0, 100)
    plt.ylim(-2e2, 6e2)
    plt.semilogx()
    if y_axis_log is True:
        plt.yscale("symlog", linthresh=1e2)
    plt.xlabel("Lag ($\\tau$)")
    plt.ylabel("% error")
    plt.legend(loc="upper left")
    # plt.show()


def plot_error_trend_scatter(
    bad_outputs_df, interp_outputs_df, title="Overall % error vs. sparsity"
):
    fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
    sfn_mape = bad_outputs_df.groupby("missing_percent_overall")["sf_2_pe"].agg(
        lambda x: np.mean(np.abs(x))
    )

    sfn_mape_i = interp_outputs_df.groupby("missing_percent_overall")["sf_2_pe"].agg(
        lambda x: np.mean(np.abs(x))
    )
    plt.scatter(sfn_mape.index, sfn_mape.values, c="C0", label="No handling", alpha=0.5)
    plt.scatter(
        sfn_mape_i.index,
        sfn_mape_i.values,
        c="purple",
        label="Linear interp.",
        alpha=0.1,
    )

    # Add regression lines
    import statsmodels.api as sm

    x = sm.add_constant(sfn_mape.index)

    model = sm.OLS(sfn_mape.values, x)
    results = model.fit()
    plt.plot(sfn_mape.index, results.fittedvalues, c="C0")

    x_i = sm.add_constant(sfn_mape_i.index)
    model = sm.OLS(sfn_mape_i.values, x_i)
    results = model.fit()
    plt.plot(sfn_mape_i.index, results.fittedvalues, c="purple")

    plt.xlabel("Fraction of data missing overall")
    plt.ylabel("MAPE")
    plt.ylim(0, 120)
    plt.title(title)
    plt.legend()
    # plt.show()


def create_heatmap_lookup(inputs, missing_measure, num_bins=25, log=False):
    """Extract the mean error for each bin of lag and missing measure.
    Args:
        num_bins: The number of bins to use in each direction (x and y)
    """

    x = inputs["lag"]
    y = inputs[missing_measure]

    heatmap, xedges, yedges = np.histogram2d(
        x, y, bins=num_bins, range=[[0, inputs.lag.max()], [0, 100]]
    )

    if log is True:
        xedges = (
            np.logspace(0, np.log10(inputs.lag.max()), num_bins + 1) - 0.01
        )  # so that first lag bin starts just before 1
        # y_bins = np.logspace(0, 2, num_bins) / 100 - 0.01
        # y_bins[-1] = 1
        xedges[-1] = inputs.lag.max() + 1

        # _, xedges, _ = np.histogram2d(
        #     x,
        #     y,
        #     bins=[x_bins, y_bins],
        # )

    data = {
        # Currently saving midpoints of bins to lookup table - could change to edges
        "lag": [],
        missing_measure: [],
        "n": [],
        "mpe": [],
        "mpe_sd": [],
        "pe_min": [],
        "pe_max": [],
    }
    # Calculate the mean value in each bin
    xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
    yidx = np.digitize(y, yedges) - 1  # as above
    means = np.full((num_bins, num_bins), fill_value=np.nan)
    counts = np.full((num_bins, num_bins), fill_value=np.nan)
    # upper = np.full((num_bins, num_bins), fill_value=np.nan)
    # lower = np.full((num_bins, num_bins), fill_value=np.nan)
    for i in range(num_bins):
        for j in range(num_bins):
            # If there are any values, calculate the mean for that bin
            if len(x[(xidx == i) & (yidx == j)]) > 0:
                # means[i, j] = np.mean(y[(xidx == i) & (yidx == j)])
                current_bin_vals = inputs["sf_2_pe"][(xidx == i) & (yidx == j)]
                mpe = np.nanmean(current_bin_vals)
                n = len(current_bin_vals)
                means[i, j] = mpe
                counts[i, j] = n

                # Calculate standard deviation for the lookup table too
                # / np.sqrt(
                #    len(inputs["sf_2_pe"][(xidx == i) & (yidx == j)])
                # )

                # upper[i, j] = lag_prop_mean + 3 * lag_prop_std_err
                # lower[i, j] = lag_prop_mean - 3 * lag_prop_std_err

                # Save values at midpoint of each bin
                data["lag"].append(0.5 * (xedges[i] + xedges[i + 1]))
                data[missing_measure].append(0.5 * (yedges[j] + yedges[j + 1]))
                data["n"].append(n)
                data["mpe"].append(mpe)
                data["mpe_sd"].append(np.nanstd(current_bin_vals))
                data["pe_min"].append(np.nanmin(current_bin_vals))
                data["pe_max"].append(np.nanmax(current_bin_vals))

    # Compute scaling factors
    data = pd.DataFrame(data)
    data["scaling"] = 1 / (1 + data["mpe"] / 100)
    # Reversed because min MPE is more negative
    data["scaling_lower"] = 1 / (1 + data["pe_max"] / 100)
    data["scaling_upper"] = 1 / (1 + data["pe_min"] / 100)

    return means, counts, [xedges, yedges], data


def create_heatmap_lookup_3D(inputs, missing_measure, num_bins=25, log=False):
    """Extract the mean error for each bin of lag and missing measure.
    Args:
        num_bins: The number of bins to use in each direction (x and y)
    """

    x = inputs["lag"]
    y = inputs[missing_measure]
    z = inputs["sf_2"]

    xedges = np.linspace(0, inputs.lag.max() + 1, num_bins + 1)  # Lags
    yedges = np.linspace(0, 100, num_bins + 1)  # Missing prop
    zedges = np.linspace(
        inputs["sf_2"].min(), inputs["sf_2"].max(), num_bins + 1
    )  # Power

    if log is True:
        xedges = (
            np.logspace(0, np.log10(inputs.lag.max()), num_bins + 1) - 0.01
        )  # so that first lag bin starts just before 1
        xedges[-1] = inputs.lag.max() + 1
        zedges = np.logspace(-2, 1, num_bins + 1)  # ranges from 0.01 to 10

    data = {
        "lag": [],
        missing_measure: [],
        "sf_2": [],
        "mpe": [],
        "mpe_sd": [],
        "pe_min": [],
        "pe_max": [],
    }
    # Calculate the mean value in each bin
    xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
    yidx = np.digitize(y, yedges) - 1  # as above
    zidx = np.digitize(z, zedges) - 1  # as above

    means = np.full((num_bins, num_bins, num_bins), fill_value=np.nan)
    for i in range(num_bins):
        for j in range(num_bins):
            for k in range(num_bins):
                # If there are any values, calculate the mean for that bin
                if len(x[(xidx == i) & (yidx == j) & (zidx == k)]) > 0:
                    # means[i, j] = np.mean(y[(xidx == i) & (yidx == j;ppoll;k.; mn)])
                    current_bin_vals = inputs["sf_2_pe"][
                        (xidx == i) & (yidx == j) & (zidx == k)
                    ]
                    # can simply get len of this array to get the number of values
                    mpe = np.nanmean(current_bin_vals)
                    means[i, j, k] = mpe

                    data["lag"].append(0.5 * (xedges[i] + xedges[i + 1]))
                    data[missing_measure].append(0.5 * (yedges[j] + yedges[j + 1]))
                    data["sf_2"].append(0.5 * (zedges[k] + zedges[k + 1]))

                    data["mpe"].append(mpe)
                    data["mpe_sd"].append(np.nanstd(current_bin_vals))
                    data["pe_min"].append(np.nanmin(current_bin_vals))
                    data["pe_max"].append(np.nanmax(current_bin_vals))

    # Compute scaling factors
    data = pd.DataFrame(data)
    data["scaling"] = 1 / (1 + data["mpe"] / 100)
    data["scaling_lower"] = 1 / (1 + data["pe_max"] / 100)
    data["scaling_upper"] = 1 / (1 + data["pe_min"] / 100)
    # data["scaling_lower"] = 1 / (1 + (data["mpe"] + 10 * data["mpe_sd"]) / 100)
    # data["scaling_upper"] = 1 / (1 + (data["mpe"] - 10 * data["mpe_sd"]) / 100)

    return means, [xedges, yedges, zedges], data


def plot_heatmap(
    means,
    edges,
    missing_measure,
    log,
    overlay_x=None,
    overlay_y=None,
    subplot=None,
    title="Correction factor extraction",
):
    """Plot the heatmap of the MPE values for each bin of lag and missing measure."""
    xedges = edges[0]
    yedges = edges[1]

    if subplot is not None:
        ax = subplot
    else:
        fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        means.T,
        origin="lower",
        cmap="bwr",
        interpolation="nearest",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect="auto",
    )

    if subplot is None:
        cbar = fig.colorbar(im, ax=ax, label="mpe", orientation="vertical")
    # Change range of colorbar
    im.set_clim(-100, 100)

    if log is True:
        ax.semilogx()
        ax.set_xlim(1, 250)

    ax.set_facecolor("black")  # So we can distinguish 0 means from missing means
    ax.set_ylabel(missing_measure)

    if subplot is None:
        ax.set_xlabel("lag")
        ax.set_title("SF estimation error using LINT")
        # plt.show()
    else:
        ax.set_title(title)
        if overlay_x is not None:
            ax.plot(overlay_x, overlay_y)
        return ax


def compute_scaling(bad_output, var, heatmap_vals, external_test_set=False):
    """
    Extracting values from each bin to create a look-up table. Note that due to binning we have to find the nearest value to get the corresponding MAPE for a given lag and proportion of pairs remaining. Using MPE as we want to maintaing the direction of the error for compensating.
    """
    df = heatmap_vals
    bad_output = bad_output.copy()  # To avoid SettingWithCopyWarning
    # If no nearest bin is found (len(nearest_row)=0), scaling will be 1 (the same)
    bad_output["scaling"] = 1
    bad_output["scaling_lower"] = 1
    bad_output["scaling_upper"] = 1

    for i, row in bad_output.iterrows():
        desired_prop = row[var]
        desired_lag = row["lag"]

        # Compute absolute differences
        # Using numpy arrays for speedup

        lag_diff = np.abs(df["lag"].to_numpy() - desired_lag)
        prop_diff = np.abs(df[var].to_numpy() - desired_prop)

        # Find the nearest row in Eucledian space
        # (note different scaling for lag and prop, but should be OK)
        combined_distance = np.sqrt(lag_diff**2 + prop_diff**2)
        # Get index of minimum value of the array
        nearest_index = np.argmin(
            combined_distance
        )  # Will pick first if multiple minima
        nearest_row = df.loc[nearest_index]

        if len(nearest_row) == 0:
            print("No nearest bin found!")
            continue

        else:
            result = nearest_row
            scaling = result["scaling"]
            scaling_lower = result["scaling_lower"]
            scaling_upper = result["scaling_upper"]

        bad_output.at[i, "scaling"] = scaling
        bad_output.at[i, "scaling_lower"] = scaling_lower
        bad_output.at[i, "scaling_upper"] = scaling_upper

    # if external_test_set is False:
    #     bad_output.loc[bad_output["sf_2_pe"] == 0, "scaling"] = (
    #         1  # Catching rows where there is no error (only for repeated data),
    #         # so scaling should be 1
    #     )

    bad_output["sf_2_corrected_2d"] = bad_output["sf_2"] * bad_output["scaling"]
    bad_output["sf_2_lower_corrected_2d"] = (
        bad_output["sf_2"] * bad_output["scaling_lower"]
    )
    bad_output["sf_2_upper_corrected_2d"] = (
        bad_output["sf_2"] * bad_output["scaling_upper"]
    )
    # Smoothing potentially jumpy correction
    bad_output["scaling_smoothed"] = (
        bad_output["scaling"].rolling(window=20, min_periods=1).mean()
    )
    bad_output["scaling_lower_smoothed"] = (
        bad_output["scaling_lower"].rolling(window=20, min_periods=1).mean()
    )
    bad_output["scaling_upper_smoothed"] = (
        bad_output["scaling_upper"].rolling(window=20, min_periods=1).mean()
    )
    bad_output["sf_2_corrected_2d_smoothed"] = (
        bad_output["sf_2"] * bad_output["scaling_smoothed"]
    )
    bad_output["sf_2_lower_corrected_2d_smoothed"] = (
        bad_output["sf_2"] * bad_output["scaling_lower_smoothed"]
    )
    bad_output["sf_2_upper_corrected_2d_smoothed"] = (
        bad_output["sf_2"] * bad_output["scaling_upper_smoothed"]
    )

    return bad_output


def compute_scaling_3d(bad_output, var, heatmap_vals, smoothing_method="linear"):
    """
    Extracting values from each bin to create a look-up table. Note that due to binning we have to find the nearest value to get the corresponding MAPE for a given lag and proportion of pairs remaining. Using MPE as we want to maintaing the direction of the error for compensating."""
    df = heatmap_vals
    bad_output = bad_output.copy()  # To avoid SettingWithCopyWarning

    # Precompute scaling factors
    df["scaling"] = 1 / (1 + df["mpe"] / 100)
    df["scaling_lower"] = 1 / (1 + (df["mpe"] + 10 * df["mpe_sd"]) / 100)
    df["scaling_upper"] = 1 / (1 + (df["mpe"] - 10 * df["mpe_sd"]) / 100)

    # If no nearest bin is found (len(nearest_row)=0), scaling will be 1 (the same)
    bad_output["scaling"] = 1
    bad_output["scaling_lower"] = 1
    bad_output["scaling_upper"] = 1

    for i, row in bad_output.iterrows():
        desired_prop = row[var]
        desired_lag = row["lag"]
        desired_sf_2 = row["sf_2"]

        # Compute absolute differences
        lag_diff = np.abs(df["lag"] - desired_lag)
        prop_diff = np.abs(df[var] - desired_prop)
        sf_2_diff = np.abs(df["sf_2"] - desired_sf_2)

        # Find the nearest row
        min_lag_diff = lag_diff.min()
        min_prop_diff = prop_diff.min()
        min_sf_2_diff = sf_2_diff.min()
        nearest_row = df.loc[
            (lag_diff == min_lag_diff)
            & (prop_diff == min_prop_diff)
            & (sf_2_diff == min_sf_2_diff)
        ]

        if len(nearest_row) == 0:
            # print("No nearest bin found!")
            continue

        # elif len(nearest_row) > 1:
        # print("More than one nearest bin found!")

        else:
            result = nearest_row.head(1)
            scaling = result["scaling"].values[0]
            scaling_lower = result["scaling_lower"].values[0]
            scaling_upper = result["scaling_upper"].values[0]

        bad_output.at[i, "scaling"] = scaling
        bad_output.at[i, "scaling_lower"] = scaling_lower
        bad_output.at[i, "scaling_upper"] = scaling_upper

    # bad_output.loc[bad_output["sf_2_pe"] == 0, "scaling"] = 1  # Catching 0 errors

    bad_output["sf_2_corrected_3d"] = bad_output["sf_2"] * bad_output["scaling"]
    bad_output["sf_2_lower_corrected_3d"] = (
        bad_output["sf_2"] * bad_output["scaling_lower"]
    )
    bad_output["sf_2_upper_corrected_3d"] = (
        bad_output["sf_2"] * bad_output["scaling_upper"]
    )
    # Smoothing potentially jumpy correction
    bad_output["scaling_3d_smoothed"] = (
        bad_output["scaling"].rolling(window=20, min_periods=1).mean()
    )

    bad_output["scaling_lower_3d_smoothed"] = (
        bad_output["scaling_lower"].rolling(window=20, min_periods=1).mean()
    )
    bad_output["scaling_upper_3d_smoothed"] = (
        bad_output["scaling_upper"].rolling(window=20, min_periods=1).mean()
    )
    bad_output["sf_2_corrected_3d_smoothed"] = (
        bad_output["sf_2"] * bad_output["scaling_3d_smoothed"]
    )
    bad_output["sf_2_lower_corrected_3d_smoothed"] = (
        bad_output["sf_2"] * bad_output["scaling_lower_3d_smoothed"]
    )
    bad_output["sf_2_upper_corrected_3d_smoothed"] = (
        bad_output["sf_2"] * bad_output["scaling_upper_3d_smoothed"]
    )
    return bad_output
