from matplotlib import pyplot as plt, tight_layout
import numpy as np
import pandas as pd

# Because Rāpoi can't handle latex apparently
plt.rcParams.update(
    {
        "text.usetex": False,
        "mathtext.fontset": "stix",  # Set the font to use for math
        "font.family": "serif",  # Set the default font family
        # "font.size": 10
    }
)

# Set seed for reproducibility
np.random.seed(42)


def compute_sf(data, lags, powers=[2], retain_increments=False):
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

            if i == 2:
                df["lag"] = lags
                df["n"] = N_array
                df["sosf"] = mean_array
                df["mapd"] = mapd_array
                df["sosf_se"] = np.array(std_array) / np.sqrt(N_array)
                if retain_increments is True:
                    df["sq_diffs"] = array

            else:
                df["n"] = N_array
                df[str(i) + "_mean"] = mean_array
                df[str(i) + "_std"] = std_array
                df[str(i) + "_std_err"] = np.array(std_array) / np.sqrt(N_array)
                if retain_increments is True:
                    df[str(i) + "_diffs"] = array

    else:
        raise ValueError(
            "This version only accepts scalar series: data must be a pd.DataFrame of shape (1, N)"
        )

    df = pd.DataFrame(df, index=lags)
    # calculate sample size as a proportion of the maximum sample size (for that lag)
    df["missing_prop"] = 1 - (df["n"] / (len(ax) - df.index))
    return df


def get_lag_vals_list(df):
    lag_vals_wide = pd.DataFrame(df["sq_diffs"].tolist(), index=df.index)
    lag_vals_wide.reset_index(inplace=True)  # Make the index a column
    lag_vals_wide.rename(columns={"index": "lag"}, inplace=True)
    lag_vals = pd.melt(
        lag_vals_wide, id_vars=["lag"], var_name="index", value_name="sq_diffs"
    )
    return lag_vals


def plot_sample(
    good_input,
    good_output,
    other_inputs,
    other_outputs,
    colour,
    input_ind=0,
    input_versions=3,  # Either number of versions to plot or a list of versions to plot
    linear=True,
):
    if linear is False:
        ncols = 3
    else:
        ncols = 4

    # Check if n is an integer
    if not isinstance(input_versions, int):
        n = len(input_versions)
        fig, ax = plt.subplots(n, ncols, figsize=(ncols * 5, n * 3))
        other_inputs_plot = [other_inputs[input_ind][i] for i in input_versions]
        other_outputs_plot = [other_outputs[input_ind][i] for i in input_versions]
    else:
        n = input_versions
        fig, ax = plt.subplots(n, ncols, figsize=(ncols * 5, n * 3))
        # Before plotting, sort the n bad inputs by missing proportion
        other_inputs_plot = other_inputs[input_ind][:n]
        other_outputs_plot = other_outputs[input_ind][:n]

    sparsities = [df["missing_prop_overall"].values[0] for df in other_outputs_plot]

    sorted_lists = zip(*sorted(zip(sparsities, other_inputs_plot)))
    sparsities_ordered, other_inputs_plot = sorted_lists

    sorted_lists = zip(*sorted(zip(sparsities, other_outputs_plot)))
    sparsities_ordered, other_outputs_plot = sorted_lists

    ax[0, 0].set_title("Input time series")
    ax[0, ncols - 2].set_title("SF$_2$ (log axes)")
    ax[0, ncols - 1].set_title("Estimation error")

    for i in range(n):
        missing = other_outputs_plot[i]["missing_prop_overall"].values[0]
        # missing = np.isnan(ts_plot).sum() / len(ts_plot)
        ax[i, 0].plot(good_input[input_ind].values, color="grey", lw=0.8)
        ax[i, 0].plot(other_inputs_plot[i], color="black", lw=0.8)

        # Add the missing % as an annotation in the top left
        ax[i, 0].annotate(
            f"{missing*100:.2f}% missing",
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(0.05, 0.85),
            textcoords="axes fraction",
            transform=ax[i, 0].transAxes,
            c="black",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        )

        mape = other_outputs_plot[i]["error_percent"].abs().mean()

        ax[i, 1].annotate(
            "MAPE = {:.2f}".format(mape),
            xy=(1, 1),
            xycoords="axes fraction",
            xytext=(0.05, 0.85),
            textcoords="axes fraction",
            transform=ax[i, 1].transAxes,
            c="black",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
        )

        ax[i, ncols - 1].plot(
            other_outputs_plot[i]["missing_prop"] * 100,
            color=colour,
            label="% pairs missing",
        )
        ax[i, ncols - 1].semilogx()
        ax[i, ncols - 1].set_ylim(0, 100)
        # Make the y-axis tick labels match the line color
        for tl in ax[i, ncols - 1].get_yticklabels():
            tl.set_color(colour)

        ax2 = ax[i, ncols - 1].twinx()
        # ax2.plot(sfn_pm["N"], color=colour, label="\# points")

        ax2.plot(other_outputs_plot[i]["error_percent"], color="black", label="% error")
        ax2.semilogx()
        ax2.set_ylim(-100, 100)
        ax2.axhline(0, color="black", linestyle="--")
        if i == 0:
            ax2.annotate(
                "% error",
                xy=(1, 1),
                xycoords="axes fraction",
                xytext=(0.75, 0.85),
                textcoords="axes fraction",
                transform=ax[i, 0].transAxes,
                c="black",
                bbox=dict(
                    facecolor="white", edgecolor="grey", boxstyle="round", alpha=0.7
                ),
            )

        # Plot scatter plot and line plot for both log-scale and linear-scale
        for j in range(ncols - 2):
            j += 1

            ax[i, j].plot(
                good_output[input_ind]["lag"],
                good_output[input_ind]["sosf"],
                color="grey",
                linewidth=2,
            )
            ax[i, j].plot(
                other_outputs_plot[i]["lag"],
                other_outputs_plot[i]["sosf"],
                color="black",
                linewidth=3,
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
                    c=colour,
                )
                suffix = " + squared diffs"

            # Plot "confidence region" of +- x SEs
            x = 5
            ax[i, j].fill_between(
                other_outputs_plot[i]["lag"],
                np.maximum(
                    other_outputs_plot[i]["sosf"]
                    - x * other_outputs_plot[i]["sosf_se"],
                    0,
                ),
                other_outputs_plot[i]["sosf"] + x * other_outputs_plot[i]["sosf_se"],
                color="lightgrey",
                alpha=0.6,
                label=f"$\pm$ {x} SE",
            )

            ax[i, j].set_ylim(1e-2, 2 * good_output[input_ind]["sosf"].max())

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
    for i in range(n):
        for j in range(ncols):
            if i < n:
                ax[i, j].set_xticklabels([])

    ax[0, ncols - 1].axhline(0, color="black", linestyle="--")
    ax[0, ncols - 1].semilogx()
    ax[0, 1].legend(loc="lower right", frameon=True)

    ax[0, ncols - 1].annotate(
        "% pairs missing",
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.05, 0.85),
        textcoords="axes fraction",
        transform=ax[i, 0].transAxes,
        c=colour,
        bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round", alpha=0.5),
    )

    if linear is True:
        ax[0, 1].set_title("SF$_2$" + suffix)

    # plt.show()


def plot_error_trend_line(
    other_outputs_df, title="SF estimation error vs. lag and global sparsity"
):
    fig, ax = plt.subplots(figsize=(6, 3))
    plt.title(title)
    # plt.plot(lag_error_mean_i, color="black", lw=3)
    plt.scatter(
        other_outputs_df["lag"],
        other_outputs_df["error_percent"],
        c=other_outputs_df["missing_prop_overall"],
        s=0.2,
        alpha=0.5,
        cmap="spring",
    )
    median_error = other_outputs_df.groupby("Lag")["error_percent"].median()
    mean_error = other_outputs_df.groupby("Lag")["error_percent"].mean()
    plt.plot(median_error, color="black", lw=2, label="Median")
    plt.plot(mean_error, color="blue", lw=2, label="Mean")

    plt.annotate(
        "MAPE = {0:.2f}".format(other_outputs_df["error_percent"].abs().mean()),
        xy=(1, 1),
        xycoords="axes fraction",
        xytext=(0.1, 0.9),
        textcoords="axes fraction",
        c="black",
    )

    cb = plt.colorbar()
    cb.set_label("% missing overall")
    # Change range of color bar
    plt.hlines(0, 1, other_outputs_df.lag.max(), color="black", linestyle="--")
    plt.clim(0, 1)
    plt.ylim(-120, 500)
    plt.semilogx()
    plt.xlabel("Lag ($\\tau$)")
    plt.ylabel("% error")
    plt.legend(loc="upper right")
    # plt.show()


def plot_error_trend_scatter(
    bad_outputs_df, interp_outputs_df, title="Overall % error vs. sparsity"
):
    fig, ax = plt.subplots(figsize=(6, 3), tight_layout=True)
    sfn_mape = bad_outputs_df.groupby("missing_prop_overall")["error_percent"].agg(
        lambda x: np.mean(np.abs(x))
    )

    sfn_mape_i = interp_outputs_df.groupby("missing_prop_overall")["error_percent"].agg(
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
        x, y, bins=num_bins, range=[[0, inputs.lag.max()], [0, 1]]
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

    data = {"Lag": [], missing_measure: [], "MPE": []}
    # Calculate the mean value in each bin
    xidx = np.digitize(x, xedges) - 1  # correcting for annoying 1-indexing
    yidx = np.digitize(y, yedges) - 1  # as above
    means = np.full((num_bins, num_bins), fill_value=np.nan)
    for i in range(num_bins):
        for j in range(num_bins):
            # If there are any values, calculate the mean for that bin
            if len(x[(xidx == i) & (yidx == j)]) > 0:
                # means[i, j] = np.mean(y[(xidx == i) & (yidx == j)])
                lag_prop_mean = np.nanmedian(
                    inputs["error_percent"][(xidx == i) & (yidx == j)]
                )
                means[i, j] = lag_prop_mean
                data["Lag"].append(xedges[i])
                data[missing_measure].append(yedges[j])
                data["MPE"].append(lag_prop_mean)

    return means, [xedges, yedges], pd.DataFrame(data)


def create_heatmap_lookup_3D(inputs, missing_measure, num_bins=25, log=False):
    """Extract the mean error for each bin of lag and missing measure.
    Args:
        num_bins: The number of bins to use in each direction (x and y)
    """

    x = inputs["lag"]
    y = inputs[missing_measure]
    z = inputs["sosf"]

    xedges = np.linspace(0, inputs.lag.max() + 1, num_bins + 1)  # Lags
    yedges = np.linspace(0, 1, num_bins + 1)  # Missing prop
    zedges = np.linspace(inputs.sosf.min(), inputs.sosf.max(), num_bins + 1)  # Power

    if log is True:
        xedges = (
            np.logspace(0, np.log10(inputs.lag.max()), num_bins + 1) - 0.1
        )  # so that first lag bin starts just before 1
        xedges[-1] = inputs.lag.max() + 1
        zedges = np.logspace(-2, 1, num_bins + 1) # ranges from 0.01 to 10

    data = {"Lag": [], missing_measure: [], "sosf": [], "MPE": []}
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
                    # means[i, j] = np.mean(y[(xidx == i) & (yidx == j)])
                    lag_prop_mean = np.nanmedian(
                        inputs["error_percent"][(xidx == i) & (yidx == j) & (zidx == k)]
                    )
                    means[i, j, k] = lag_prop_mean
                    data["Lag"].append(xedges[i])
                    data[missing_measure].append(yedges[j])
                    data["sosf"].append(zedges[k])
                    data["MPE"].append(lag_prop_mean)

    return means, [xedges, yedges, zedges], pd.DataFrame(data)


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
        cbar = fig.colorbar(im, ax=ax, label="MPE", orientation="vertical")
    # Change range of colorbar
    im.set_clim(-100, 100)

    if log is True:
        ax.semilogx()
        ax.set_xlim(1, 250)

    ax.set_facecolor("black")  # So we can distinguish 0 means from missing means
    ax.set_ylabel(missing_measure)

    if subplot is None:
        ax.set_xlabel("Lag")
        ax.set_title("SF estimation error using LINT")
        # plt.show()
    else:
        ax.set_title(title)
        if overlay_x is not None:
            ax.plot(overlay_x, overlay_y)
        return ax


def compute_scaling(bad_output, var, heatmap_vals):
    """
    Extracting values from each bin to create a look-up table. Note that due to binning we have to find the nearest value to get the corresponding MAPE for a given lag and proportion of pairs remaining. Using MPE as we want to maintaing the direction of the error for compensating."""
    df = heatmap_vals
    for i, row in bad_output.iterrows():
        desired_prop = row[var]
        desired_lag = i
        # Calculate absolute differences between desired Lag and all Lag values in the DataFrame
        df["lag_diff"] = np.abs(df["Lag"] - desired_lag)
        df["prop_diff"] = np.abs(df[var] - desired_prop)

        # Find the row with the minimum absolute difference
        nearest_row = df.loc[
            (df["lag_diff"] == np.min(df["lag_diff"]))
            & (df["prop_diff"] == np.min(df["prop_diff"]))
        ]

        # print("Desired lag: {}".format(desired_lag))
        # print("Desired prop: {}".format(desired_prop))
        # print("Nearest row:")
        # print(nearest_row)

        if len(nearest_row) > 1:
            print("More than one nearest bin found!")
            result = nearest_row.head(1)
            MPE = result["MPE"].values[0]
            scaling = 1 / (1 + MPE / 100)

        elif len(nearest_row) == 0:
            print("No nearest bin found for lag {}! Scaling set to 1".format(i))
            scaling = 1
        else:
            result = nearest_row.head(1)
            MPE = result["MPE"].values[0]
            scaling = 1 / (1 + MPE / 100)

        bad_output.loc[i, "scaling"] = scaling
        bad_output.loc[bad_output["error"] == 0, "scaling"] = 1  # Catching 0 errors

    bad_output["sosf_corrected"] = bad_output["sosf"] * bad_output["scaling"]

    # Smoothing potentially jumpy correction
    bad_output["scaling_smoothed"] = (
        bad_output["scaling"].rolling(window=20, min_periods=1).mean()
    )
    bad_output["sosf_corrected_smoothed"] = (
        bad_output["sosf"] * bad_output["scaling_smoothed"]
    )

    return bad_output


def compute_scaling_3d(bad_output, var, heatmap_vals, smoothing_method="linear"):
    """
    Extracting values from each bin to create a look-up table. Note that due to binning we have to find the nearest value to get the corresponding MAPE for a given lag and proportion of pairs remaining. Using MPE as we want to maintaing the direction of the error for compensating."""
    df = heatmap_vals
    for i, row in bad_output.iterrows():
        desired_prop = row[var]
        desired_lag = i
        desired_sosf = row["sosf"]
        # Calculate absolute differences between desired Lag and all Lag values in the DataFrame
        df["lag_diff"] = np.abs(df["Lag"] - desired_lag)
        df["prop_diff"] = np.abs(df[var] - desired_prop)
        df["sosf_diff"] = np.abs(df["sosf"] - desired_sosf)

        # Find the row with the minimum absolute difference
        nearest_row = df.loc[
            (df["lag_diff"] == np.min(df["lag_diff"]))
            & (df["prop_diff"] == np.min(df["prop_diff"]))
            & (df["sosf_diff"] == np.min(df["sosf_diff"]))
        ]

        # print("Desired lag: {}".format(desired_lag))
        # print("Desired prop: {}".format(desired_prop))
        # print("Nearest row:")
        # print(nearest_row)

        if len(nearest_row) > 1:
            print("More than one nearest bin found!")
            result = nearest_row.head(1)
            MPE = result["MPE"].values[0]
            scaling = 1 / (1 + MPE / 100)

        elif len(nearest_row) == 0:
            print("No nearest bin found for lag {}! Scaling set to 1".format(i))
            scaling = 1
        else:
            result = nearest_row.head(1)
            MPE = result["MPE"].values[0]
            scaling = 1 / (1 + MPE / 100)

        bad_output.loc[i, "scaling"] = scaling
        bad_output.loc[bad_output["error"] == 0, "scaling"] = 1  # Catching 0 errors

    bad_output["sosf_corrected_3d"] = bad_output["sosf"] * bad_output["scaling"]
    # Smoothing potentially jumpy correction
    bad_output["scaling_smoothed"] = (
        bad_output["scaling"].rolling(window=20, min_periods=1).mean()
    )
    bad_output["sosf_corrected_smoothed"] = (
        bad_output["sosf"] * bad_output["scaling_smoothed"]
    )

    return bad_output
