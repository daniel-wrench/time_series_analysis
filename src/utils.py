import scipy.signal as signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cdflib
import statsmodels.api as sm
from pprint import pprint
from scipy.optimize import curve_fit
import random

plt.rcParams.update({"font.size": 9})
plt.rc("text", usetex=True)


# Get MSE between two curves
def calc_mse(curve1, curve2):
    mse = np.sum((curve1 - curve2) ** 2) / len(curve1)
    if mse == np.inf:
        mse = np.nan
    return mse


# Get MAPE between two curves
def calc_mape(curve1, curve2):
    curve1 = curve1 + 0.000001  # Have to add this so there is no division by 0
    mape = np.sum(np.abs((curve1 - curve2) / curve1)) / len(curve1)
    if mape == np.inf:
        mape = np.nan
    return mape


def remove_data(array, proportion, chunks=None, sigma=0.1):
    num_obs = proportion * len(array)

    if chunks is None:
        remove_idx = random.sample(range(len(array)), int(num_obs))

    else:
        mean_obs = num_obs / chunks
        std = sigma * 0.341 * 2 * mean_obs
        remove_idx = []

        for i in range(chunks):
            num_obs = round(random.gauss(mu=mean_obs, sigma=std))
            # Comment out the line above and replace num_obs below with mean_obs to revert to equal sized chunks
            if num_obs < 0:
                raise Exception("sigma too high, got negative obs")
            start = random.randrange(
                start=1, stop=len(array) - num_obs
            )  # Starting point for each removal should be far enough from the start and end of the series
            remove = np.arange(start, start + num_obs)

            remove_idx.extend(remove)

        prop_missing = len(np.unique(remove_idx)) / len(array)

        while prop_missing < proportion:
            start = random.randrange(
                start=1, stop=len(array) - num_obs
            )  # Starting point for each removal should be far enough from the start and end of the series
            remove = np.arange(start, start + num_obs)
            remove_idx.extend(remove)

            prop_missing = len(np.unique(remove_idx)) / len(array)

    remove_idx = [int(x) for x in remove_idx]  # Converting decimals to integers
    array_bad = array.copy()
    array_bad[remove_idx] = np.nan

    # Will be somewhat different from value specified if removed in chunks
    prop_removed = np.sum(pd.isna(array_bad)) / len(array)
    idx = np.arange(len(array))
    array_bad_idx = np.delete(idx, remove_idx)

    return array_bad, array_bad_idx, prop_removed


def read_cdf(cdf_file_path: str) -> cdflib.cdfread.CDF:
    """
    Read a .cdf file as a cdf file object
    args:
      cdf_file_path: path to cdf
    """
    try:
        cdf_file_object = cdflib.CDF(cdf_file_path)
    except Exception:
        raise Exception("Exception while reading file")
    return cdf_file_object


def convert_cdf_to_dataframe(
    cdf_file_object: cdflib.cdfread.CDF, varlist=None
) -> pd.DataFrame:
    """
    Convert a cdf file object to a Pandas DataFrame.
    args:
      varlist: list of strings. Specify the variables to include in the resulting DataFrame as they appear in the .cdf file.
               Multi-dimensional attributes are split into multiple columns with the names attribute_x
    """
    if varlist is None:
        varlist = (
            cdf_file_object.cdf_info()["rVariables"]
            + cdf_file_object.cdf_info()["zVariables"]
        )
    variables_to_read = varlist.copy()
    for var_name in varlist:
        if (
            var_name
            not in cdf_file_object.cdf_info()["rVariables"]
            + cdf_file_object.cdf_info()["zVariables"]
        ):
            print(f'variable name "{var_name}" not in cdf file; skipping it')
            variables_to_read.remove(var_name)
    result_dict = {}
    for var_name in variables_to_read:
        variable_values = cdf_file_object.varget(var_name)
        if len(variable_values.shape) == 1:  # single attribute
            result_dict.update({var_name: variable_values})
        else:  # multi-dimensional attribute
            expanded_dict = {
                f"{var_name}_{dim_index}": variable_values[:, dim_index]
                for dim_index in range(variable_values.shape[1])
            }
            result_dict.update(expanded_dict)
    return pd.DataFrame(result_dict)


def replace_fill_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Replace dataframe fill values with numpy NaN. Fill values are documented in https://omniweb.gsfc.nasa.gov/html/omni_min_data.html.
    args:
      dataframe: pandas dataframe to remove fill values from
    """
    df_cleaned = dataframe.replace(
        {
            9.9: np.nan,
            999: np.nan,
            999.99: np.nan,
            999999: np.nan,
            99.99: np.nan,
            9999.99: np.nan,
            9999999.0: np.nan,
            99999.99: np.nan,
            99999.9: np.nan,
            99999: np.nan,
        },
        inplace=False,
    )
    return df_cleaned


def format_epochs(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Epoch attribute into a DateTime object. Epoch column must exist.
    args:
      dataframe: pandas dataframe to format
    """
    assert type(dataframe) == pd.core.frame.DataFrame, "Input is not of type DataFrame"
    assert (
        "Epoch" in dataframe.columns or "EPOCH" in dataframe.columns
    ), "Epoch column does not exist"
    result_dataframe = dataframe.copy()
    result_dataframe["Epoch"] = result_dataframe["Epoch"].apply(
        lambda x: cdflib.epochs.CDFepoch.to_datetime(x, to_np=True)[0]
    )
    result_dataframe.rename({"Epoch": "Timestamp"}, axis="columns", inplace=True)
    return result_dataframe


def resample_time_series(
    dataframe: pd.DataFrame, cadence: str, agg_types: dict = None
) -> pd.DataFrame:
    """
    Resample time series data to a regular cadence in order to merge data from different sources.
    args:
      dataframe: time series dataframe slice
      cadence: time cadence to resample on
      agg_types: aggregation types specified for all variables, recommended stay as mean
    """
    if not cadence:
        return dataframe.set_index("Timestamp")
    assert type(dataframe) == pd.core.frame.DataFrame, "Input is not of type DataFrame"
    assert "Timestamp" in dataframe.columns, "Timestamp column does not exist"
    if not agg_types:
        return dataframe.resample(cadence, on="Timestamp").mean()
    return dataframe.resample(cadence, on="Timestamp").agg(agg_types)


def join_dataframes_on_timestamp(
    dataframe_1: pd.DataFrame, dataframe_2: pd.DataFrame
) -> pd.DataFrame:
    """
    Join dataframes on the Timestamp attribute. Dataframes should be (re)sampled on the same cadence.
    args:
      dataframe_1 and dataframe_2: dataframes to join, must contain a synched, formatted datetime column
    """
    assert (
        type(dataframe_1.index) == pd.core.indexes.datetimes.DatetimeIndex
    ), "dataframe_1 does not have a Datetime index"
    assert (
        type(dataframe_2.index) == pd.core.indexes.datetimes.DatetimeIndex
    ), "dataframe_2 does not have a Datetime index"
    joined_df = pd.merge(dataframe_1, dataframe_2, how="outer", on="Timestamp")
    return joined_df


def mask_outliers(dataframe: pd.DataFrame, threshold_dict: dict) -> pd.DataFrame:
    """
    Mask values outside of the specified thresholds with numpy NaNs. Interpolate NaNs
    inputs:
      threshold_dict: dict. Pairs of column names and thresholds to identify outliers.
    """
    if not threshold_dict:
        return dataframe
        # Selected only non-time columns due to issues with pd.DataFrame.interpolate not handling certain time columns
    result_dataframe = dataframe.copy()
    for column_name, thresholds in threshold_dict.items():
        result_dataframe[column_name] = result_dataframe[column_name].mask(
            (result_dataframe[column_name] <= thresholds[0])
            | (result_dataframe[column_name] >= thresholds[-1])
        )
    return result_dataframe


def pipeline(file_path, varlist, cadence=None, thresholds={}):
    """
    Read a CDF file and return a cleaned dataframe

    Args:
        file_path: path to CDF file
        varlist: list of variables to include in the dataframe
        cadence: cadence to resample the dataframe to
        thresholds: dictionary of thresholds to mask outliers

    Returns:
        dataframe: cleaned dataframe
    """
    try:
        df = convert_cdf_to_dataframe(read_cdf(file_path), varlist=varlist)
        df = replace_fill_values(df)
        df = format_epochs(df)
        df = mask_outliers(df, thresholds)
        if cadence not in [0, "0S", None]:
            df = resample_time_series(df, cadence=cadence)
        # df.set_index('Timestamp', inplace=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def print_cdf_info(file_path: str) -> None:
    """
    Print the cdf info of the specified file, for debugging purposes
    """
    pprint(read_cdf(file_path).cdf_info())


def SmoothySpec(a, nums=None):
    """Smooth a curve using a moving average smoothing"""
    b = a.copy()
    if nums is None:
        nums = 2 * len(b) // 3
    for i in range(nums):
        b[i + 1 : -1] = 0.25 * b[i:-2] + 0.5 * b[i + 1 : -1] + 0.25 * b[i + 2 :]
    return b


def fitpowerlaw(ax, ay, xi, xf):
    idxi = np.argmin(abs(ax - xi))
    idxf = np.argmin(abs(ax - xf))
    xx = np.linspace(xi, xf, 100)
    z = np.polyfit(np.log(ax[idxi:idxf]), np.log(ay[idxi:idxf]), 1)
    p = np.poly1d(z)
    pwrl = np.exp(p(np.log(xx)))
    return z, xx, pwrl


def compute_spectral_stats(
    time_series,
    f_min_inertial=None,
    f_max_inertial=None,
    f_min_kinetic=None,
    f_max_kinetic=None,
    timestamp=None,
    di=None,
    velocity=None,
    plot=False,
):
    """Computes the power spectrum for a scalar or vector time series.
    Also computes the power-law fit in the inertial and kinetic ranges,
    and the spectral break between the two ranges, if specified.

    ### Args:

    - time_series: list of 1 (scalar) or 3 (vector) pd.Series. The function automatically detects
    the cadence if timestamped index, otherwise dt = 1s
    - f_min_inertial: (Optional) Minimum frequency for the power-law fit in the inertial range
    - f_max_inertial: (Optional) Maximum frequency for the power-law fit in the inertial range
    - f_min_kinetic: (Optional) Minimum frequency for the power-law fit in the kinetic range
    - f_max_kinetic: (Optional) Maximum frequency for the power-law fit in the kinetic range
    - timestamp: (Optional, only used for plotting) Timestamp of the data
    - di: (Optional, only used for plotting) Ion inertial length in km
    - velocity: (Optional, only used for plotting) Solar wind velocity in km/s
    - plot: (Optional) Whether to plot the PSD

    ### Returns:

    - z_i: Slope in the inertial range
    - z_k: Slope in the kinetic range
    - spectral_break: Frequency of the spectral break between the two ranges
    - f_periodogram: Frequency array of the periodogram
    - power_periodogram: Power array of the periodogram
    - p_smooth: Smoothed power array of the periodogram
    - xi: Frequency array of the power-law fit in the inertial range
    - xk: Frequency array of the power-law fit in the kinetic range
    - pi: Power array of the power-law fit in the inertial range
    - pk: Power array of the power-law fit in the kinetic range


    """

    # Check if the data has a timestamp index
    if isinstance(time_series[0].index, pd.DatetimeIndex):
        # Get the cadence of the data
        dt = time_series[0].index[1] - time_series[0].index[0]
        dt = dt.total_seconds()
    else:
        # If not, assume 1 second cadence
        dt = 1

    x_freq = 1 / dt

    # Convert the time series into a numpy array
    np_array = np.array(time_series)

    if np_array.shape[0] == 3:  # If the input is a vector
        f_periodogram, power_periodogram_0 = signal.periodogram(
            np_array[0], fs=x_freq, window="boxcar", scaling="density"
        )
        power_periodogram_0 = (x_freq / 2) * power_periodogram_0

        f_periodogram, power_periodogram_1 = signal.periodogram(
            np_array[1], fs=x_freq, window="boxcar", scaling="density"
        )
        power_periodogram_1 = (x_freq / 2) * power_periodogram_1

        f_periodogram, power_periodogram_2 = signal.periodogram(
            np_array[2], fs=x_freq, window="boxcar", scaling="density"
        )
        power_periodogram_2 = (x_freq / 2) * power_periodogram_2

        power_periodogram = (
            power_periodogram_0 + power_periodogram_1 + power_periodogram_2
        ) / 3

    elif np_array.shape[0] == 1:  # If the input is a scalar
        f_periodogram, power_periodogram = signal.periodogram(
            np_array[0], fs=x_freq, window="boxcar", scaling="density"
        )
        power_periodogram = (x_freq / 2) * power_periodogram

    # Slowest part of this function - takes ~ 10 seconds
    p_smooth = SmoothySpec(power_periodogram)

    # If the user has specified a range for the power-law fits
    if f_min_inertial is not None:
        qk, xk, pk = fitpowerlaw(
            f_periodogram, p_smooth, f_min_kinetic, f_max_kinetic
        )  # Kinetic range
        qi, xi, pi = fitpowerlaw(
            f_periodogram, p_smooth, f_min_inertial, f_max_inertial
        )  # Inertial range

        try:
            powerlaw_intersection = np.roots(qk - qi)
            spectral_break = np.exp(powerlaw_intersection)
        except Exception as e:
            print("could not compute power-law intersection: {}".format(e))
            spectral_break = [np.nan]

        if round(spectral_break[0], 4) == 0 or spectral_break[0] > 1:
            spectral_break = [np.nan]

    else:
        qi = [np.nan]
        qk = [np.nan]
        spectral_break = [np.nan]
        xi = [np.nan]
        xk = [np.nan]
        pi = [np.nan]
        pk = [np.nan]

    if plot is True:
        fig, ax = plt.subplots(figsize=(3.3, 2), constrained_layout=True)
        ax.set_ylim(1e-6, 1e6)

        ax.semilogy(
            f_periodogram,
            power_periodogram,
            label="Raw periodogram",
            color="black",
            alpha=0.2,
        )
        ax.semilogy(
            f_periodogram, p_smooth, label="Smoothed periodogram", color="black"
        )

        # If the power-law fits have succeeded, plot them
        if not np.isnan(qi[0]):
            ax.semilogy(
                xi,
                pi * 3,
                c="black",
                ls="--",
                lw=0.8,
                label="Inertial range power-law fit: $\\alpha_i$ = {0:.2f}".format(
                    qi[0]
                ),
            )
            ax.semilogy(
                xk,
                pk * 3,
                c="black",
                ls="--",
                lw=0.8,
                label="Kinetic range power-law fit: $\\alpha_k$ = {0:.2f}".format(
                    qk[0]
                ),
            )

        ax.tick_params(which="both", direction="in")
        ax.semilogx()

        if spectral_break[0] is not np.nan:
            ax.axvline(
                np.exp(np.roots(qk - qi)),
                alpha=0.6,
                color="black",
                label="Spectral break: $f_d={0:.2f}$".format(spectral_break[0]),
            )

        # Adding in proton inertial frequency
        if di is not None and velocity is not None:
            f_di = velocity / (2 * np.pi * di)
            ax.axvline(
                f_di,
                color="black",
                alpha=0.6,
                label="Proton inertial frequency: $f_{di}=$" + "{0:.2f}".format(f_di),
            )
            ax.text(f_di * 1.2, 1e-5, "$f_{{di}}$")

        # bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5)
        ax.text(xi[0] * 5, pi[0], "$f^{q_i}$")
        ax.text(xk[0] * 2, pk[0], "$f^{q_k}$")
        ax.text(spectral_break[0] / 2, 1e-5, "$f_b$")

        if timestamp is not None:
            # Add box with timestamp and values of qi and qk
            textstr = "\n".join(
                (
                    str(timestamp[:-3])
                    + "-"
                    + "23:59",  # NOTE - this is a hacky way to get the end timestamp
                    r"$q_i=%.2f$" % (qi[0],),
                    r"$q_k=%.2f$" % (qk[0],),
                    r"$f_b=%.2f$" % (spectral_break[0],),
                    r"$f_{{di}}=%.2f$" % (f_di,),
                )
            )
            props = dict(boxstyle="round", facecolor="gray", alpha=0.2)
            # Place the text box. (x, y) position is in axis coordinates.
            ax.text(
                0.05,
                0.1,
                textstr,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="bottom",
                bbox=props,
            )

        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("PSD (nT$^2$Hz$^{-1}$)")
        # plt.grid()
        # plt.show()

        return qi[0], qk[0], spectral_break[0], f_periodogram, p_smooth, fig, ax
    else:
        return (
            qi[0],
            qk[0],
            spectral_break[0],
            f_periodogram,
            power_periodogram,
            p_smooth,
            xi,
            xk,
            pi,
            pk,
        )


def compute_nd_acf(time_series, nlags, plot=False):
    """Compute the autocorrelation function for a scalar or vector time series.

    Args:

    - time_series: list of 1 (scalar) or 3 (vector) pd.Series. The function automatically detects
    the cadence if timestamped index, otherwise dt = 1s.
    - nlags: The number of lags to calculate the ACF up to
    - plot: Whether to plot the ACF

    Returns:

    - time_lags: The x-values of the ACF, in seconds, given the cadence of measurements
    - R: The values of the ACF from lag 0 to nlags

    """

    # Convert the time series into a numpy array
    np_array = np.array(time_series)

    if np_array.shape[0] == 3:
        acf = (
            # missing="conservative" ignores NaNs when computing the ACF
            sm.tsa.acf(np_array[0], fft=True, nlags=nlags, missing="conservative")
            + sm.tsa.acf(np_array[1], fft=True, nlags=nlags, missing="conservative")
            + sm.tsa.acf(np_array[2], fft=True, nlags=nlags, missing="conservative")
        )
        acf /= 3

    elif np_array.shape[0] == 1:
        acf = sm.tsa.acf(np_array[0], fft=True, nlags=nlags)

    else:
        raise ValueError(
            "Array is not 3D or 1D. If after a 1D acf, try putting square brackets around the pandas series in np.array()"
        )

    # Check if the data has a timestamp index
    if isinstance(time_series[0].index, pd.DatetimeIndex):
        # Get the cadence of the data
        dt = time_series[0].index[1] - time_series[0].index[0]
        dt = dt.total_seconds()
    else:
        # If not, assume 1 second cadence
        dt = 1

    time_lags = np.arange(0, nlags + 1) * dt

    # Optional plotting
    if plot is True:
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(time_lags, acf)
        ax.set_xlabel("$\\tau$ (sec)")
        ax.set_ylabel("Autocorrelation")

        # For plotting secondary axes
        def sec2lag(x):
            return x / dt

        def lag2sec(x):
            return x * dt

        secax_x = ax.secondary_xaxis("top", functions=(sec2lag, lag2sec))
        secax_x.set_xlabel("$\\tau$ (lag)")

        def sec2km(x):
            return x * 400

        def km2sec(x):
            return x / 400

        # use of a float for the position:
        secax_x2 = ax.secondary_xaxis(-0.2, functions=(sec2km, km2sec))
        secax_x2.set_xlabel("$r$ (km)")

        plt.show()

    return time_lags, acf


def compute_outer_scale_exp_trick(
    autocorrelation_x: np.ndarray, autocorrelation_y: np.ndarray, plot=False
):
    """
    computes the correlation scale through the "1/e" estimation method.
    autocorrelation_x assumed already in time scale
    """
    for i, j in zip(autocorrelation_y, autocorrelation_x):
        if i <= np.exp(-1):
            # print(i, j)
            idx_2 = np.where(autocorrelation_x == j)[0]
            idx_1 = idx_2 - 1
            x2 = autocorrelation_x[idx_2]
            x1 = autocorrelation_x[idx_1]
            y1 = autocorrelation_y[idx_1]
            y2 = autocorrelation_y[idx_2]
            x_opt = x1 + ((y1 - np.exp(-1)) / (y1 - y2)) * (x2 - x1)
            # print(autocorrelation_x[idx_1], autocorrelation_y[idx_1])
            # print(autocorrelation_x[idx_2], autocorrelation_y[idx_2])
            # print('e:', np.exp(-1))
            # print(x_opt)

            try:
                # Optional plotting, set up to eventually display all 3 corr scale methods
                if plot is True:
                    fig, ax = plt.subplots(
                        1, 1, figsize=(3.3, 2.5), constrained_layout=True
                    )
                    # fig.subplots_adjust(left=0.2, top=0.8, bottom=0.8)

                    ax.plot(
                        autocorrelation_x / 1000,
                        autocorrelation_y,
                        c="black",
                        label="Autocorrelation",
                        lw=0.5,
                    )
                    ax.set_xlabel("$\\tau (10^3$ s)")
                    ax.set_ylabel("$R(\\tau)$")

                    def sec2km(x):
                        return x * 1000 * 400 / 1e6

                    def km2sec(x):
                        return x / 1000 / 400 * 1e6

                    # use of a float for the position:
                    secax_x2 = ax.secondary_xaxis("top", functions=(sec2km, km2sec))
                    secax_x2.set_xlabel("$r$ ($10^6$ km)")
                    secax_x2.tick_params(which="both", direction="in")
                    ax.axhline(
                        np.exp(-1),
                        color="black",
                        ls="--",
                        label="$1/e\\rightarrow\\lambda_C^{{1/e}}$={:.0f}s".format(
                            x_opt[0]
                        ),
                    )
                    ax.axvline(x_opt[0] / 1000, color="black", ls="--")
                    ax.tick_params(which="both", direction="in")
                    # label="$1/e\\rightarrow \lambda_C^{1/e}=${:.0f}s".format(x_opt[0]))
                    return round(x_opt[0], 3), fig, ax
                else:
                    return round(x_opt[0], 3)
            except Exception:
                return 0

    # none found
    return -1


def exp_fit(r, lambda_c):
    """
    fit function for determining correlation scale, through the optimal lambda_c value
    """
    return np.exp(-1 * r / lambda_c)


def para_fit(x, a):
    """
    fit function for determining taylor scale, through the optimal lambda_c value
    """
    return a * x**2 + 1


def compute_outer_scale_exp_fit(
    time_lags, acf, seconds_to_fit, fig=None, ax=None, plot=False
):
    dt = time_lags[1] - time_lags[0]
    num_lags_for_lambda_c_fit = int(seconds_to_fit / dt)
    c_opt, c_cov = curve_fit(
        exp_fit,
        time_lags[:num_lags_for_lambda_c_fit],
        acf[:num_lags_for_lambda_c_fit],
        p0=1000,
    )
    lambda_c = c_opt[0]

    # Optional plotting
    if plot is True:
        if fig is not None and ax is not None:
            fig = fig
            ax = ax

            ax.plot(
                np.array(range(int(seconds_to_fit))) / 1000,
                exp_fit(np.array(range(int(seconds_to_fit))), *c_opt),
                label="Exp. fit$\\rightarrow\\lambda_C^{{\\mathrm{{fit}}}}$={:.0f}s".format(
                    lambda_c
                ),
                lw=3,
                c="black",
            )

        return lambda_c, fig, ax
    else:
        return lambda_c


def compute_outer_scale_integral(time_lags, acf, fig=None, ax=None, plot=False):
    dt = time_lags[1] - time_lags[0]
    idx = np.argmin(np.abs(acf))  # Getting the index where the ACF falls to 0
    integral = np.sum(acf[:idx]) * dt  # Computing integral up to that index

    # Optional plotting
    if plot is True:
        # Optional plotting
        if fig is not None and ax is not None:
            fig = fig
            ax = ax

            ax.fill_between(
                time_lags / 1000,
                0,
                acf,
                where=acf > 0,
                color="black",
                alpha=0.2,
                label="Integral$\\rightarrow\\lambda_C^{{\mathrm{{int}}}}$={:.0f}s".format(
                    integral
                ),
            )
            ax.set_xlabel("$\\tau$ ($10^3$s)")
            ax.tick_params(which="both", direction="in")
            # Plot the legend
            ax.legend(loc="upper right")

        return integral, fig, ax
    else:
        return integral


def compute_taylor_scale(time_lags, acf, tau_fit, plot=False, show_intercept=False):
    """Compute the Taylor microscale

    Args:

    - time_lags: The x-values of the ACF, in seconds, given the cadence of measurements
    - acf: The y-values of the ACF
    - tau_fit: number of lags to fit the parabola over
    """

    # If using seconds_fit as the fitting argument instead:

    dt = time_lags[1] - time_lags[0]
    # tau_fit = int(seconds_fit/dt)

    t_opt, t_cov = curve_fit(
        para_fit, time_lags[:tau_fit], acf[:tau_fit], p0=10
    )  # Initial guess for the parameters
    lambda_t = (-1 * t_opt[0]) ** -0.5

    extended_parabola_x = np.arange(0, 1.2 * lambda_t, 0.1)
    extended_parabola_y = para_fit(extended_parabola_x, *t_opt)

    if plot is True:
        fig, ax = plt.subplots(2, 1, figsize=(3.3, 4), constrained_layout=True)
        # fig.subplots_adjust(hspace=0.1, left=0.2, top=0.8)

        ax[0].scatter(
            time_lags / dt,  # Plotting firstly in lag space for clearer visualisation
            acf,
            label="Autocorrelation",
            s=12,
            c="black",
            alpha=0.5,
        )

        ax[0].plot(
            (extended_parabola_x / dt),
            (extended_parabola_y),
            "-y",
            label="Parabolic fit \nup to $\\tau_\mathrm{fit}\\rightarrow\\tau_\mathrm{TS}^\mathrm{est}$",
            c="black",
        )

        ax[0].axvline(
            tau_fit * (time_lags[1] / dt - time_lags[0] / dt),
            ls="--",
            # label=f"$\\tau_{{fit}}={tau_fit}$ lags",
            c="black",
            alpha=0.6,
        )

        ax[0].set_xlim(-1, 45)
        ax[0].set_ylim(0.986, 1.001)

        if show_intercept is True:
            ax[0].set_ylim(0, 1.05)
            ax[0].set_xlim(-1, 200)  # lambda_t/dt + 5
            ax[0].axvline(lambda_t / dt, ls="dotted", c="black", alpha=0.6)

        ax[0].set_xlabel("$\\tau$ (lags)")
        ax[0].xaxis.set_label_position("top")
        ax[0].set_ylabel("$R(\\tau)$")
        ax[0].tick_params(
            which="both",
            direction="in",
            top=True,
            bottom=False,
            labeltop=True,
            labelbottom=False,
        )

        # For plotting secondary axis, in units of r(km)
        def lag2km(x):
            return x * dt * 400

        def km2lag(x):
            return x / (dt * 400)

        secax_x = ax[0].secondary_xaxis(1.3, functions=(lag2km, km2lag))
        secax_x.set_xlabel("$r$ (km)")
        secax_x.tick_params(which="both", direction="in")

        ax[0].legend(loc="upper right")
        ax[0].annotate("(a)", (2, 0.9875), transform=ax[0].transAxes, size=12)
        ax[0].annotate(
            "$\\tau_\mathrm{fit}$",
            (10, 0.9875),
            transform=ax[0].transAxes,
            size=12,
            alpha=0.6,
        )
        # ax[0].annotate('$\\tau_\mathrm{TS}^\mathrm{est}\\rightarrow=$', (35, 0.9875), transform=ax[0].transAxes, size=10, alpha=0.6)

        return lambda_t, fig, ax

    else:
        return lambda_t


def compute_taylor_chuychai(
    time_lags,
    acf,
    tau_min,
    tau_max,
    fig=None,
    ax=None,
    q=None,
    tau_fit_single=None,
    save=False,
    figname="",
):
    """Compute a refined estimate of the Taylor microscale using a linear extrapolation method from Chuychai et al. (2014).

    Args:

    - time_lags: The x-values of the ACF, in seconds, given the cadence of measurements
    - acf: The y-values of the ACF
    - tau_min: Minimum value for the upper lag to fit the parabola over. This should not be too small, because the data has finite time resolution and there may be limited data available at the shortest time lags. (You will see divergent behaviour if this happens.)
    - tau_max: Maximum value for the upper lag to fit the parabola over
    - q: Slope of the dissipation range
    """

    dt = time_lags[1] - time_lags[0]

    tau_fit = np.arange(tau_min, tau_max + 1)
    tau_ts = np.array([])

    for i in tau_fit:
        lambda_t = compute_taylor_scale(time_lags, acf, tau_fit=i)
        tau_ts = np.append(tau_ts, lambda_t)

    # Performing linear extrapolation back to tau_fit = 0
    z, cov = np.polyfit(x=tau_fit, y=tau_ts, deg=1, cov=True)
    f = np.poly1d(z)

    ts_est_extra = z[1]  # Extracting y-intercept

    # Getting standard deviation of y-intercept
    # (will plot +- 1 standard deviation)
    ts_est_extra_std = np.sqrt(cov[1, 1])

    # Getting extrapolation line for plotting
    other_x = np.arange(0, tau_max + 1)
    other_y = f(other_x)

    # Applying correction factor q from Chuychai et al. (2014)
    if q is not None:
        q_abs = np.abs(q)
        if q_abs < 2:
            r = -0.64 * (1 / q_abs) + 0.72
        elif q_abs >= 2 and q_abs < 4.5:
            r = -2.61 * (1 / q_abs) + 1.7
        elif q_abs >= 4.5:
            r = -0.16 * (1 / q_abs) + 1.16

    else:
        r = 1

    ts_est = r * ts_est_extra
    ts_est_std = r * ts_est_extra_std

    # Optional plotting
    if fig is not None and ax is not None:
        ax[1].scatter(
            tau_fit,
            tau_ts,
            label="Fitted values $\\tau_\mathrm{TS}^\mathrm{est}$",
            s=12,
            c="black",
            alpha=0.5,
            marker="x",
        )

        ax[1].plot(
            other_x,
            other_y,
            label="R.E.$\\rightarrow\\tau_\mathrm{{TS}}^\mathrm{{ext}}$={:.0f}s".format(
                ts_est_extra
            ),
            c="black",
        )

        if tau_fit_single is not None:
            ax[1].axvline(
                tau_fit_single,
                ls="--",
                # ymin=0.5,
                # ymax=1,
                c="black",
                alpha=0.6,
            )

        if q is not None:
            ax[1].plot(
                0,
                ts_est,
                "*",
                color="green",
                label="C.C.$\\rightarrow\\tau_\mathrm{{TS}}$={:.0f}s".format(ts_est),
                markersize=10,
            )

        ax[1].set_xlabel("")
        ax[1].set_xticks([])

        ax[1].set_ylabel("$\\tau$(s)")
        ax[1].tick_params(which="both", direction="in")

        # For plotting secondary axis, units of tau(s)
        def sec2lag(x):
            return x / dt

        def lag2sec(x):
            return x * dt

        secax_x2 = ax[1].secondary_xaxis(0, functions=(lag2sec, sec2lag))

        secax_x2.set_xlabel("$\\tau_\\mathrm{fit}$(s)")
        secax_x2.tick_params(which="both", direction="in")

        # Add legend with specific font size
        ax[1].legend(loc="lower right")
        ax[1].set_xlim(-1, 45)  # Set to 200 if wanting to see extrapolation
        ax[1].set_ylim(-3, max(tau_ts) + 1)
        ax[1].annotate("(b)", (2, 24), size=12)

        return ts_est, ts_est_std, fig, ax

    else:
        return ts_est, ts_est_std
