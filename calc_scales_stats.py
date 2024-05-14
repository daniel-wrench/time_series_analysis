# # Master stats doc
# Here I show the calculation of all the stats for my time series estimation work. These will be put into a function that returns all of them, which we will then calculate the error of for missing data. Note that streamlit code allows calculation for missing data
#
# - Compare utils script to reynolds and update accordingly if needed
# - View fluctuations instead as an optional parameter to the plotting function, plot mean values
# - Run filter beforehand
#

import numpy as np
import pandas as pd
import src.utils as utils
import src.params as params
import glob
import pickle

# Prior to analysis, extract data from CDF files and save as pickle files


# def get_subfolders(path):
#     return sorted(glob.glob(path + "/*"))


# def get_cdf_paths(subfolder):
#     return sorted(glob.iglob(subfolder + "/*.cdf"))


# mfi_file_list = get_cdf_paths("data/raw/wind/mfi/")
# proton_file_list = get_cdf_paths("data/raw/wind/3dp/")


# data_B = utils.pipeline(
#     mfi_file_list[0],
#     varlist=[params.timestamp, params.Bwind, params.Bwind_vec],
#     thresholds=params.mag_thresh,
#     cadence=params.dt_hr,
# )

# data_B = data_B.rename(columns={params.Bx: "Bx", params.By: "By", params.Bz: "Bz"})
# data_B = data_B.interpolate(method="linear")
# mag_int_hr = data_B["2016-01-01 12:00":]

# print(mag_int_hr.head())
# mag_int_hr.to_pickle("data/processed/wind/wind/mfi/20160101.pkl")

# data_protons = utils.pipeline(
#     proton_file_list[0],
#     varlist=[
#         params.timestamp,
#         params.np,
#         params.nalpha,
#         params.Tp,
#         params.Talpha,
#         params.V_vec,
#     ],
#     thresholds=params.proton_thresh,
#     cadence=params.dt_protons,
# )

# data_protons = data_protons.rename(
#     columns={
#         params.Vx: "Vx",
#         params.Vy: "Vy",
#         params.Vz: "Vz",
#         params.np: "np",
#         params.nalpha: "nalpha",
#         params.Tp: "Tp",
#         params.Talpha: "Talpha",
#     }
# )

# data_protons = data_protons.interpolate(method="linear")
# proton_int = data_protons["2016-01-01 12:00":]

# print(proton_int.head())
# proton_int.to_pickle("data/processed/wind/wind/3dp/20160101.pkl")

# Define analysis functions


def structure(data, ar_lags, ar_powers):
    """
    Routine to compute the Structure coefficients of a certain series or number of series to different orders
    of structure at lags given in ar_lags. (This is an adapted version of Tulasi's function from TurbAn\Analysis\TimeSeries\DataAnalysisRoutines.py,
    now working with scalars as well as vectors)
    Input:
            data: pd.DataFrame of data to be analysed. Must have shape (1, N) or (3, N)
            ar_lags: The array consisting of lags, being the number of points to shift each of the series
            ar_powers: The array consisting of the Structure orders to perform on the series for all lags
    Output:
            df: The DataFrame containing the structure coefficients corresponding to ar_lags for each order in ar_powers
    """
    # run through ar_lags and ar_powers so that for each power, run through each lag
    df = {}

    if data.shape[1] == 1:
        ax = data.iloc[:, 0].copy()
        for i in ar_powers:
            array = []
            for lag in ar_lags:
                lag = int(lag)
                dax = np.abs(ax.shift(-lag) - ax)
                strct = dax.pow(i).mean()
                array += [strct]

            df[str(i)] = array

    elif data.shape[1] == 3:
        ax = data.iloc[:, 0].copy()
        ay = data.iloc[:, 1].copy()
        az = data.iloc[:, 2].copy()

        for i in ar_powers:
            array = []
            for lag in ar_lags:
                lag = int(lag)
                dax = np.abs(ax.shift(-lag) - ax)
                day = np.abs(ay.shift(-lag) - ay)
                daz = np.abs(az.shift(-lag) - az)
                strct = (dax.pow(2) + day.pow(2) + daz.pow(2)).pow(0.5).pow(i).mean()
                array += [strct]
            df[str(i)] = array

    df = pd.DataFrame(df, index=ar_lags)
    return df


# Alternative sfn method
# Currently only works for scalar input

# def calc_sfn(data, p, freq=1, max_lag_prop=0.2):
#     # Calculate lags
#     lag_function = {}
#     for i in np.arange(
#         1, round(max_lag_prop * len(data))
#     ):  # Limiting maximum lag to 20% of dataset length
#         lag_function[i] = data.diff(i)

#     # Initialise dataframe
#     structure_functions = pd.DataFrame(index=np.arange(1, len(data)))

#     # Converting lag values from points to seconds
#     structure_functions["lag"] = structure_functions.index / freq

#     for order in p:
#         lag_dataframe = (
#             pd.DataFrame(lag_function) ** order
#         )  # or put in .abs() before order: this only changes the odd-ordered functions
#         structure_functions[str(order)] = pd.DataFrame(lag_dataframe.mean())

#     return structure_functions.dropna()


def calc_scales_stats(time_series, var_name, params_dict):
    """
    Calculate a rangle of scale-domain statistics for a given time series
    Low-res time series is currently used for ACF and SF,
    original for spectrum and taylor scale

    :param time_series: list of 1 (scalar) or 3 (vector) pd.Series
    :param var_name: str
    :param params_dict: dict of the following parameters:
        spectrum: bool, if wanting to calculate spectral stats
        f_min_inertial: float, optional, for above
        f_max_inertial: float, optional
        f_min_kinetic: float, optional
        f_max_kinetic: float, optional
        nlags_lr: int, # lags for low-res ACF
        nlags_hr: int, optional, only if calculating Taylor scale
        dt_lr: str, optional, only if wanting to reduce comp. burden for ACF and SF calculations
        tau_min: float, optional, only if calculating Taylor scale
        tau_max: float, optional, only if calculating Taylor scale

    :return: pd.DataFrame, dict
    """

    if len(time_series) == 3:
        # Compute magnitude of vector time series
        x_mean = time_series[0].mean()
        y_mean = time_series[1].mean()
        z_mean = time_series[2].mean()
        magnitude = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)

        # Calculate rms fluctuations
        x_fluct = time_series[0] - x_mean
        y_fluct = time_series[1] - y_mean
        z_fluct = time_series[2] - z_mean
        rms_fluct = np.sqrt(np.mean(x_fluct**2 + y_fluct**2 + z_fluct**2))

    # Compute magnitude of scalar time series
    elif len(time_series) == 1:
        magnitude = time_series[0].mean()
        fluct = time_series[0] - magnitude
        rms_fluct = np.sqrt(np.mean(fluct**2))

    else:
        raise ValueError("Time series must be of length 1 or 3")

    # Compute autocorrelations and power spectra
    if params_dict["dt_lr"] is not None:
        time_series_low_res = [
            x.resample(params_dict["dt_lr"]).mean() for x in time_series
        ]
    else:
        time_series_low_res = time_series

    # MATCHES UP WITH LINE 361 IN SRC/PROCESS_DATA.PY
    time_lags_lr, acf_lr = utils.compute_nd_acf(
        time_series=time_series_low_res, nlags=params_dict["nlags_lr"]
    )  # Removing "S" from end of dt string

    corr_scale_exp_trick = utils.compute_outer_scale_exp_trick(time_lags_lr, acf_lr)

    # Use estimate from 1/e method to select fit amount
    corr_scale_exp_fit = utils.compute_outer_scale_exp_fit(
        time_lags_lr, acf_lr, np.round(2 * corr_scale_exp_trick)
    )

    corr_scale_int = utils.compute_outer_scale_integral(time_lags_lr, acf_lr)

    slope_k = np.nan

    if params_dict["spectrum"] is True:
        # ~1min per interval due to spectrum smoothing algorithm
        try:
            (
                slope_i,
                slope_k,
                break_s,
                f_periodogram,
                power_periodogram,
                p_smooth,
                xi,
                xk,
                pi,
                pk,
            ) = utils.compute_spectral_stats(
                time_series=time_series,
                f_min_inertial=params_dict["f_min_inertial"],
                f_max_inertial=params_dict["f_max_inertial"],
                f_min_kinetic=params_dict["f_min_kinetic"],
                f_max_kinetic=params_dict["f_max_kinetic"],
            )

        except Exception as e:
            print("Error: spectral stats calculation failed: {}".format(e))
            # print("Interval timestamp: {}".format(int_start))

    else:
        slope_i = np.nan
        slope_k = np.nan
        break_s = np.nan
        f_periodogram = np.nan
        power_periodogram = np.nan
        p_smooth = np.nan
        xi = np.nan
        xk = np.nan
        pi = np.nan
        pk = np.nan

    if params_dict["nlags_hr"] is not None:
        time_lags_hr, acf_hr = utils.compute_nd_acf(
            time_series=time_series, nlags=params_dict["nlags_hr"]
        )

        taylor_scale_u, taylor_scale_u_std = utils.compute_taylor_chuychai(
            time_lags_hr,
            acf_hr,
            tau_min=params_dict["tau_min"],
            tau_max=params_dict["tau_max"],
        )

        if not np.isnan(slope_k):
            taylor_scale_c, taylor_scale_c_std = utils.compute_taylor_chuychai(
                time_lags_hr,
                acf_hr,
                tau_min=params_dict["tau_min"],
                tau_max=params_dict["tau_max"],
                q=slope_k,
            )

        else:
            taylor_scale_c = np.nan
            taylor_scale_c_std = np.nan
    else:
        taylor_scale_u = np.nan
        taylor_scale_u_std = np.nan
        taylor_scale_c = np.nan
        taylor_scale_c_std = np.nan
        time_lags_hr = np.nan
        acf_hr = np.nan

    int_lr_df = pd.concat(time_series_low_res, axis=1)
    sfns = structure(int_lr_df, np.arange(1, round(0.25 * len(int_lr_df))), [2, 4])

    # Calculate kurtosis (currently not component-wise)
    sdk = sfns[["2", "4"]].copy()
    sdk.columns = ["2", "4"]
    sdk["2^2"] = sdk["2"] ** 2
    kurtosis = sdk["4"].div(sdk["2^2"])

    # Store these results in a dictionary
    stats_dict = {
        var_name: {
            "time_series": time_series_low_res,
            "magnitude": magnitude,
            "rms_fluct": rms_fluct,
            "times": time_lags_lr,
            "time_lags_hr": time_lags_hr,
            "xi": xi,
            "xk": xk,
            "pi": pi,
            "pk": pk,
            "cr": acf_lr,
            "acf_hr": acf_hr,
            "qi": slope_i,
            "qk": slope_k,
            "fb": break_s,
            "tce": corr_scale_exp_trick,
            "tcf": corr_scale_exp_fit,
            "tci": corr_scale_int,
            "ttu": taylor_scale_u,
            "ttu_std": taylor_scale_u_std,
            "ttc": taylor_scale_c,
            "ttc_std": taylor_scale_c_std,
            "f_periodogram": f_periodogram,
            "power_periodogram": power_periodogram,
            "p_smooth": p_smooth,
            "sfn": sfns,  # multiple orders
            "sdk": kurtosis,
        }
    }

    return int_lr_df, stats_dict
