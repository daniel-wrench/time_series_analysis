# # Master stats doc
# Here I show the calculation of all the stats for my time series estimation work. These will be put into a function that returns all of them, which we will then calculate the error of for missing data. Note that streamlit code allows calculation for missing data
#
# - Compare utils script to reynolds and update accordingly if needed
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
# mag_int_hr.to_pickle("data/processed/wind/mfi/20160101.pkl")

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
# proton_int.to_pickle("data/processed/wind/3dp/20160101.pkl")

# Define analysis functions


def structure(data, ar_lags, ar_powers):
    """
    Routine to compute the Structure coefficients of a certain series or number of series to different orders
    of structure at lags given in ar_lags
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
            for l in ar_lags:
                dax = np.abs(ax.shift(-l) - ax)
                strct = dax.pow(i).mean()
                array += [strct]
            df[str(i)] = array

    elif data.shape[1] == 3:
        ax = data.iloc[:, 0].copy()
        ay = data.iloc[:, 1].copy()
        az = data.iloc[:, 2].copy()

        for i in ar_powers:
            array = []
            for l in ar_lags:
                dax = np.abs(ax.shift(-l) - ax)
                day = np.abs(ay.shift(-l) - ay)
                daz = np.abs(az.shift(-l) - az)
                strct = (dax.pow(2) + day.pow(2) + daz.pow(2)).pow(0.5).pow(i).mean()
                array += [strct]
            df[str(i)] = array

    df = pd.DataFrame(df)
    return df


def calc_scales_stats(time_series, var_name, params_dict):
    """
    Calculate a rangle of scale-domain statistics for a given time series
    Low-res time series is currently used for ACF and SF,
    high-res for spectrum and taylor scale


    :param time_series: list of 1 (scalar) or 3 (vector) pd.Series
    :param var_name: str
    :param params_dict: dict
    :return: dict
    """
    # Compute autocorrelations and power spectra

    time_series_low_res = [x.resample(params_dict["dt_lr"]).mean() for x in time_series]

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

    time_lags_hr, acf_hr = utils.compute_nd_acf(
        time_series=time_series, nlags=params_dict["nlags_hr"]
    )

    slope_k = np.nan
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

    if params_dict["tau_min"] is not None:
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

    int_lr_df = pd.concat(time_series_low_res, axis=1)
    sfns = structure(int_lr_df, np.arange(1, round(0.2 * len(int_lr_df))), [1, 2, 3, 4])

    # Calculate kurtosis (currently not component-wise)
    sdk = sfns[["2", "4"]].copy()
    sdk.columns = ["2", "4"]
    sdk["2^2"] = sdk["2"] ** 2
    kurtosis = sdk["4"].div(sdk["2^2"])

    # Store these results in a dictionary
    stats_dict = {
        var_name: {
            "time_series": time_series_low_res,
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
            "break_s": break_s,
            "corr_scale_exp_trick": corr_scale_exp_trick,
            "corr_scale_exp_fit": corr_scale_exp_fit,
            "corr_scale_int": corr_scale_int,
            "taylor_scale_u": taylor_scale_u,
            "taylor_scale_u_std": taylor_scale_u_std,
            "taylor_scale_c": taylor_scale_c,
            "taylor_scale_c_std": taylor_scale_c_std,
            "f_periodogram": f_periodogram,
            "power_periodogram": power_periodogram,
            "p_smooth": p_smooth,
            "sfn": sfns,  # multiple orders
            "sdk": kurtosis,
        }
    }

    return int_lr_df, stats_dict


# TESTING OUT THE FUNCTION

timestamp = "20160101"

mag_int_hr = pd.read_pickle("data/processed/wind/mfi/" + timestamp + ".pkl")

# Frequency bounds are taken from Wang et al. (2018, JGR)
mag_params = {
    "f_min_inertial": 0.005,
    "f_max_inertial": 0.2,
    "f_min_kinetic": 0.5,
    "f_max_kinetic": 1.4,
    "nlags_lr": 2000,
    "nlags_hr": 100,
    "dt_lr": "5S",
    "tau_min": 10,
    "tau_max": 50,
}

proton_int = pd.read_pickle("data/processed/wind/3dp/" + timestamp + ".pkl")

proton_params = {
    "f_min_inertial": None,
    "f_max_inertial": None,
    "f_min_kinetic": None,
    "f_max_kinetic": None,
    "nlags_lr": 2000,
    "nlags_hr": 100,
    "dt_lr": "5S",
    "tau_min": None,
    "tau_max": None,
}


# MAG FIELD: SINGLE COMPONENT
# Get raw time series and turbulent quantities
flr, flt = calc_scales_stats([mag_int_hr.Bx], "Bx", mag_params)

# Save dictionary for later plotting
flr.to_pickle("data/processed/" + "Bx_raw_" + timestamp + ".pkl")
with open("data/processed/" + "Bx_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)


# MAG FIELD: FULL VECTOR
flr, flt = calc_scales_stats(
    [mag_int_hr.Bx, mag_int_hr.By, mag_int_hr.Bz], "B", mag_params
)

flr.to_pickle("data/processed/" + "B_raw_" + timestamp + ".pkl")
with open("data/processed/" + "B_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)


# VELOCITY: FULL VECTOR
flr, flt = calc_scales_stats(
    [proton_int.Vx, proton_int.Vy, proton_int.Vz], "V", proton_params
)

flr.to_pickle("data/processed/" + "V_raw_" + timestamp + ".pkl")
with open("data/processed/" + "V_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)


# PROTON DENSITY: SCALAR
flr, flt = calc_scales_stats([proton_int.np], "np", proton_params)

flr.to_pickle("data/processed/" + "np_raw_" + timestamp + ".pkl")
with open("data/processed/" + "np_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)
