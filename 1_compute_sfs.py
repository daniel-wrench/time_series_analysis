# STEP 1. FOR EACH INTERVAL: standardise, duplicate, gap, calculate SFs
# This will be distributed across job arrays on an HPC


# Import dependencies

import pickle
import pandas as pd
import numpy as np
import ts_dashboard_utils as ts
import src.utils as utils  # copied directly from Reynolds project, normalize() added
import src.params as params
import src.sf_funcs as sf
import src.data_import_funcs as dif
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import sys

sns.set_theme(style="whitegrid", font_scale=1.5)
# plt.rcParams.update({"font.size": 16})

# Ensure necessary directories exist
os.makedirs("plots/temp", exist_ok=True)


# For current Wind importing

sys_arg_dict = {
    # arg1
    "mag_path": params.mag_path,
    "proton_path": params.proton_path,
    "electron_path": params.electron_path,
    # arg2
    "mag_vars": [params.timestamp, params.Bwind, params.Bwind_vec],
    "proton_vars": [params.timestamp, params.np, params.Tp],
    "electron_vars": [params.timestamp, params.ne, params.Te],
    # arg3
    "mag_thresh": params.mag_thresh,
    "proton_thresh": params.proton_thresh,
    "electron_thresh": params.electron_thresh,
    # arg4
    "dt_hr": params.dt_hr,
    "int_size": params.int_size,
    # arg5
    "dt_lr": params.dt_lr,
}


# Read in data and split into standardised intervals

# Previously each core read in multiple files at a time. I think it will be better for each core to do one file at a time,
# especially given that each raw file contains sufficient *approximate* correlation lengths for us to then calculate the
# *local* outer scale and create our standardise intervals using that.


times_to_gap = 3
minimum_missing_chunks = 0.7
np.random.seed(123)  # For reproducibility

spacecraft = sys.argv[1]
os.makedirs(f"data/processed/{spacecraft}", exist_ok=True)

raw_file_list = sorted(glob.iglob(f"data/raw/{spacecraft}/" + "/*.cdf"))

# Selecting one file to read in
file_index = int(sys.argv[2])

if spacecraft == "psp":
    # takes < 1s/file, ~50 MB mem usage, 2-3 million rows

    psp_data = dif.read_cdfs(
        [raw_file_list[file_index]],  # LIMIT HERE!
        {"epoch_mag_RTN": (0), "psp_fld_l2_mag_RTN": (0, 3), "label_RTN": (0, 3)},
    )
    psp_data_ready = dif.extract_components(
        psp_data,
        var_name="psp_fld_l2_mag_RTN",
        label_name="label_RTN",
        time_var="epoch_mag_RTN",
        dim=3,
    )
    psp_df = pd.DataFrame(psp_data_ready)
    psp_df["Time"] = pd.to_datetime("2000-01-01 12:00") + pd.to_timedelta(
        psp_df["epoch_mag_RTN"], unit="ns"
    )
    psp_df = psp_df.drop(columns="epoch_mag_RTN").set_index("Time")

    df_raw = psp_df["B_R"].rename("B")  # Giving generic name for spacecraft consistency
    print("\n")
    print(df_raw.info())

    del psp_data, psp_data_ready, psp_df

elif spacecraft == "wind":
    # Takes ~90s/file, 11 MB mem usage, 1 million rows

    df = utils.pipeline(
        raw_file_list[file_index],
        varlist=sys_arg_dict["mag_vars"],
        thresholds=sys_arg_dict["mag_thresh"],
        cadence=sys_arg_dict["dt_hr"],
    )

    print(
        "Reading {0}: {1:.1f}% missing".format(
            raw_file_list[0], df.iloc[:, 0].isna().sum() / len(df) * 100
        )
    )

    # Ensuring observations are in chronological order
    df_wind_hr = df.sort_index()

    # df_wind_hr = pd.read_pickle("data/processed/" + params.mag_path + params.dt_hr + ".pkl")
    df_wind_hr = df_wind_hr.rename(
        columns={
            params.Bwind: "Bwind",
            params.Bx: "Bx",
            params.By: "By",
            params.Bz: "Bz",
        }
    )

    missing = df_wind_hr.iloc[:, 0].isna().sum() / len(df_wind_hr)

    if missing > 0.4:
        # Replacing values in lists with na
        print("Large missing %")
    else:
        int_wind_hr = df_wind_hr.interpolate().ffill().bfill()

    df_raw = int_wind_hr["Bx"].rename(
        "B"
    )  # Giving generic name for spacecraft consistency

    print(df_raw.info())

    del df_wind_hr, int_wind_hr

else:
    raise ValueError("Spacecraft not recognized")


# The following chunk gives some metadata - not necessary for the pipeline

### 0PTIONAL CODE ###

# if df_raw.isnull().sum() == 0:
#     print("No missing data")
# else:
#     print(f"{df_raw.isnull().sum()} missing points")
# print("Length of interval: " + str(df_raw.notnull().sum()))
# print("Duration of interval: " + str(df_raw.index[-1] - df_raw.index[0]))
# x = df_raw.values

# # Frequency of measurements
# print("Duration between some adjacent data points:")
# print(df_raw.index[2] - df_raw.index[1])
# print(df_raw.index[3] - df_raw.index[2])
# print(df_raw.index[4] - df_raw.index[3])

# a = df_raw.index[2] - df_raw.index[1]
# x_freq = 1 / (a.microseconds / 1e6)
# print("\nFrequency is {0:.1f} Hz (2dp)".format(x_freq))

# print("Mean = {}".format(np.mean(x)))
# print("Standard deviation = {}\n".format(np.std(x)))

### 0PTIONAL CODE END ###


if spacecraft == "psp":
    tc_approx = 500  # starting-point correlation time, in seconds
    cadence_approx = 0.1  # time resolution (dt) of the data, in seconds
    nlags = 100000  # number of lags to compute the ACF over

elif spacecraft == "wind":
    tc_approx = 2000  # s
    cadence_approx = 1  # s
    nlags = 30000

tc_n = 10  # Number of actual (computed) correlation times we want in our standardised interval...
interval_length = 10000  # ...across this many points

df = df_raw.resample(str(cadence_approx) + "S").mean()

ints = []
tc_list = []
cadence_list = []

time_lags_lr, r_vec_lr = utils.compute_nd_acf(
    [df],
    nlags=nlags,
    plot=False,
)

tc, fig, ax = utils.compute_outer_scale_exp_trick(time_lags_lr, r_vec_lr, plot=True)
output_file_path = (
    raw_file_list[file_index]
    .replace("data/raw", "plots/temp")
    .replace(".cdf", "_acf.jpg")
)
plt.savefig(output_file_path, bbox_inches="tight")
plt.close()

if tc == -1:
    tc = tc_approx
    new_cadence = tc_n * tc / interval_length
    print(
        f"tc not found for this interval, setting to 500s (default) -> cadence = {new_cadence}s"
    )

else:
    new_cadence = tc_n * tc / interval_length
    print(
        f"Using the 1/e trick, tc was calculated to be {np.round(tc,2)}s -> data resampled to new cadence of {np.round(new_cadence,2)}s, for {tc_n}tc across {interval_length} points"
    )

tc_list.append(tc)
cadence_list.append(new_cadence)

try:
    interval_approx_resampled = df.resample(
        str(np.round(new_cadence, 3)) + "S"
    ).mean()  # Resample to higher frequency

    for i in range(
        0, len(interval_approx_resampled) - interval_length + 1, interval_length
    ):
        interval = interval_approx_resampled.iloc[i : i + interval_length]
        # Check if interval is complete
        if interval.isnull().sum() / len(interval) < 0.01:
            # Linear interpolate
            interval = interval.interpolate(method="linear")
            int_norm = utils.normalize(interval)
            ints.append(int_norm)
        else:
            print("Too many NaNs in interval, skipping")


except Exception as e:
    print(f"An error occurred: {e}")

print(
    "Given this correlation length, this file yields",
    len(ints),
    "standardised interval/s (see details below)",
)
if len(ints) == 0:
    print("NO GOOD INTERVALS WITH GIVEN SPECIFICATIONS: not proceeding with analysis")

else:
    print("These will be now decimated in", times_to_gap, "different ways:")

    # Delete original dataframes
    del df_raw

    fig, ax = plt.subplots(figsize=(9, 3))
    plt.plot(df, alpha=0.3, c="black")
    plt.axvline(df.index[0], c="black", linestyle="dashed")
    [
        plt.axvline(interval.index[-1], c="black", linestyle="dashed")
        for interval in ints
    ]
    [plt.plot(interval, c="black") for interval in ints]
    plt.axhline(0, c="black", linewidth=0.5, linestyle="--")
    plt.suptitle(
        f"Standardised solar wind interval/s from {spacecraft}, given local conditions",
        y=1.1,
        fontsize=20,
    )
    # Add subtitle
    plt.title(
        f"{tc_n}$\lambda_C$ ($\lambda_C=${int(tc)}s) across {interval_length} points, $\langle x \\rangle=0$, $\sigma=1$"
    )

    # ax.set_xlim(interval_list_approx[0].index[0], interval_list_approx[2].index[-1])
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )
    ax.set_ylabel(f"{interval.name}")

    output_file_path = (
        raw_file_list[file_index]
        .replace("data/raw", "plots/temp")
        .replace(".cdf", "_ints.jpg")
    )
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()

    files_metadata = pd.DataFrame(
        {
            "file_index": file_index,
            "file_start": df.index[0],
            "file_end": df.index[-1],
            "tc": tc_list,
            "cadence": cadence_list,
        }
    )

    ints_metadata = pd.DataFrame(
        {
            "int_start": [interval.index[0] for interval in ints],
            "int_end": [interval.index[-1] for interval in ints],
        }
    )
    ints_metadata.reset_index(inplace=True)
    ints_metadata.rename(columns={"index": "int_index"}, inplace=True)
    ints_metadata.insert(0, "file_index", file_index)
    print(ints_metadata.head())

    # Analyse intervals (get true SF and slope)

    lags = np.arange(1, 0.1 * len(ints[0]))

    # Logarithmically-spaced lags?
    # vals = np.logspace(0, 3, 0.25 * len(ints[0]))
    # lags = np.unique(vals.astype(int))

    powers = [2]
    pwrl_range = [10, 100]

    sfs = pd.DataFrame()

    for i, input in enumerate(ints):
        # print(f"\nCore {core} processing standardised interval {i}")
        good_output, slope = sf.compute_sf(
            pd.DataFrame(input), lags, powers, False, False, pwrl_range
        )
        good_output.insert(0, "int_index", i)
        good_output.insert(1, "file_index", file_index)
        sfs = pd.concat([sfs, good_output])
        ints_metadata.loc[ints_metadata["int_index"] == i, "slope"] = slope

    # NON-ESSENTIAL: plot example SF + slope
    check_int = 0
    slope = ints_metadata.loc[ints_metadata["int_index"] == check_int, "slope"].values[
        0
    ]
    timestamp = ints_metadata.loc[ints_metadata["int_index"] == check_int, "int_start"][
        0
    ]

    plt.plot(
        sfs.loc[sfs["int_index"] == check_int, "lag"],
        sfs.loc[sfs["int_index"] == check_int, "sf_2"],
    )
    dif.pltpwrl(10, 0.1, 10, 100, slope, lw=4, ls="--", color="black")
    # Annotate with slope
    plt.annotate(
        f"Log-log slope: {slope:.3f}",
        xy=(0.3, 0.8),
        xycoords="axes fraction",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5),
        fontsize=14,
    )
    plt.semilogx()
    plt.semilogy()
    plt.title(f"Original $S_2(\\tau)$ for interval beginning {timestamp}")

    output_file_path = (
        raw_file_list[file_index]
        .replace("data/raw", "plots/temp")
        .replace(".cdf", "_sf_example.jpg")
    )
    plt.savefig(output_file_path, bbox_inches="tight")
    plt.close()

    # Duplicate, gap, interpolate, re-analyse intervals

    # Here we gap the original intervals different ways, then calculate SF and corresponding slope for gappy (naive) and
    # interpolated (lint) versions of each of these duplicate intervals.

    index_list = []
    version_list = []
    handling_list = []
    missing_list = []
    missing_chunks_list = []
    slopes_list = []

    sfs_gapped = pd.DataFrame()
    ints_gapped = pd.DataFrame()

    for index in range(len(ints)):
        input = ints[index]

        for j in range(times_to_gap):
            total_removal = np.random.uniform(0, 0.95)
            ratio_removal = np.random.uniform(minimum_missing_chunks, 1)
            # print("Nominal total removal: {0:.1f}%".format(total_removal * 100))
            # print("Nominal ratio: {0:.1f}%".format(ratio_removal * 100))
            prop_remove_chunks = total_removal * ratio_removal
            prop_remove_unif = total_removal * (1 - ratio_removal)
            bad_input_chunks, bad_input_ind_chunks, prop_removed_chunks = (
                ts.remove_data(
                    input, prop_remove_chunks, chunks=np.random.randint(1, 10)
                )
            )
            # Add the uniform gaps on top of chunks gaps
            bad_input, bad_input_ind, prop_removed = ts.remove_data(
                bad_input_chunks, prop_remove_unif
            )
            if prop_removed >= 0.95 or prop_removed == 0:
                # print(">95% or 0% data removed, skipping")
                continue

            bad_output = sf.compute_sf(
                pd.DataFrame(bad_input), lags, powers, False, False
            )
            bad_output["file_index"] = file_index
            bad_output["int_index"] = index
            bad_output["version"] = j
            bad_output["gap_handling"] = "naive"
            sfs_gapped = pd.concat([sfs_gapped, bad_output])

            for handling in ["naive", "lint"]:
                index_list.append(index)
                version_list.append(j)
                missing_list.append(prop_removed * 100)
                missing_chunks_list.append(prop_removed_chunks * 100)

                handling_list.append(handling)

                if handling == "naive":
                    slopes_list.append(slope)
                    # Once we are done with computing the SF, add some metadata to the interval
                    bad_input_df = pd.DataFrame(bad_input)
                    bad_input_df.reset_index(inplace=True)
                    bad_input_df["file_index"] = file_index
                    bad_input_df["int_index"] = index
                    bad_input_df["version"] = j
                    bad_input_df["gap_handling"] = handling
                    ints_gapped = pd.concat([ints_gapped, bad_input_df])

                elif handling == "lint":
                    interp_input = bad_input.interpolate(method="linear")
                    interp_output = sf.compute_sf(
                        pd.DataFrame(interp_input), lags, powers, False, False
                    )

                    # # Once we are done with computing the SF, add some metadata to the interval
                    interp_input_df = pd.DataFrame(interp_input)
                    interp_input_df.reset_index(
                        inplace=True
                    )  # Make time a column, not an index
                    interp_input_df["file_index"] = file_index
                    interp_input_df["int_index"] = index
                    interp_input_df["version"] = j
                    interp_input_df["gap_handling"] = handling
                    ints_gapped = pd.concat([ints_gapped, interp_input_df])

                    interp_output["file_index"] = file_index
                    interp_output["int_index"] = index
                    interp_output["version"] = j
                    interp_output["gap_handling"] = handling

                    # Correcting sample size and uncertainty for linear interpolation, same values as no handling
                    interp_output["n"] = bad_output["n"]
                    interp_output["missing_percent"] = bad_output["missing_percent"]
                    interp_output["sf_2_se"] = bad_output["sf_2_se"]

                    sfs_gapped = pd.concat([sfs_gapped, interp_output])

    ints_gapped_metadata = pd.DataFrame(
        {
            "file_index": file_index,
            "int_index": index_list,
            "version": version_list,
            "missing_percent_overall": missing_list,
            "missing_percent_chunks": missing_chunks_list,
            "gap_handling": handling_list,
        }
    )

    print("Exporting processed dataframes to pickle file")

    output_file_path = (
        raw_file_list[file_index].replace("raw", "processed").replace(".cdf", ".pkl")
    )

    with open(output_file_path, "wb") as f:
        pickle.dump(
            {
                "files_metadata": files_metadata,
                "ints_metadata": ints_metadata,
                "ints": ints,
                "ints_gapped_metadata": ints_gapped_metadata,
                "ints_gapped": ints_gapped,
                "sfs": sfs,
                "sfs_gapped": sfs_gapped,
            },
            f,
        )
