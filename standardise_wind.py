import numpy as np

# import os
# while ".toplevel" not in os.listdir():
#     os.chdir("..")

import src.utils as utils
import src.params as params

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

files = [
    "data/raw/wind/wi_h2_mfi_20160101_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160102_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160103_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160104_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160105_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160106_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160107_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160108_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160109_v05.cdf",
    "data/raw/wind/wi_h2_mfi_20160110_v05.cdf",
    
]

# Original freq is 0.007s. Resampling to less rapid but still sufficiently high cadence, then splitting into chunks with ~15 correlation times

tce_approx = 3000  # s
tce_approx_n = 15
cadence_approx = 0.1  # s

tce_n = 10  # Number of correlation times we want...
interval_length = 10000  # ...across this many points
good_inputs_list = []

for file in files:
    df = utils.pipeline(
        file,
        varlist=sys_arg_dict["mag_vars"],
        thresholds=sys_arg_dict["mag_thresh"],
        cadence=sys_arg_dict["dt_hr"],
    )
    # print("Reading {0}: {1:.2f}% missing".format(file, df.iloc[:,0].isna().sum()/len(df)*100))

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

    df_raw = int_wind_hr["Bx"]

    df = df_raw.resample(str(cadence_approx) + "S").mean()
    interval_length_approx = int(tce_approx * tce_approx_n / cadence_approx)

    # We have approximately 15 correlation times in 10,000 points. Let's now be more precise, and calculate the correlation time from each chunk

    # Split df into subsets

    interval_list_approx = [
        df[i : i + interval_length_approx]
        for i in range(
            0, len(df) - interval_length_approx + 1, int(interval_length_approx)
        )
    ]

    del df  # free up memory

    for interval_approx in interval_list_approx:
        time_lags_lr, r_vec_lr = utils.compute_nd_acf(
            [interval_approx],
            nlags=4000,
            plot=False,
        )

        tce = utils.compute_outer_scale_exp_trick(time_lags_lr, r_vec_lr, plot=False)

        if tce == -1:
            tce = 2000
            new_cadence = tce_n * tce / interval_length
            # print(
            #     f"tce not found for this interval, setting to 2000s (default) -> cadence = {new_cadence}s"
            # )

        else:
            new_cadence = tce_n * tce / interval_length
            # print(
            #     f"tce calculated to be {np.round(tce,2)}s -> cadence = {np.round(new_cadence,2)}s (for {tce_n}tce across {interval_length} points)"
            # )

        try:
            interval_approx_resampled = interval_approx.resample(
                str(np.round(new_cadence, 3)) + "S"
            ).mean()  # Resample to higher frequency

            for i in range(
                0, len(interval_approx_resampled) - interval_length + 1, interval_length
            ):
                interval = interval_approx_resampled.iloc[i : i + interval_length]
                # Check if interval is complete
                if interval.isnull().sum() > 0:
                    print(
                        "interval contains missing data even after down-sampling; skipping"
                    )
                    # Note: due to merging cannot identify specific file with missing data here
                    # only file list as here:
                    # print("corresponding input file list: ", file_list_split[rank])
                    continue
                else:
                    # print("Interval successfully processed")
                    int_norm = utils.normalize(interval)
                    good_inputs_list.append(int_norm)

        except Exception as e:
            print(f"An error occurred: {e}")
            continue

    if len(good_inputs_list) == 0:
        print("No good inputs found (good_inputs_list is empty). Exiting.")
        exit(1)

print(
    "\nNumber of standardised intervals: " + str(len(good_inputs_list))
    # + "\n(may be more than one per original chunk for small cadences)"
)

# for i in range(len(good_inputs_list)):
#     plt.plot(good_inputs_list[i])
# plt.show()
import pickle

# Export to pickle using with open
with open("standardised_wind.pkl", "wb") as f:
    pickle.dump(good_inputs_list, f)
