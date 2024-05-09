# SF error analysis
# ### Semi-empirical approach to computing second-order statistics of gapped series
#
# Statistical moments of increments.
# $$D_p(\tau)=\langle | x(t+\tau)-x(t))^p | \rangle$$

import glob
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import ts_dashboard_utils as ts
import utils as utils  # copied directly from Reynolds project, normalize() added
import sf_funcs as sf
import sys
import data_import_funcs as dif

# Setting up parallel processing (or not, if running locally)
try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

except ImportError:
    # Set default/empty single-process values if MPI is not available
    print("MPI not available, running in single-process mode.")

    class DummyComm:
        def Get_size(self):
            return 1

        def Get_rank(self):
            return 0

        def Barrier(self):
            pass

        def bcast(self, data, root=0):
            return data

    comm = DummyComm()
    size = comm.Get_size()
    rank = comm.Get_rank()
    status = None

# ## Load in the data
# A magnetic field time series from PSP

# Get list of files in psp directory and split between cores
# (if running in parallel)
raw_file_list = sorted(glob.iglob("data/raw/psp/" + "/*.cdf"))  # LOCAL
# raw_file_list = sorted(
#     #    glob.iglob("data/raw/psp/fields/l2/mag_rtn/2018/" + "/*.cdf")
#     glob.iglob(
#         "/nfs/scratch/wrenchdani/time_series_analysis/data/raw/psp/fields/l2/mag_rtn/2019/"
#         + "/*.cdf"
#     )
# )  # HPC

file_list_split = np.array_split(raw_file_list, size)

# Broadcast the list of files to all cores
file_list_split = comm.bcast(file_list_split, root=0)

# For each core, load in the data from the files assigned to that core
print("Core ", rank, "reading data")
psp_data = dif.read_cdfs(
    file_list_split[rank],
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

df_raw = psp_df["B_R"]

# Optionally, check the data for missing data and its frequency, get summary stats

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
# print("\nFrequency is {0:.2f} Hz (2dp)".format(x_freq))

# print("Mean = {}".format(np.mean(x)))
# print("Standard deviation = {}\n".format(np.std(x)))

### 0PTIONAL CODE END ###

# ## Standardise each interval to contain 8 correlation times


# Original freq is 0.007s. Resampling to less rapid but still sufficiently high cadence, then splitting into chunks with ~15 correlation times

tce_approx = 500  # s
tce_approx_n = 15
cadence_approx = 0.1  # s

tce_n = 8  # Number of correlation times we want...
interval_length = 1000  # ...across this many points
good_inputs_list = []

df = df_raw.resample(str(cadence_approx) + "S").mean()
interval_length_approx = int(tce_approx * tce_approx_n / cadence_approx)

# We have approximately 10 correlation times in 10,000 points. Let's now be more precise, and calculate the correlation time from each chunk

# Split df into subsets

interval_list_approx = [
    df[i : i + interval_length_approx]
    for i in range(0, len(df) - interval_length_approx + 1, int(interval_length_approx))
]

del df  # free up memory

for interval_approx in interval_list_approx:
    time_lags_lr, r_vec_lr = utils.compute_nd_acf(
        [interval_approx],
        nlags=10000,
        plot=False,
    )

    tce = utils.compute_outer_scale_exp_trick(time_lags_lr, r_vec_lr, plot=False)

    if tce == -1:
        tce = 500
        new_cadence = tce_n * tce / interval_length
        # print(
        #     f"tce not found for this interval, setting to 500s (default) -> cadence = {new_cadence}s"
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

print(
    "\nNumber of standardised intervals: " + str(len(good_inputs_list))
    # + "\n(may be more than one per original chunk for small cadences)"
)

# Logarithmically-spaced lags?
# vals = np.logspace(0, 3, 0.25 * len(good_inputs_list[0]))
# lags = np.unique(vals.astype(int))
lags = np.arange(1, 0.25 * len(good_inputs_list[0]))
powers = [0.5, 2]
times_to_gap = int(sys.argv[1])

good_outputs_list = []
all_bad_inputs_list = []
all_bad_outputs_list = []
all_interp_inputs_list = []
all_interp_outputs_list = []

print("Core ", rank, "creating gaps and calculating structure functions")
for i, input in enumerate(good_inputs_list):
    # print(f"\nCore {rank} processing standardised interval {i}")
    good_output = sf.compute_sf(pd.DataFrame(input), lags, powers)
    good_outputs_list.append(good_output)

    bad_inputs_list = []
    bad_outputs_list = []
    interp_inputs_list = []
    interp_outputs_list = []

    for total_removal in np.random.uniform(0, 0.9, times_to_gap):
        # Remove data (up to about 90%, may be some numerical issues with large %)
        # in both chunks and uniformly - split given by ratio_removal
        ratio_removal = np.random.uniform()
        # print("Nominal total removal: {0:.1f}%".format(total_removal * 100))
        # print("Nominal ratio: {0:.1f}%".format(ratio_removal * 100))
        prop_remove_chunks = total_removal * ratio_removal
        prop_remove_unif = total_removal * (1 - ratio_removal)
        bad_input_temp, bad_input_ind, prop_removed = ts.remove_data(
            input, prop_remove_chunks, chunks=np.random.randint(1, 10)
        )
        bad_input, bad_input_ind, prop_removed = ts.remove_data(
            bad_input_temp, prop_remove_unif
        )
        if prop_removed >= 0.95 or prop_removed == 0:
            # print(">95% or 0% data removed, skipping")
            continue

        # print(
        #     "Core {0} removed {1:.1f}% (approx. {2:.1f}% in chunks, {3:.1f}% uniformly from int {4})".format(
        #         rank,
        #         prop_removed * 100,
        #         prop_remove_chunks * 100,
        #         prop_remove_unif * 100,
        #         i,
        #     )
        # )

        bad_inputs_list.append(bad_input.values)

        # Linearly interpolate the missing data
        interp_input = bad_input.interpolate(method="linear")
        interp_inputs_list.append(interp_input.values)

        bad_output = sf.compute_sf(pd.DataFrame(bad_input), lags, powers)
        for estimator in ["classical", "ch", "dowd"]:
            bad_output[estimator + "_error"] = (
                bad_output[estimator] - good_output[estimator]
            )
            bad_output[estimator + "_error_percent"] = (
                bad_output[estimator + "_error"] / good_output[estimator] * 100
            )
        bad_output["missing_prop_overall"] = prop_removed
        bad_output["lint"] = False
        bad_outputs_list.append(bad_output)

        interp_output = sf.compute_sf(pd.DataFrame(interp_input), lags, powers)

        for estimator in ["classical", "ch", "dowd"]:
            interp_output[estimator + "_error"] = (
                interp_output[estimator] - good_output[estimator]
            )
            interp_output[estimator + "_error_percent"] = (
                interp_output[estimator + "_error"] / good_output[estimator] * 100
            )
        interp_output["missing_prop_overall"] = prop_removed
        interp_output["missing_prop"] = bad_output["missing_prop"]
        interp_output["missing_prop"] = bad_output["missing_prop"]
        interp_output["classical_se"] = bad_output["classical_se"]
        # NOTE: Seems sensible uncertainty is the same for both
        interp_output["lint"] = True
        interp_outputs_list.append(interp_output)

    all_bad_inputs_list.append(bad_inputs_list)
    all_bad_outputs_list.append(bad_outputs_list)
    all_interp_inputs_list.append(interp_inputs_list)
    all_interp_outputs_list.append(interp_outputs_list)

# converting from pd.Series to list of np.arrays to save space
all_good_inputs_list = [interval.values for interval in good_inputs_list]

print("Core ", rank, " saving outputs")
# Export each list of outputs to a pickle file
list_of_list_of_dfs = [
    good_inputs_list,
    good_outputs_list,
    all_bad_inputs_list,
    all_bad_outputs_list,
    all_interp_inputs_list,
    all_interp_outputs_list,
]

with open(
    f"data/processed/sfs_psp_core_{rank}.pkl",
    # f"/nfs/scratch/wrenchdani/time_series_analysis/data/processed/sfs_psp_core_{rank}.pkl",
    "wb",
) as f:
    pickle.dump(list_of_list_of_dfs, f)

print("Core ", rank, " finished")

# # Quick check of results
# fig, ax = plt.subplots(2, 2)
# for i in range(2):
#     ax[i, 0].plot(good_inputs_list[-i].values)
#     ax[i, 0].plot(all_interp_inputs_list[-i][-1])
#     ax[i, 0].plot(all_bad_inputs_list[-i][-1])
#     ax[i, 1].plot(good_outputs_list[-i]["classical"])
#     ax[i, 1].plot(all_interp_outputs_list[-i][-1]["classical"])
#     ax[i, 1].plot(all_bad_outputs_list[-i][-1]["classical"])

# plt.savefig("data/processed/validation_plot.png")
# print("Validation plot saved")
