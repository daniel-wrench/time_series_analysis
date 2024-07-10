## WIND TEST SET DATA PROCESSING

import glob
import pickle
import pandas as pd
import numpy as np
import ts_dashboard_utils as ts
import src.utils as utils  # copied directly from Reynolds project, normalize() added
import src.sf_funcs as sf
import sys
import src.data_import_funcs as dif
import matplotlib.pyplot as plt

### CHECK RESULTS

with open("data/processed/sfs_wind_core_0.pkl", "rb") as f:
    check = pickle.load(f)


###


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


# import pickle file
with open("data/processed/standardised_wind.pkl", "rb") as f:
    good_inputs_list = pickle.load(f)


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
    f"data/processed/sfs_wind_core_{rank}.pkl",
    # "/nfs/scratch/wrenchdani/time_series_analysis/data/processed_small/sfs_psp_core_{0:03d}.pkl".format(rank),
    "wb",
) as f:
    pickle.dump(list_of_list_of_dfs, f)

print("Core ", rank, " finished")

# # Quick check of results
fig, ax = plt.subplots(2, 2)
for i in range(2):
    ax[i, 0].plot(good_inputs_list[-i].values)
    ax[i, 0].plot(all_interp_inputs_list[-i][-1])
    ax[i, 0].plot(all_bad_inputs_list[-i][-1])
    ax[i, 1].plot(good_outputs_list[-i]["classical"])
    ax[i, 1].plot(all_interp_outputs_list[-i][-1]["classical"])
    ax[i, 1].plot(all_bad_outputs_list[-i][-1]["classical"])

plt.savefig("data/processed/validation_plot.png")
print("Validation plot saved")
