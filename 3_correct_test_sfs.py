# STEP 3: FOR EACH INTERVAL IN TEST SET: apply correction
# As with 1_compute_sfs, only working on one file at a time


import pickle
import pandas as pd
import numpy as np
import src.sf_funcs as sf
import glob
import seaborn as sns
import sys

sns.set_theme(style="whitegrid", font_scale=1.5)

n_bins = 10
times_to_gap = 3
pwrl_range = [10, 100]

# Importing lookup table
lookup_table_2d = pd.read_csv(
    f"data/processed/lookup_table_2d_{n_bins}bins.csv", index_col=0
)
lookup_table_3d = pd.read_csv(
    f"data/processed/lookup_table_3d_{n_bins}bins.csv", index_col=0
)

spacecraft = sys.argv[1]  # "psp" or "wind"
file_index_test = int(sys.argv[2])
# this simply refers to one of the files in the test files, not the "file_index" variable referring to the original raw file

# Importing processed time series and structure functions
if spacecraft == "wind":
    input_file_list = [sorted(glob.glob("data/processed/wind/wi_*v05.pkl"))][0]
elif spacecraft == "psp":
    input_file_list = [sorted(glob.glob("data/processed/psp/test/psp_*v02.pkl"))][0]
else:
    raise ValueError("Spacecraft must be 'psp' or 'wind'")

(
    files_metadata,
    ints_metadata,
    ints,
    ints_gapped_metadata,
    ints_gapped,
    sfs,
    sfs_gapped,
) = sf.load_and_concatenate_dataframes([input_file_list[file_index_test]])
print(f"Loaded {input_file_list[file_index_test]}")


# Apply 2D and 3D scaling to test set, report avg errors
print(
    f"Correcting {len(ints_metadata)} intervals using 2D error heatmap with {n_bins} bins"
)
sfs_lint_corrected_2d = sf.compute_scaling(
    sfs_gapped[sfs_gapped["gap_handling"] == "lint"], "missing_percent", lookup_table_2d
)


print(
    f"Correcting {len(ints_metadata)} intervals using 3D error heatmap with {n_bins} bins"
)
sfs_lint_corrected_2d_3d = sf.compute_scaling_3d(
    sfs_lint_corrected_2d[sfs_lint_corrected_2d["gap_handling"] == "lint"],
    "missing_percent",
    lookup_table_3d,
)


correction_wide = sfs_lint_corrected_2d_3d[
    [
        "file_index",
        "int_index",
        "version",
        "lag",
        "missing_percent",
        "sf_2_corrected_2d",
        "sf_2_corrected_3d",
    ]
]
correction_long = pd.wide_to_long(
    correction_wide,
    ["sf_2"],
    i=["file_index", "int_index", "version", "lag", "missing_percent"],
    j="gap_handling",
    sep="_",
    suffix=r"\w+",
)
correction_bounds_wide = sfs_lint_corrected_2d_3d[
    [
        "file_index",
        "int_index",
        "version",
        "lag",
        "missing_percent",
        "sf_2_lower_corrected_2d",
        "sf_2_lower_corrected_3d",
        "sf_2_upper_corrected_2d",
        "sf_2_upper_corrected_3d",
    ]
]
correction_bounds_long = pd.wide_to_long(
    correction_bounds_wide,
    ["sf_2_lower", "sf_2_upper"],
    i=["file_index", "int_index", "version", "lag", "missing_percent"],
    j="gap_handling",
    sep="_",
    suffix=r"\w+",
)

corrections_long = pd.merge(
    correction_long,
    correction_bounds_long,
    how="inner",
    on=["file_index", "int_index", "version", "lag", "missing_percent", "gap_handling"],
).reset_index()


# Adding the corrections, now as a form of "gap_handling", back to the gapped SF dataframe
sfs_gapped_corrected = pd.concat([sfs_gapped, corrections_long])

# Merging the original SFs with the corrected ones to then calculate errors
sfs_gapped_corrected = pd.merge(
    sfs,
    sfs_gapped_corrected,
    how="inner",
    on=["file_index", "int_index", "lag"],
    suffixes=("_orig", ""),
)


# Calculate lag-scale errors (sf_2_pe)
# This is the first time we calculate these errors, for this specific dataset (they were calculated before for the training set)
#
# Previously this didn't work as we had two sf_2_orig columns as the result of merging a dataframe that had already previously been merged. However, this initial merge is no longer taking place, as it is only now that we are calculating any errors *of any sort, including lag-specific ones*, for this particular dataset.


sfs_gapped_corrected["sf_2_pe"] = (
    (sfs_gapped_corrected["sf_2"] - sfs_gapped_corrected["sf_2_orig"])
    / sfs_gapped_corrected["sf_2_orig"]
    * 100
)


# Calculate interval-scale errors
# This is the first time we do this. We do not need these values for the training set, because we only use that for calculating the correction factor, which uses lag-scale errors..


# Adding rows as placeholders for when we correct with 2D and 3D heatmaps and want to calculate errors

dup_df = ints_gapped_metadata.replace(
    ["naive", "lint"], ["corrected_2d", "corrected_3d"]
)
ints_gapped_metadata = pd.concat([ints_gapped_metadata, dup_df])


for i in files_metadata.file_index.unique():
    for j in range(len(ints_metadata["file_index"] == i)):
        for k in range(times_to_gap):
            for gap_handling in sfs_gapped_corrected.gap_handling.unique():
                # Calculate MAPE for 2D and 3D corrected SFs

                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == i)
                    & (ints_gapped_metadata["int_index"] == j)
                    & (ints_gapped_metadata["version"] == k)
                    & (ints_gapped_metadata["gap_handling"] == gap_handling),
                    "mape",
                ] = np.mean(
                    np.abs(
                        sfs_gapped_corrected.loc[
                            (sfs_gapped_corrected["file_index"] == i)
                            & (sfs_gapped_corrected["int_index"] == j)
                            & (sfs_gapped_corrected["version"] == k)
                            & (sfs_gapped_corrected["gap_handling"] == gap_handling),
                            "sf_2_pe",
                        ]
                    )
                )

                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == i)
                    & (ints_gapped_metadata["int_index"] == j)
                    & (ints_gapped_metadata["version"] == k)
                    & (ints_gapped_metadata["gap_handling"] == gap_handling),
                    "mpe",
                ] = np.mean(
                    sfs_gapped_corrected.loc[
                        (sfs_gapped_corrected["file_index"] == i)
                        & (sfs_gapped_corrected["int_index"] == j)
                        & (sfs_gapped_corrected["version"] == k)
                        & (sfs_gapped_corrected["gap_handling"] == gap_handling),
                        "sf_2_pe",
                    ]
                )

                # Calculate power-law slope for 2D and 3D corrected SFs
                current_int = sfs_gapped_corrected.loc[
                    (sfs_gapped_corrected["file_index"] == i)
                    & (sfs_gapped_corrected["int_index"] == j)
                    & (sfs_gapped_corrected["version"] == k)
                    & (sfs_gapped_corrected["gap_handling"] == gap_handling)
                ]

                # Fit a line to the log-log plot of the structure function over the given range

                slope = np.polyfit(
                    np.log(
                        current_int.loc[
                            (current_int["lag"] >= pwrl_range[0])
                            & (current_int["lag"] <= pwrl_range[1]),
                            "lag",
                        ]
                    ),
                    np.log(
                        current_int.loc[
                            (current_int["lag"] >= pwrl_range[0])
                            & (current_int["lag"] <= pwrl_range[1]),
                            "sf_2",
                        ]
                    ),
                    1,
                )[0]

                ints_gapped_metadata.loc[
                    (ints_gapped_metadata["file_index"] == i)
                    & (ints_gapped_metadata["int_index"] == j)
                    & (ints_gapped_metadata["version"] == k)
                    & (ints_gapped_metadata["gap_handling"] == gap_handling),
                    "slope",
                ] = slope


slope = np.polyfit(
    np.log(
        current_int.loc[
            (current_int["lag"] >= pwrl_range[0])
            & (current_int["lag"] <= pwrl_range[1]),
            "lag",
        ]
    ),
    np.log(
        current_int.loc[
            (current_int["lag"] >= pwrl_range[0])
            & (current_int["lag"] <= pwrl_range[1]),
            "sf_2",
        ]
    ),
    1,
)[0]


# Calculate slope errors
ints_gapped_metadata = pd.merge(
    ints_gapped_metadata,
    ints_metadata.drop(["int_start", "int_end"], axis=1),
    how="inner",
    on=["file_index", "int_index"],
    suffixes=("", "_orig"),
)


# maybe come back to this method of getting true slopes, could be fun

# # Create a dictionary from df2 with composite keys
# value2_dict = df2.set_index(['key1', 'key2'])['value2'].to_dict()

# # Create a composite key in df1 and map the values
# df1['composite_key'] = list(zip(df1['key1'], df1['key2']))
# df1['value2'] = df1['composite_key'].map(value2_dict)


ints_gapped_metadata["slope_pe"] = (
    (ints_gapped_metadata["slope"] - ints_gapped_metadata["slope_orig"])
    / ints_gapped_metadata["slope_orig"]
    * 100
)
ints_gapped_metadata["slope_ape"] = np.abs(ints_gapped_metadata["slope_pe"])


# Export the dataframes in one big pickle file
output_file_path = input_file_list[file_index_test].replace(".pkl", "_corrected.pkl")

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
