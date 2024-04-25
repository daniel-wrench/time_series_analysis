import numpy as np
import pandas as pd
import random


def calc_sfn(data, p, freq=1, max_lag_prop=0.2):
    # Calculate lags
    lag_function = {}
    for i in np.arange(
        1, round(max_lag_prop * len(data))
    ):  # Limiting maximum lag to 20% of dataset length
        lag_function[i] = data.diff(i)

    # Initialise dataframe
    structure_functions = pd.DataFrame(index=np.arange(1, len(data)))

    # Converting lag values from points to seconds
    structure_functions["lag"] = structure_functions.index / freq

    for order in p:
        lag_dataframe = (
            pd.DataFrame(lag_function) ** order
        )  # or put in .abs() before order: this only changes the odd-ordered functions
        structure_functions[str(order)] = pd.DataFrame(lag_dataframe.mean())

    return structure_functions.dropna()


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


def remove_data(array, proportion, chunks=None):
    """
    Function to remove data from a time series array, either randomly or in chunks

    Args:
        array: numpy array or pd.Series, time series data
        proportion: float, proportion of data to remove
        chunks: int, number of chunks to remove data in

    Returns:
        array_bad: numpy array, time series data with missing values
        array_bad_idx: numpy array, indices of the missing values
        prop_removed: float, proportion of data removed
    """

    num_obs = proportion * len(array)
    remove_idx = []

    if chunks is None:
        remove_idx = np.random.choice(len(array), size=int(num_obs), replace=False)
    else:
        num_obs_per_chunk = int(num_obs / chunks)
        # std = sigma * 0.341 * 2 * mean_obs

        for _ in range(chunks):
            start = np.random.randint(low=1, high=len(array) - num_obs_per_chunk)
            remove_idx.extend(range(start, start + num_obs_per_chunk))

    array_bad = array.copy()
    array_bad[remove_idx] = np.nan

    prop_removed = np.sum(np.isnan(array_bad)) / len(array)
    # Will be somewhat different from value specified if removed in chunks

    # Below are needed if interpolating a numpy array, rather than a dataframe
    idx = np.arange(len(array))
    array_bad_idx = np.delete(idx, remove_idx)
    # In which case the interpolation looks like so:
    # bad_input_adjacent = bad_input[pd.notna(bad_input)]
    # interp_input = np.interp(
    #     np.arange(len(input)), bad_input_ind, bad_input_adjacent
    # )

    return array_bad, array_bad_idx, prop_removed
