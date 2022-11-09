import numpy as np
import pandas as pd
import random

def calc_sfn(data, p, freq=1, max_lag_prop=0.2):
    #Calculate lags
    lag_function = {}
    for i in np.arange(1, round(max_lag_prop*len(data))): #Limiting maximum lag to 20% of dataset length
        lag_function[i] = data.diff(i)

    #Initialise dataframe
    structure_functions = pd.DataFrame(index = np.arange(1,len(data)))

    # Converting lag values from points to seconds
    structure_functions["lag"] = structure_functions.index/freq 

    for order in p:
        lag_dataframe = pd.DataFrame(lag_function)**order # or put in .abs() before order: this only changes the odd-ordered functions
        structure_functions[str(order)] = pd.DataFrame(lag_dataframe.mean())
        
    return structure_functions.dropna()

# Get MSE between two curves
def calc_mse(curve1, curve2):
    mse = np.sum((curve1-curve2)**2)/len(curve1)
    if mse == np.inf:
      mse = np.nan
    return(mse) 

# Get MAPE between two curves
def calc_mape(curve1, curve2):
    curve1 = curve1 + 0.000001 # Have to add this so there is no division by 0
    mape = np.sum(np.abs((curve1-curve2)/curve1))/len(curve1)
    if mape == np.inf:
      mape = np.nan
    return(mape) 

def remove_data(array, proportion, chunks = None, sigma = 0.1):
    num_obs = proportion * len(array)

    if chunks is None:
        remove_idx = random.sample(range(len(array)), int(num_obs))

    else:
        mean_obs = num_obs/chunks
        std = sigma*0.341*2*mean_obs
        remove_idx = []

        for i in range(chunks) :
            num_obs = round(random.gauss(mu = mean_obs, sigma = std))
            #Comment out the line above and replace num_obs below with mean_obs to revert to equal sized chunks
            if num_obs < 0:
                raise Exception('sigma too high, got negative obs')
            start = random.randrange(start = 1, stop = len(array) - num_obs) #Starting point for each removal should be far enough from the start and end of the series
            remove = np.arange(start, start + num_obs)

            remove_idx.extend(remove)
            
        prop_missing = len(np.unique(remove_idx))/len(array)

        while prop_missing < proportion:
            start = random.randrange(start = 1, stop = len(array) - num_obs) #Starting point for each removal should be far enough from the start and end of the series
            remove = np.arange(start, start + num_obs)
            remove_idx.extend(remove)

            prop_missing = len(np.unique(remove_idx))/len(array)

    remove_idx = [int(x) for x in remove_idx] #Converting decimals to integers
    array_bad = array.copy()
    array_bad[remove_idx] = np.nan

    # Will be somewhat different from value specified if removed in chunks
    prop_removed = np.sum(pd.isna(array_bad))/len(array)
    idx = np.arange(len(array))
    array_bad_idx = np.delete(idx, remove_idx)

    return array_bad, array_bad_idx, prop_removed