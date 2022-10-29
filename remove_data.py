import random
import pandas as pd
import numpy as np

def remove_data(array, proportion, chunks = None, sigma = 0.1):
    num_obs = proportion * len(array)

    if chunks is None:
        remove_idx = random.sample(range(len(array)), int(num_obs))

    else:
        obs_per_chunk = num_obs/chunks
        mean_obs = num_obs/chunks
        std = sigma*0.341*2*mean_obs
        remove_idx = []

        for i in range(chunks) :
            num_obs = round(random.gauss(mu = mean_obs, sigma = std))
            #Comment out the line above and replace num_obs below with mean_obs to revert to equal sized chunks
            if num_obs < 0:
                raise Exception('sigma too high, got negative obs')
            y = 0
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

    # Will be somewhat different from value specified
    prop_removed = np.sum(pd.isna(array_bad))/len(array)

    return array_bad, prop_removed