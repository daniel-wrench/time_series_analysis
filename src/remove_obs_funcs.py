
#####These functions remove observations at random (individually or in chunks) from a dated DataFrame (also available for arrays)

# https://docs.python.org/3/library/random.html#module-random 

import pandas as pd
import numpy as np
import random
import copy

#DataFrame versions (WORKING)

def remove_obs_df(df, percent):
    num_obs = percent * len(df)
    remove = random.sample(range(len(df)), int(num_obs))
    final_dataset = df.drop(df.index[remove])

    print("The indexes of the removed observations are:")
    print(remove)
    print("The proportion of data removed is:", format(1 - len(final_dataset)/len(df)))
    return final_dataset

def remove_chunks_df(df, proportion, chunks, sigma, missing_ind_col = False):
    '''Remove randomly-sized chunks from random points in a dataframe (such that at least the proportion specified is removed)
    Works for single column dataframes and multiple column dataframes, where it removes chunks in the same places.
    Parameters:
    
    - df = DataFrame
    - proportion = total proportion (decimal) of observations to remove
    - chunks = number of chunks of observations to remove
    - sigma = variance of sizes of chunks. Chunk sizes are drawn from a normal distribution with mean = len(df)*proportion/chunks and standard deviation = sigma*0.341*2*mean. If this is too large the function will return an error due to negative chunk sizes being selected.
    - missing_ind_col = whether to create a missing indicator column

    Returns:
    - The original dataframe with some values replaced with NA, and an optional additional column "missing" that indicates whether a value has been removed at that timestamp (1 = removed)
    '''
    num_obs = proportion * len(df)
    mean_obs = num_obs/chunks
    std = sigma*0.341*2*mean_obs
    all_removes = []

    for i in range(chunks) :
        num_obs = round(random.gauss(mu = mean_obs, sigma = std))
        #Comment out the line above and replace num_obs below with mean_obs to revert to equal sized chunks
        if num_obs < 0:
            raise Exception('sigma too high, got negative obs')
        y = 0
        start = random.randrange(start = 1, stop = len(df) - num_obs) #Starting point for each removal should be far enough from the start and end of the series
        remove = np.arange(start, start + num_obs)

        all_removes.extend(remove)
        
    prop_missing = len(np.unique(all_removes))/len(df)

    while prop_missing < proportion:
        start = random.randrange(start = 1, stop = len(df) - num_obs) #Starting point for each removal should be far enough from the start and end of the series
        remove = np.arange(start, start + num_obs)
        all_removes.extend(remove)

        prop_missing = len(np.unique(all_removes))/len(df)

    all_removes = [int(x) for x in all_removes] #Converting decimals to integers
    
    # Setting the chosen indices as NA
    
    #final_dataset = df.copy()
    final_dataset = copy.deepcopy(df)
    final_dataset.iloc[all_removes,:] = np.nan

    if missing_ind_col==True:
        # Creating a binary indicator vector where values have been removed
        indicator_vec = np.zeros(len(df))
        indicator_vec[all_removes] = 1

        #Adding this indicator vector as a column to the dataframe
        final_dataset['missing'] = indicator_vec

    #print("The proportion of data removed is:", sum(final_dataset.missing/len(df)))
    return final_dataset

###TESTING FUNCTION

# returns = np.random.normal(loc=0.02, scale=0.05, size=1000)
# df = pd.DataFrame({'col1':returns, 'col2':returns+1})
# dates = pd.date_range('2011', periods = len(returns))
# df = df.set_index(dates)

# plt.plot(df)
# plt.show()

# new_data_2 = remove_chunks_df(df, 0, 5, 0.5) 
# new_data_2 = new_data_2.resample('D').mean()
# new_data_2.iloc[:,0:2].plot()
# new_data_2.plot()
# plt.show()

#Proportion missing
#new_data_2.iloc[:,1].isnull().sum()/len(new_data_2)

#Array versions

def remove_obs_array(array, percent):
    num_obs = percent * len(array)
    remove = random.sample(range(len(array)), int(num_obs))
    return np.delete(array, remove)

def remove_chunks_array(array, percent, chunks):
    num_obs = percent * len(array)
    obs_per_chunk = num_obs/chunks
    all_removes = []
    for i in range(chunks) :
        
        start = random.randrange(len(array))
        remove = np.arange(start, start + obs_per_chunk)
        all_removes.extend(remove)
    all_removes = [int(x) for x in all_removes]
    #print(all_removes)
    return np.delete(array, all_removes)
