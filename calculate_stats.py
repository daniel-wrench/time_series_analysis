
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def calc_struct_sdk(data, freq, plot):
    #Calculate lags
    lag_function = {}
    for i in np.arange(1, round(0.2*len(data))): #Limiting maximum lag to 20% of dataset length
        lag_function[i] = data.diff(i)

    #Initialise dataframe
    structure_functions = pd.DataFrame(index = np.arange(1,len(data)))

    #Calculate different orders of structure functions
    for p in np.arange(1, 5):
        lag_dataframe = pd.DataFrame(lag_function).abs()**p
        structure_functions[p] = pd.DataFrame(lag_dataframe.mean())

    #Converting lag values from points to seconds
    structure_functions.index = structure_functions.index/freq 

    #Calculate kurtosis
    sdk = structure_functions[[2, 4]]
    sdk.columns = ['2', '4']
    sdk['2^2'] = sdk['2']**2
    sdk['kurtosis'] = sdk['4'].div(sdk['2^2'])

    if plot == True:

        #Plot structure functions
        fig, axs = plt.subplots(1,2)
        axs[0].plot(structure_functions)
        axs[0].semilogx()
        axs[0].semilogy()
        axs[0].set(title = 'Structure functions (orders 1-4)',
                xlabel = 'Lag (s)')
        axs[1].plot(sdk['kurtosis'])
        axs[1].semilogx()
        axs[1].set(title = 'Kurtosis', 
                xlabel = 'Lag (s)')
        plt.show()
    
    else:
        return structure_functions.dropna()

def plot_results(predictions_arr, observed_arr, no_samples, spacecraft, model, log = False):
    """Plot observed curves against predicted for a sample of *validation* intervals. To be used for hyperparameter tuning by visual inspection of predictions.
    
    Parameters:
    - log: Whether to use log-log plots or not
    """
    predicted = np.load(predictions_arr)
    predicted = pd.DataFrame(predicted)
    predicted = predicted.transpose()

    observed = np.load(observed_arr)
    observed = pd.DataFrame(observed)
    observed = observed.transpose()
    
    # plots will be shown on a nplotx,nplotx grid
    if np.modf(np.sqrt(no_samples))[0] == 0:
       nplotx=int(np.modf(np.sqrt(no_samples))[1])
    else: 
       nplotx=int(np.modf(np.sqrt(no_samples))[1]+1)

    fig, axs = plt.subplots(nplotx,nplotx,sharey=True, figsize = (10,10))

    if log == True:
        for i in np.arange(0, no_samples):
            r, c = int(np.arange(0, no_samples)[i]/nplotx), np.mod(np.arange(0, no_samples)[i],nplotx)
            axs[r, c].plot(observed[i], label = "observed" )
            axs[r, c].plot(predicted[i], label = "predicted")
            axs[r, c].semilogx()
            axs[r, c].semilogy()
    else: 
        for i in np.arange(0, no_samples):
            r, c = int(np.arange(0, no_samples)[i]/nplotx), np.mod(np.arange(0, no_samples)[i],nplotx)
            axs[r, c].plot(observed[i], label = "observed" )
            axs[r, c].plot(predicted[i], label = "predicted")

    axs[0,0].legend()
    axs[0,0].set(title = "Model predictions on " + spacecraft + " validation set")
    plt.savefig('results_interim/' + model + spacecraft + '_predictions_plot.png')
    plt.cla()


def normalize(data):
    # Remove any NA values to calculate mean and std, but leave them in the output set
    clean_data = data[~np.isnan(data)]
    mean = np.mean(clean_data)
    std = clean_data.std()
    result = (data - mean)/std
    return result


def plot_validation_error(path):
    loss = np.load(path + 'loss.npy')
    val_loss = np.load(path + 'val_loss.npy')

    # Get epoch at which val loss was minimum
    min_loss = pd.DataFrame(val_loss)[val_loss==val_loss.min()]

    plt.plot(loss)
    plt.plot(val_loss)
    plt.ylabel('loss (MSE)')
    plt.xlabel('epoch')
    plt.legend(['training set ', 'validation set'], loc='upper left') 
    plt.suptitle('MSE during training of neural network')
    plt.title("Min validation loss of " + str(round(min_loss.iloc[0,0], 4)) + " at epoch " + str(min_loss.index[0]),
    fontsize = 10)
    plt.savefig(path + 'nn_training_plot.png')
    plt.clf()
 
