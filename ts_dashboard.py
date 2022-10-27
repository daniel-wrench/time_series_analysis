# TO-DO

# Add data removal feature (test stats on this outside of dashboard)


import pandas as pd
import seaborn as sb
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import fft
from remove_data import remove_data
from astropy import timeseries


sb.set()

st.title("The Effect of Missing Data on Statistics of Time Series")

########################################################################

psp_df = pd.read_pickle("data_processed/psp_fld_l2_mag_rtn_201811.pkl")
psp_df_scalar = psp_df.loc[:, "B_R"][:10000]
psp = psp_df_scalar.values
psp_freq = 73.24

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

@st.cache
def simulate_stochastic_processes():
    wn = np.random.normal(size=10000)
    bm = np.cumsum(wn)
    return wn, bm

########################################################################

datasets = ["White noise", "Brownian motion", "Solar wind magnetic field (PSP)"]
dataset = st.sidebar.selectbox("Select the dataset", datasets, index=0)
missing = st.sidebar.slider("Select amount of data to remove", 0, 100)
removal_type = st.sidebar.radio("Select how to remove data", ["Uniformly", "In chunks"])

wn, bm = simulate_stochastic_processes()

dset = pd.DataFrame({
    "name": datasets,
    "data": [wn, bm, psp],
    "freq": [1, 1, psp_freq]})
dset = dset.set_index("name")

#################################

st.write("You have selected ", dataset, " with ", missing, "% removed ", str.lower(removal_type))

x = dset.data[dataset]
x_freq = dset.freq[dataset]

@st.cache
def calculate_stats():

    # Calculate FT
    N = len(x)
    T = 1/x_freq
    freqs_positive = fft.fftfreq(N, T)[:N//2]
    # ft_x = fft.fft(x, norm = "backward") # Normalisation of 1/n only on the backward term
    # power_ft = 1.0/N * np.abs(ft_x[0:(N//2)])**2

    # Removing any NA values - leads to clean but un-evenly sampled data for LS method
    power_ft = timeseries.LombScargle(np.arange(N)/x_freq, x, normalization="psd").power(freqs_positive[1:])

    # Calculate ACF
    acf = sm.tsa.acf(x, nlags=len(x), missing = "conservative") # also available, "drop", but must reduce nlags to number of non-missing obs
    
    # Calculate SF
    sfn = calc_sfn(pd.Series(x), [2], 1, 0.9999)

    return freqs_positive, power_ft, acf, sfn["2"].values

@st.cache
def calculate_missing_stats(missing, removal_type):

    if removal_type == "In chunks":
        x_bad, prop_removed = remove_data(x, missing/100, chunks = 10, sigma = 0.1)
    elif removal_type == "Uniformly": 
        x_bad, prop_removed = remove_data(x, missing/100)

    # Calculate FT
    N = len(x_bad)
    T = 1/x_freq
    freqs_positive = fft.fftfreq(N, T)[:N//2]
    # ft_x = fft.fft(x, norm = "backward") # Normalisation of 1/n only on the backward term
    # power_ft = 1.0/N * np.abs(ft_x[0:(N//2)])**2

    # Removing any NA values - leads to clean but un-evenly sampled data for LS method
    power_ft = timeseries.LombScargle((np.arange(N)/x_freq)[pd.notna(x_bad)], x[pd.notna(x_bad)], normalization="psd").power(freqs_positive[1:])

    # Calculate ACF
    acf = sm.tsa.acf(x_bad, nlags=len(x_bad), missing = "conservative") # also available, "drop", but must reduce nlags to number of non-missing obs
    
    # Calculate SF
    sfn = calc_sfn(pd.Series(x_bad), [2], 1, 0.9999)

    return x_bad, prop_removed, freqs_positive, power_ft, acf, sfn["2"].values

freqs_positive, power_ft, acf, sfn = calculate_stats()

fig_data, ax_data = plt.subplots()
sb.lineplot(data = x, ax=ax_data, color = "green").set(title="Time series")

fig_acf, ax_acf = plt.subplots()
sb.lineplot(data = acf, ax = ax_acf, color = "green").set(title="Autocorrelation")

fig_sf, ax_sf = plt.subplots()
sb.lineplot(data = sfn, ax= ax_sf, color = "green").set(title="Second-order structure function")

fig_psd, ax_psd = plt.subplots()
sb.lineplot(data = pd.Series(power_ft, freqs_positive[1:]), ax= ax_psd, color = "green").set(title="Power spectrum", xscale = "log", yscale = "log")

if missing > 0:
    x_missing, prop_removed, freqs_positive_missing, power_ft_missing, acf_missing, sfn_missing = calculate_missing_stats(missing, removal_type)
    
    sb.lineplot(data = x_missing, ax=ax_data, color = "red")
    sb.lineplot(data = acf_missing, ax = ax_acf, color = "red")
    sb.lineplot(data = sfn_missing, ax=ax_sf, color = "red")

    # Couldn't get power_ft_missing to nicely overlay power_ft, hence combining into df in this workaround
    fig_psd, ax_psd = plt.subplots()
    psd_bad = pd.Series(power_ft_missing, freqs_positive_missing[1:])
    psd_good = pd.Series(power_ft, freqs_positive[1:])
    df = pd.DataFrame({"good":psd_good, "bad":psd_bad})

    palette = {"good":"green", "bad":"red"}

    sb.lineplot(data = df, ax=ax_psd, palette = palette, dashes = False, legend=False).set(title="Power spectrum", xscale = "log", yscale = "log")

st.pyplot(fig_data)
st.pyplot(fig_acf)
st.pyplot(fig_sf)
st.pyplot(fig_psd)

# fig_ft = go.Figure()
# fig_ft.add_trace(go.Scatter(x=freqs_positive[1:], y=power_ft, name='FT of complete data'))
# st.plotly_chart(fig_ft)