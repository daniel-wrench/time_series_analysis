# TO-DO

# Add FBm


import pandas as pd
import streamlit as st
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import fft
from remove_data import remove_data
from astropy import timeseries

st.set_page_config(layout="wide")

st.title("The Effect of Missing Data on Statistics of Time Series")

if 'sfn_log' not in st.session_state:
    st.session_state['sfn_log'] = True

########################################################################

psp_df = pd.read_pickle("psp_mag_scalar_int.pkl")
psp = psp_df.values
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
dataset = st.sidebar.selectbox("Select dataset", datasets, index=0)
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
ax_data.plot(x, c = '0.6')

fig_acf, ax_acf = plt.subplots()
ax_acf.plot(acf, c = "grey")

fig_sfn, ax_sfn = plt.subplots()
ax_sfn.plot(sfn, c = "grey")

fig_psd, ax_psd = plt.subplots()
ax_psd.plot(freqs_positive[1:], power_ft, c = "grey")
ax_psd.semilogx()
ax_psd.semilogy()

if missing > 0:
    x_missing, prop_removed, freqs_positive_missing, power_ft_missing, acf_missing, sfn_missing = calculate_missing_stats(missing, removal_type)
    
    ax_data.plot(x_missing, color = "black")
    ax_acf.plot(acf_missing, color = "black")
    ax_sfn.plot(sfn_missing, color = "black")
    ax_psd.plot(freqs_positive_missing[1:], power_ft_missing, color = "black")

col1, col2, col3, col4 = st.columns(4)

with col1:
   st.header("Time series")
   st.pyplot(fig_data)
   st.latex(r'''x(t)''')

with col2:
   st.header("Autocorrelation")
   st.pyplot(fig_acf)
   st.latex(r'''R(\tau)=\langle x(t)x(t+\tau)\rangle''')
   st.write("Missing values are allowed for by removing nans when computing the mean and cross-products that are used to estimate the autocovariance.")

with col3:
    st.header("Structure function")
    # Currently have opposite logic due to not quite right reactivity
    # (both this plot and the following plot reload after ticking the checkbox,
    # and uses previous value of checkbox)
    if st.session_state['sfn_log'] == False: 
        ax_sfn.semilogx()
        ax_sfn.semilogy()    
    st.pyplot(fig_sfn)
    st.latex(r'''D(\tau)=\langle |(x(t+\tau)-x(t)|^2\rangle''')
    st.write("Missing values are allowed for by removing nans when computing the mean and cross-products that are used to estimate the structure function.")
    sfn_log = st.checkbox("Log-log plot")
    st.session_state['sfn_log'] = sfn_log

with col4:
   st.header("Power spectrum")
   st.pyplot(fig_psd)
   st.latex(r'''X(k) = x(t)e^{2\pi i kn/N}''')
   st.write("Missing values are allowed for by using the Lomb-Scargle periodogram method.")


# fig_ft = go.Figure()
# fig_ft.add_trace(go.Scatter(x=freqs_positive[1:], y=power_ft, name='FT of complete data'))
# st.plotly_chart(fig_ft)