# TO-DO

# Add FBm
# Add other gap-handling methods
# Add "loss function" to quantity accuracy of each method (MSE, MAPE)

import pandas as pd
import streamlit as st
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
from scipy import fft
import ts_dashboard_utils as utils 
from astropy import timeseries

st.set_page_config(layout="wide")

st.title("Time Series Statistics and the Effect of Data Gaps")

if 'log' not in st.session_state:
    st.session_state.log = False

########################################################################

@st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
def plot_sfn(missing, log):
    if log == True:  
        ax_sfn.semilogx()
        ax_sfn.semilogy()   

    if missing > 0:
        ax_sfn.plot(sfn_missing, color = "black")
    return fig_sfn 

@st.cache
def simulate_stochastic_processes():
    wn = np.random.normal(size=10000)
    bm = np.cumsum(wn)
    return wn, bm

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
    sfn = utils.calc_sfn(pd.Series(x), [2], 1, 0.9999)

    return freqs_positive, power_ft, acf, sfn["2"].values

@st.cache
def calculate_missing_stats(x_bad, x_bad_ind, gap_method):

    if gap_method == "Linear interpolation":
        x_bad_cleaned = x_bad[pd.notna(x_bad)]
        x_bad = np.interp(np.arange(len(x)), x_bad_ind, x_bad_cleaned)

    # Calculate FT
    N = len(x_bad)
    T = 1/x_freq
    freqs_positive = fft.fftfreq(N, T)[:N//2]
    # ft_x = fft.fft(x, norm = "backward") # Normalisation of 1/n only on the backward term
    # power_ft = 1.0/N * np.abs(ft_x[0:(N//2)])**2

    if gap_method == "Linear interpolation": 
        power_ft = timeseries.LombScargle((np.arange(N)/x_freq), x_bad, normalization="psd").power(freqs_positive[1:])
    # Removing any NA values - leads to clean but un-evenly sampled data for LS method
    else:
        power_ft = timeseries.LombScargle((np.arange(N)/x_freq)[pd.notna(x_bad)], x[pd.notna(x_bad)], normalization="psd").power(freqs_positive[1:])

    # Calculate ACF
    acf = sm.tsa.acf(x_bad, nlags=len(x_bad), missing = "conservative") # also available, "drop", but must reduce nlags to number of non-missing obs
    
    # Calculate SF
    sfn = utils.calc_sfn(pd.Series(x_bad), [2], 1, 0.9999)

    return x_bad, prop_removed, freqs_positive, power_ft, acf, sfn["2"].values

########################################################################

psp_df = pd.read_pickle("psp_mag_scalar_int.pkl")
psp = psp_df.values
psp_freq = 73.24

datasets = ["White noise", "Brownian motion", "Solar wind magnetic field (PSP)"]
formulae = [r'''x(t)\sim N(0,1)''', r'''x(t) = \sum_{i=1}^tW(t),\newline W(t)\sim N(0,1)''', r'''x(t)''']
dataset = st.sidebar.selectbox("Select dataset", datasets, index=0)
missing = st.sidebar.slider("Select amount of data to remove", 0, 100)
removal_type = st.sidebar.radio("Select how to remove data", ["Uniformly", "In chunks"])
gap_method = st.sidebar.radio("Select method for handling data gaps", ["None", "Linear interpolation"])

explanations = {
    "None":"""
    For the autocorrelation and structure function, missing values are allowed for by removing nans when computing the mean and 
    cross-products that are used to estimate the statistic. For the power spectrum, missing values are allowed for
    by using the Lomb-Scargle periodogram method.""",
    "Linear interpolation": """Gaps are interpolated using a straight line between the two data points on either side of the gap."""}
expander = st.sidebar.expander("See explanation of method")
expander.write(explanations[gap_method])

wn, bm = simulate_stochastic_processes()

dset = pd.DataFrame({
    "name": datasets,
    "formula":formulae,
    "data": [wn, bm, psp],
    "freq": [1, 1, psp_freq]})
dset = dset.set_index("name")

#################################
formula = dset.formula[dataset]
x = dset.data[dataset]
x_freq = dset.freq[dataset]

st.write("You have selected ", dataset, " with ", missing, "% removed ", str.lower(removal_type), ", handled using", gap_method)
st.markdown("*Hide the sidebar on the left to expand the plots. They can be further expanded by clicking the arrows in the top right of each plot.*")

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

@st.cache
def remove_data_better(array, proportion, chunks, sigma, removal_type):
    if removal_type == "In chunks":
        x_bad, x_bad_ind, prop_removed = utils.remove_data(array, proportion, chunks, sigma)
    elif removal_type == "Uniformly":
        x_bad, x_bad_ind, prop_removed = utils.remove_data(array, proportion)
    return x_bad, x_bad_ind, prop_removed

if missing > 0:
    x_bad, x_bad_ind, prop_removed = remove_data_better(x, missing/100, chunks = 10, sigma = 0.1, removal_type=removal_type)

    x_missing, prop_removed, freqs_positive_missing, power_ft_missing, acf_missing, sfn_missing = calculate_missing_stats(x_bad, x_bad_ind, gap_method)
    
    ax_data.plot(x_missing, color = "black")
    ax_acf.plot(acf_missing, color = "black")
    ax_psd.plot(freqs_positive_missing[1:], power_ft_missing, color = "black")

col1, col2, col3, col4 = st.columns(4)

with col1:
   st.subheader("Time series")
   st.pyplot(fig_data)
   st.latex(formula)

with col2:
   st.subheader("Autocorrelation")
   st.pyplot(fig_acf)
   st.latex(r'''R(\tau)=\langle x(t)x(t+\tau)\rangle''')

with col3:
    st.subheader("Structure function")
    fig_sfn = plot_sfn(missing, st.session_state.log)
    st.pyplot(fig_sfn)
    st.latex(r'''D(\tau)=\langle |(x(t+\tau)-x(t)|^2\rangle''')
    sfn_log = st.checkbox("Log-log plot", key = "log")

with col4:
    st.subheader("Power spectrum")
    st.pyplot(fig_psd)
    st.latex(r'''P(f) = |X(f)|^2 \newline X(f) = \int_{-\infty}^{\infty} x(t)e^{2\pi i ft}''')