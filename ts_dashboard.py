# TO-DO

# Add data removal feature (test stats on this outside of dashboard)


import pandas as pd
import seaborn as sb
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import signal, fft

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

@st.cache
def calculate_stats(missing, removal_type):
    # Calculate FT
    N = len(x)
    T = 1/x_freq
    freqs_positive = fft.fftfreq(N, T)[:N//2]
    ft_x = fft.fft(x, norm = "backward") # Normalisation of 1/n only on the backward term
    power_ft = 1.0/N * np.abs(ft_x[0:(N//2)])**2

    # Calculate ACF
    acf = sm.tsa.acf(x, nlags=len(x))

    sfn = calc_sfn(pd.Series(x), [2], 1, 0.9999)

    return freqs_positive, power_ft, acf, sfn

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

freqs_positive, power_ft, acf, sfn = calculate_stats(missing, removal_type)

########

fig_data, ax_data = plt.subplots()
ax_data = sb.lineplot(data = x).set(title="Time series")
st.pyplot(fig_data)

fig_acf, ax_acf = plt.subplots()
ax_acf = sb.lineplot(data = acf).set(title="Autocorrelation")
st.pyplot(fig_acf)

fig_sf, ax_sf = plt.subplots()
ax_sf = sb.lineplot(data = sfn["2"]).set(title="Second-order structure function")
st.pyplot(fig_sf)

#################################
#if x != y:
fig_ft = go.Figure()
fig_ft.add_trace(go.Scatter(x=freqs_positive, y=power_ft, name='FT'))
fig_ft.add_trace(go.Scatter(x=freqs_positive[1:], y = 1000*freqs_positive[1:]**(-5/3), name = "-5/3 power law"))
fig_ft.update_xaxes(type="log", title_text="Frequency (Hz)")
fig_ft.update_yaxes(type="log", title_text='PSD [nT^2/Hz]')
fig_ft.update_layout(
    #legend_title_text="Method", 
    #width=700, 
    #height=450, 
    title_text = "Power spectrum")

st.plotly_chart(fig_ft)
#else:
    # fig = sb.relplot(data=dset_select, x=x, y=y, hue="Species")