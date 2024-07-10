# Time Series analysis

## `1_get_stats.py`
Run flipbook code to get all spectral stats for a given time series

## `2_plot_ts_stats.py` 
Plot those stats

## `3_plot_lag_stats.py`
Plot the 3 SF estimators for a given interval, then plot the tau-scattergrams and PDFs for 3 specific lags.

## `get_gapped_sf.py` (Rāpoi job)
Gap and calculate SFs for a lot of time series. Needs to run on PSP data (training and test set) and Wind data (external test set: much smaller)
*Currently two documents*

## `plot_results.py` (Rāpoi local job)
Calculate heatmaps, apply correction factors to test set, and produce plots of these results.
*Currently two documents*

Apply correction factor `compute_scaling()` on both test sets

(Add Wind in here)

## `sf_funcs_demo.ipynb` 
Demonstration on small dataset of pre-processing of data and gathering results (not the gapping part)

## WIND UPDATE
- RE-RUNNING LOCAL CODES ON EXISTING PSP CODE
-- get_gapped: good
-- plot_results: good (pending updates to plotting results, 3d heatmaps, overlaying traces, etc.)
- Get Wind pkl from Reynolds work
-- HR version. 2016-01-02 had long corr length so difficult to standardise. 01-04 is better, but still just gives us one interval of 8 lengths. Therefore, we need a week's worth to have a few to test the correction on. To get this, compare Re processing to PSP processing and update accordingly: because Re work outputs each file separately, rather than merging them together, which this work does do. *Run interactively in a new script*
- Read into get_gapped_sf, run interactively to check standardisation
- Run remainder of code

I will need to run whole pipeline on Wind data, but it will be much smaller, and it will use the PSP correction factor. In plot_results, we will not run create_heatmap_lookup() for Wind, and will instead use the PSP lookup table as an argument to compute_scaling()

## TO-DO
- Add PDF, mean, rms to calc_scales_stats?
- Add ability to select exactly which outputs are calculated?
- Add option to plot fluctuation time series instead of actual values
- Change to not read in separate time series file when plotting?
- Run analysis code on z+ and z- and calculate epsilons too

- 7min for 5 files
- 30min for 20 files
- 85min for 50 files = 50x1040 inputs
- 130min for 50 files, 2 bin sizes = 50x1040 inputs
- 7hours for 79 files, 3 bin sizes = 100x1000 inputs

[Streamlit dashboard](https://daniel-wrench-time-series-analysis-ts-dashboard-4a8iw8.streamlitapp.com/)

- time_series_stats.ipynb: demonstration of key time series statistics on solar wind dataset: ACF, SFn, PSD etc.
- stochastic_processes.ipynb: simulations of theoretical stochastic processes: white noise, RW, O-U etc.
- arima_forecasting.ipynb: demonstration of workflow for fitting an ARIMA model, including how to check for stationarity