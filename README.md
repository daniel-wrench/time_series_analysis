# Time Series analysis

## `1_get_stats.py`
Run flipbook code to get all spectral stats for a given time series

## `2_plot_ts_stats.py` 
Plot those stats

## `3_plot_lag_stats.py`
Plot the 3 estimators for a given interval, then plot the tau-scattergrams and PDFs for 3 specific lags.

## `get_gapped_sf.py` (Rāpoi job)
Gap and calculate SFs for a lot of time series

## `plot_results.py` (Rāpoi local job)
Calculate heatmaps, apply correction factors to test set, and produce plots of these results 

(Add Wind in here)

## `sf_funcs_demo.ipynb` 
Demonstration on small datset of entire pipeline

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