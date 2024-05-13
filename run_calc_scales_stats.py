# TESTING OUT THE FUNCTION

import pandas as pd
from calc_scales_stats import calc_scales_stats
import pickle
from Kea.Kea.simulator import fbm as fbm
import numpy as np

timestamp = "20160101"

mag_int_hr = pd.read_pickle("data/processed/wind/mfi/" + timestamp + ".pkl")

mag_int_hr.head()

# LET'S CREATE AN FBM FIELD WITH THE SAME FREQs AS THIS WIND INTERVAL

N = 10000
D = 1
L = N
dk = 2.0 * np.pi / L
BREAK = 100.0 * dk

# x = fbm.create_fbm(grid_dims=(1000,), phys_dims=(1000,), alphas=(-1,-5/3), breaks=(100,), )
x_periodic = fbm.create_fbm(
    grid_dims=[4 * N for _ in range(D)],
    phys_dims=[L for _ in range(D)],
    alphas=(-1.0, -5.0 / 3.0),
    gfunc="smooth_pow",
    func_kwargs={"breaks": (BREAK,), "delta": 0.25},
)

dataset_name = "Multi-fractal\ fBm"
fbm_field = pd.Series(
    x_periodic[:N], name="fbm", index=pd.date_range("2016-01-01", periods=N, freq="S")
)

# Frequency bounds are taken from Wang et al. (2018, JGR)
mag_params = {
    "spectrum": True,
    "f_min_inertial": 0.005,
    "f_max_inertial": 0.2,
    "f_min_kinetic": 0.5,
    "f_max_kinetic": 1.4,
    "nlags_lr": 2000,
    "nlags_hr": 100,
    "dt_lr": "5S",
    # LR version is used for calculation SFN and ACF; original HR for spectrum and taylor scale
    "tau_min": 10,
    "tau_max": 50,
}

fbm_params = {
    "spectrum": True,
    "f_min_kinetic": 0.02,
    "f_max_kinetic": 0.2,
    "f_min_inertial": 0.001,
    "f_max_inertial": 0.1,
    "nlags_lr": 2000,
    "nlags_hr": 100,
    "dt_lr": "1S",
    # LR version is used for calculation SFN and ACF; original HR for spectrum and taylor scale
    "tau_min": 10,
    "tau_max": 50,
}

proton_int = pd.read_pickle("data/processed/wind/3dp/" + timestamp + ".pkl")

proton_params = {
    "spectrum": False,
    "f_min_inertial": None,
    "f_max_inertial": None,
    "f_min_kinetic": None,
    "f_max_kinetic": None,
    "nlags_lr": 2000,
    "nlags_hr": None,
    "dt_lr": "5S",
    "tau_min": None,
    "tau_max": None,
}


# MAG FIELD: SINGLE COMPONENT
# Get raw time series and turbulent quantities
flr, flt = calc_scales_stats([mag_int_hr.Bx], "Bx", mag_params)

# # Save dictionary for later plotting
flr.to_pickle("data/processed/wind/" + "Bx_raw_" + timestamp + ".pkl")
with open("data/processed/wind/" + "Bx_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)

## FBM FIELD: SINGLE COMPONENT
flr, flt = calc_scales_stats([fbm_field], "fbm", fbm_params)

# # Save dictionary for later plotting
flr.to_pickle("data/processed/wind/" + "fbm_raw" + ".pkl")
with open("data/processed/wind/" + "fbm_turb" + ".pkl", "wb") as file:
    pickle.dump(flt, file)


# # MAG FIELD: FULL VECTOR
flr, flt = calc_scales_stats(
    [mag_int_hr.Bx, mag_int_hr.By, mag_int_hr.Bz], "B", mag_params
)

flr.to_pickle("data/processed/wind/" + "B_raw_" + timestamp + ".pkl")
with open("data/processed/wind/" + "B_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)


# VELOCITY: FULL VECTOR
flr, flt = calc_scales_stats(
    [proton_int.Vx, proton_int.Vy, proton_int.Vz], "V", proton_params
)

flr.to_pickle("data/processed/wind/" + "V_raw_" + timestamp + ".pkl")
with open("data/processed/wind/" + "V_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)


# PROTON DENSITY: SCALAR
flr, flt = calc_scales_stats([proton_int.np], "np", proton_params)

flr.to_pickle("data/processed/wind/" + "np_raw_" + timestamp + ".pkl")
with open("data/processed/wind/" + "np_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)
