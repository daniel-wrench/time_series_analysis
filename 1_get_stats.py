# TESTING OUT THE FUNCTION

import pandas as pd
from src.calc_scales_stats import calc_scales_stats
import pickle

timestamp = "20160101"

wind_raw = pd.read_pickle("data/processed/wind/mfi/" + timestamp + ".pkl")
wind_resampled = wind_raw.resample("0.1S").mean()
wind = wind_resampled[10000:20000]

psp_raw = pd.read_pickle("data/processed/psp/psp_fld_l2_mag_rtn_201811.pkl")
psp_resampled = psp_raw.resample("0.1S").mean()
psp = psp_resampled[:10000]

fbm = pd.read_pickle("data/processed/fbm_field_" + timestamp + ".pkl")

# Frequency bounds are taken from Wang et al. (2018, JGR)
wind_params = {
    "spectrum": True,
    "f_min_inertial": None,  # 0.005,
    "f_max_inertial": None,  # 0.2,
    "f_min_kinetic": None,  # 0.5,
    "f_max_kinetic": None,  # 1.4,
    "nlags_lr": 2500,  # Should be the same as max_lag * int_length for SF calculation
    "nlags_hr": 100,
    "dt_lr": "0.1S",
    # LR version is used for calculation SFN and ACF; original HR for spectrum and taylor scale
    "tau_min": 10,
    "tau_max": 50,
}

fbm_params = {
    "spectrum": True,
    "f_min_kinetic": None,  # 0.02,
    "f_max_kinetic": None,  # 0.2,
    "f_min_inertial": None,  # 0.001,
    "f_max_inertial": None,  # 0.1,
    "nlags_lr": 2500,
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
flr, flt = calc_scales_stats([wind.Bx], "Bx", wind_params)

# # Save dictionary for later plotting
flr.to_pickle("data/processed/wind/" + "Bx_raw_" + timestamp + ".pkl")

with open("data/processed/wind/" + "Bx_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)

flr, flt = calc_scales_stats([psp.B_R], "B_R", wind_params)

# # Save dictionary for later plotting
flr.to_pickle("data/processed/psp/" + "B_R_raw_" + timestamp + ".pkl")
with open("data/processed/psp/" + "B_R_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)

## FBM FIELD: SINGLE COMPONENT
flr, flt = calc_scales_stats([fbm], "fbm", fbm_params)

# # Save dictionary for later plotting
flr.to_pickle("data/processed/" + "fbm_raw" + ".pkl")
with open("data/processed/" + "fbm_turb" + ".pkl", "wb") as file:
    pickle.dump(flt, file)


# # MAG FIELD: FULL VECTOR
flr, flt = calc_scales_stats([wind.Bx, wind.By, wind.Bz], "B", wind_params)

flr.to_pickle("data/processed/wind/" + "B_raw_" + timestamp + ".pkl")
with open("data/processed/wind/" + "B_turb_" + timestamp + ".pkl", "wb") as file:
    pickle.dump(flt, file)


# VELOCITY: FULL VECTOR
# flr, flt = calc_scales_stats(
#     [proton_int.Vx, proton_int.Vy, proton_int.Vz], "V", proton_params
# )

# flr.to_pickle("data/processed/wind/" + "V_raw_" + timestamp + ".pkl")
# with open("data/processed/wind/" + "V_turb_" + timestamp + ".pkl", "wb") as file:
#     pickle.dump(flt, file)


# # PROTON DENSITY: SCALAR
# flr, flt = calc_scales_stats([proton_int.np], "np", proton_params)

# flr.to_pickle("data/processed/wind/" + "np_raw_" + timestamp + ".pkl")
# with open("data/processed/wind/" + "np_turb_" + timestamp + ".pkl", "wb") as file:
#     pickle.dump(flt, file)

print("Done!")
