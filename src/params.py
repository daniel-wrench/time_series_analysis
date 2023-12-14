# This file specifies the filepaths, variable names, thresholds and interval sizes
# for the initial data processing

# See here for brief description of all Wind datasets:
# https://cdaweb.gsfc.nasa.gov/misc/NotesW.html

# See also accessing Wind data with HelioPy:
# https://buildmedia.readthedocs.org/media/pdf/heliopy/0.6.0/heliopy.pdf 

timestamp = "Epoch"
int_size = "12H"
start_date = "19950101"
end_date = "20081231"

# Not using OMNI currently
omni_path = "omni/omni_cdaweb/hro2_1min/"
vsw = "flow_speed"
p = "Pressure"
Bomni = "F"
omni_thresh = {
    "flow_speed": [0, 1000],
    "Pressure": [0, 200],
    "F": [0, 50]
}

electron_path = "wind/3dp/3dp_elm2/"
ne = "DENSITY"
Te = "AVGTEMP"

electron_thresh = {
    "DENSITY": [0, 200],
    "AVGTEMP": [0, 1000]
}


# Metadata: 
# https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_pm_3dp_00000000_v01.skt
# https://hpde.io/NASA/NumericalData/Wind/3DP/PM/PT03S

proton_path = "wind/3dp/3dp_pm/"
np = "P_DENS" # density in #/cm3
nalpha = "A_DENS" # alpha particle density in #/cm3
Talpha = "A_TEMP"
Tp = "P_TEMP" # temperature in eV
V_vec = "P_VELS" # velocity in km/s
Vx = "P_VELS_0"
Vy = "P_VELS_1"
Vz = "P_VELS_2"
proton_thresh = {
    "P_DENS": [0, 1000],
    "P_TEMP": [0, 500],
    "A_DENS": [0, 1000],
    "A_TEMP": [0, 500]
}

mag_path = "wind/mfi/mfi_h2/"
Bwind = "BF1" # not using currently
Bwind_vec = "BGSE"
Bx = "BGSE_0"
By = "BGSE_1"
Bz = "BGSE_2"
mag_thresh = None

# Parameters for estimating numerical variables
dt_lr = "5S"
nlags_lr = 2000
dt_hr = "0.092S"
dt_protons = "3S"
nlags_hr = 100
tau_min = 10
tau_max = 50

# Frequency bounds are taken from Wang et al. (2018, JGR)
f_min_inertial = 0.005
f_max_inertial = 0.2
f_min_kinetic = 0.5
f_max_kinetic = 1.4
