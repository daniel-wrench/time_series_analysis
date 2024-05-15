# Simulating fBm using Mark's package and calculating some stats of it

from Kea.Kea.simulator import fbm as fbm
from Kea.Kea.statistics.spectra import per_spectra as ps
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as random

from fbm import FBM

# Two ways of creating fBm

N = 10000
D = 1
L = N

# 1. Specifying Hurst parameter (using fbm package)

# for H in [0.25, 0.5, 0.75]:
#     fbm_generator = FBM(n=N, hurst=H, length=L, method="daviesharte")
#     process = fbm_generator.fbm()
#     plt.plot(process, label=f"H = {H}")
# plt.title("fBm with different Hurst parameters")
# plt.legend()
# plt.show()

# 2. Specifying power-law/s (using Mark's Kea package)

dk = 2.0 * np.pi / L
BREAK = 100.0 * dk

random.seed(42)
x_periodic = fbm.create_fbm(
    grid_dims=[4 * N for _ in range(D)],
    phys_dims=[L for _ in range(D)],
    alphas=(-1.0, -5.0 / 3.0),
    gfunc="smooth_pow",
    func_kwargs={"breaks": (BREAK,), "delta": 0.25},
)

x = x_periodic[:N]  # removing periodicities inherent in complete field

#################################

dataset_name = "Multi-fractal\ fBm"

plt.plot(x_periodic, label="Periodic fBM field")
plt.plot(x, label="$\\mathbf{%s}$" % dataset_name)
plt.legend()
plt.show()

# Compute the spectra
for data in [x, x_periodic]:
    # for data in [x]:
    spectrum = ps.modal_spectrum(ar1=data, phys_dims=[L for _ in range(D)])
    plt.plot(spectrum[0][0], spectrum[1])
# plt.plot(np.arange(len(x_subset)), np.arange(len(x_subset)) ** (-5/3), color="black", linestyle="--")
plt.axvline(BREAK, color="black", linestyle="--", label="Break frequency")
plt.semilogx()
plt.semilogy()
plt.xlim(spectrum[0][0].min(), spectrum[0][0].max())
plt.ylim(1e-5, 1e3)
plt.legend()
plt.title("$\\mathbf{%s}$: Power spectral density" % dataset_name)
plt.show()

#################################
# Saving results for later analysis

timestamp = "20160101"  # flipbook plots require timestamp, so using Wind interval start date for now
fbm_field = pd.Series(
    x, name="fbm", index=pd.date_range(timestamp, periods=N, freq="S")
)
fbm_field.to_pickle("data/processed/fbm_field_" + timestamp + ".pkl")
