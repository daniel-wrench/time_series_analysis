# This script reads in a time series file and its corresponding turbulent
# statistics and plots them
# It is a modified version of plotz_tq.py, which was written by Tulasi Parashar

# Note that it is designed to be able to loop over a dictionary of dictionaries
# of turbulent quantities of various time series for a single interval

# TO-DO: Simplify by not inputting a separate 'r' raw time series file, just access
# it from the turbulent statistics dictionary file

import glob
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import TurbAn.Analysis.Simulations.AnalysisFunctions as af
# have commented out import in TurbAn\Analysis\Simulations\AnalysisFunctions.py and TurbAn\Analysis\TimeSeries\Time_Series_Analysis.py


def plotz(tfiles, rfiles, name, var_name, unit, spacecraft, plotpoints, timestamp):
    pdf = PdfPages("plots/stats_" + spacecraft + name + "_" + timestamp + ".pdf")
    lags_to_examine = [5, 50, 500]
    for flt, flr in zip(tfiles, rfiles):
        if not os.path.exists(flt) or not os.path.exists(flr):
            print("File path does not exist:", flt, flr)
            continue
        print(tfiles.index(flt), flt)
        d = pickle.load(open(flt, "rb"))
        dr = pd.read_pickle(flr)
        jumps = len(dr) // plotpoints

        # If a vector quantity, create list of component names
        if name in ["B", "V", "zpp", "zmp", "b_v"]:
            names = [name + "x", name + "y", name + "z"]
        else:
            names = [name]
        #     try:
        plt.clf()
        plt.figure(figsize=(14, 6))
        ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=3)
        ax1 = plt.subplot2grid((2, 4), (0, 3), colspan=1)
        ax2 = plt.subplot2grid((2, 4), (1, 0))
        ax3 = plt.subplot2grid((2, 4), (1, 1))
        ax4 = plt.subplot2grid((2, 4), (1, 2))
        ax5 = plt.subplot2grid((2, 4), (1, 3))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)
        for i, var in enumerate(names):
            # Plot the raw time series dr
            ax0.plot(dr.index[::jumps], dr[var][::jumps], ".-", label=var)
            # Plot histogram of the raw time series
            ax1.hist(dr[var], bins=50, alpha=0.5, label=var)
            ax1.set_xlabel(unit)
            # Overlay an "equivalent Gaussian" (same mean and variance)
            x_vals = np.linspace(dr[var].min(), dr[var].max(), 100)
            y_vals = (
                1
                / np.sqrt(2 * np.pi * dr[var].var())  # type: ignore
                * np.exp(-0.5 * (x_vals - dr[var].mean()) ** 2 / dr[var].var())  # type: ignore
            )
            ax1.plot(
                x_vals,
                y_vals * len(dr[var]) * 0.1,
                color="black",
                linestyle=":",
                # label=r"$N(\mu, \sigma^2)$",
            )

            # Remove y-axis tickmarks from the histogram
            ax1.yaxis.set_ticks_position("none")
            ax1.yaxis.set_ticklabels([])

            # Annotate the values of the first four moments of the distributions
            if len(names) == 1:
                ax1.annotate(
                    rf"$\mu$={dr[var].mean():.2f}",
                    (0.05, 0.85),
                    xycoords="axes fraction",
                    fontsize=9,
                )
                ax1.annotate(
                    rf"$\sigma^2$={dr[var].var():.2f}",
                    (0.05 + 0.2 * i, 0.75),
                    xycoords="axes fraction",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        edgecolor="white",
                        alpha=0.5,
                        pad=0.2,
                    ),
                )
                ax1.annotate(
                    f"Skew={dr[var].skew():.2f}",
                    (0.05, 0.65),
                    xycoords="axes fraction",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        edgecolor="white",
                        alpha=0.5,
                        pad=0.2,
                    ),
                )
                ax1.annotate(
                    f"Kurt={(dr[var].kurtosis()+3):.2f}",  # Add 3 to get actual kurtosis
                    (0.05, 0.55),
                    xycoords="axes fraction",
                    fontsize=9,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        edgecolor="white",
                        alpha=0.5,
                        pad=0.2,
                    ),
                )

        # ax0.set_xlabel("Time")
        ax0.set_ylabel(unit)
        ax0.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax0.xaxis.get_major_locator())
        )
        ax0.set_title(
            "Time series statistics of $\\mathbf{%s}$" % var_name,
            #  beginning {str(dr.index[0])}
            fontsize=14,
        )
        ax1.set_title("Distribution and moments")
        ax2.set_title("Power spectrum")
        ax3.set_title("Autocorrelation function")
        ax4.set_title("Structure function")
        ax5.set_title("Kurtosis")
        ax0.legend()

        # Plot the power spectrum
        ax2.plot(
            d[name]["f_periodogram"], d[name]["power_periodogram"], alpha=0.5, c="C0"
        )
        ax2.plot(d[name]["f_periodogram"], d[name]["p_smooth"], label="PSD", c="C0")
        # ax2.legend(loc="upper right")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_ylim(
            np.percentile(d[name]["power_periodogram"], 0.01),
            10 * np.max(d[name]["power_periodogram"]),
        )
        af.pltpwrl(
            1e-3,
            d[name]["p_smooth"][5] * 80.0,
            xi=1e-2,
            xf=1,
            alpha=-5.0 / 3,
            label="-5/3",
            ax=ax2,
            color="purple",
            ls="--",
        )
        ax2.set_xlabel("Frequency (Hz)")

        if not np.isnan(d[name]["qi"]):
            ax2.plot(
                d[name]["xi"],
                d[name]["pi"] * 3,
                c="black",
                ls="--",
                label="Inertial range power-law fit",
            )
            ax2.plot(
                d[name]["xk"],
                d[name]["pk"] * 3,
                c="black",
                ls="--",
                label="Kinetic range power-law fit",
            )
            ax2.text(d[name]["xi"][0] * 5, d[name]["pi"][0], "$f^{q_i}$")
            ax2.text(d[name]["xk"][0] * 2, d[name]["pk"][0], "$f^{q_k}$")

            ax2.text(d[name]["fb"] / 2, 4e-5, "$f_b$")
            ax2.axvline(d[name]["fb"], color="black", ls="dotted")

            # Add box with timestamp and values of qi and qk
            textstr = "\n".join(
                (
                    # str(dr.index[0])[:-3]
                    # + "-"
                    # + "23:59",  # NOTE - this is a hacky way to get the end timestamp
                    r"$q_i=%.2f$" % (d[name]["qi"],),
                    r"$q_k=%.2f$" % (d[name]["qk"],),
                    r"$f_b=%.2f$" % (d[name]["fb"],),  # ,
                    # r'$f_{{di}}=%.2f$' % (f_di, )
                )
            )

            props = dict(boxstyle="round", facecolor="gray", alpha=0.2)
            # Place the text box. (x, y) position is in axis coordinates.
            ax2.text(
                0.05,
                0.1,
                textstr,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment="bottom",
                bbox=props,
            )

        # Plot the correlation function cr
        # (already divided by 3 and correlation scale calculated)

        # if len(names) == 3:
        # ax3.plot(d[name]['times'],d[name]['cr']/3.,label='ACF')
        # tcor=d[name]['times'][np.argmin(np.abs(d[name]['cr']/3.-1./np.e))]
        # else:
        ax3.plot(d[name]["times"][: len(d[name]["cr"])], d[name]["cr"], label="ACF")
        # tcor=d[name]['times'][np.argmin(np.abs(d[name]['cr']-1./np.e))]
        tcor = d[name]["tce"]
        # Correlation scale is available in the stats file, don't need to calculate here
        ax3.axvline(tcor, linestyle="--", c="black")
        ax3.text(tcor * 1.2, 0.8, r"$t_{{corr}}$")  # = {0:.2f} s".format(tcor))
        ax3.set_xlabel("$\Delta$t (s)")
        # ax3.legend()

        # Plot the structure function sfn
        ax4.plot(
            d[name]["times"][: len(d[name]["sfn"])],
            d[name]["sfn"].loc[:, "2"],
            label="S$^{(2)}$",
        )
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        yyy = d[name]["sfn"].loc[:, "2"].values[58]
        af.pltpwrl(
            50,
            yyy * 2.0,
            xi=1,
            xf=500,
            alpha=2.0 / 3,
            label="2/3",
            ax=ax4,
            color="purple",
            ls="--",
        )
        # yy2=d[name]['sfn'].iloc[:,2].values[1]*1.1
        # ax4.text(3,yy2,r'S$^{{(2)}}(dt_{{max}})$={0:.03e}'.format(d[name]['sfn'][0,-1]))
        ax4.set_xlabel("$\Delta$t (s)")
        # ax4.legend()

        # Plot the kurtosis sdk (not calculated component-wise currently)

        # if len(names) == 3:
        #    ax5.semilogx(d[names[0]]['tau'],d[names[0]]['sdk'],label='$\kappa_{'+names[0]+'}$')
        #    ax5.semilogx(d[names[1]]['tau'],d[names[1]]['sdk'],label='$\kappa_{'+names[1]+'}$')
        #    ax5.semilogx(d[names[2]]['tau'],d[names[2]]['sdk'],label='$\kappa_{'+names[2]+'}$')
        # else:
        ax5.plot(
            d[name]["times"][: len(d[name]["sdk"])],
            d[name]["sdk"],
            label="$\kappa_{" + name + "}$",
        )
        ax5.set_xscale("log")
        # Add a horizontal linae at k =3
        ax5.axhline(3, color="black", linestyle="--")
        ax5.set_xlabel("$\Delta$t (s)")
        # ax5.legend()

        # Plot the PDF
        # if len(names) == 3:
        #    ax5.semilogy(d[names[0]]['bn1'   ],d[names[0]]['pdf1'   ],label=r'$\Delta = 1 dt$')
        #    ax5.semilogy(d[names[0]]['bn10'  ],d[names[0]]['pdf10'  ],label=r'$\Delta = 10 dt$')
        #    ax5.semilogy(d[names[0]]['bn100' ],d[names[0]]['pdf100' ],label=r'$\Delta = 100 dt$')
        #    ax5.semilogy(d[names[0]]['bn1000'],d[names[0]]['pdf1000'],label=r'$\Delta = 1000 dt$')
        # else:
        #    ax5.semilogy(d[name]['bn1'   ],d[name]['pdf1'   ],label=r'$\Delta = 1 dt$')
        #    ax5.semilogy(d[name]['bn10'  ],d[name]['pdf10'  ],label=r'$\Delta = 10 dt$')
        #    ax5.semilogy(d[name]['bn100' ],d[name]['pdf100' ],label=r'$\Delta = 100 dt$')
        #    ax5.semilogy(d[name]['bn1000'],d[name]['pdf1000'],label=r'$\Delta = 1000 dt$')
        # ax5.legend()
        # ax5.set_xlabel(r'$\Delta$'+names[0]+r'$/\sigma_{\Delta '+names[0]+'}$')
        # ax5.set_ylabel('PDF')
        for lag in lags_to_examine:
            for ax in [ax3, ax4, ax5]:
                # Multiplying by re-sampled freq to get correct placement
                ax.axvline(
                    lag * d[name]["times"][1],
                    color="red",
                    linestyle="--",
                    alpha=0.2,
                    lw=0.5,
                )
        # plt.tight_layout()
        pdf.savefig(bbox_inches="tight")
        plt.close()
        pdf.close()


timestamp = "20160101"
plotpoints = 400
# basedir=os.path.abspath(input('Base dir? '))
# RAW_DIR=basedir + '/data_processed/'
# TRB_DIR=basedir + '/data_processed/'
# tfiles=sorted(glob.glob(TRB_DIR+'e1_8hr_*.p'))
tfiles = glob.glob("data/processed/wind/B_turb" + timestamp + ".pkl")
# rfiles=sorted(glob.glob(TRB_DIR+'df_e1_8hr_*.p'))
rfiles = glob.glob("data/processed/wind/B_raw" + timestamp + ".pkl")

# Future vars we might want to calculate stats for
# allnames=['vp','b','np_moment','np','wp_fit','va','b_v','zpp','zmp','sc']

plotz(
    tfiles=glob.glob("data/processed/wind/B_turb_" + timestamp + ".pkl"),
    rfiles=glob.glob("data/processed/wind/B_raw_" + timestamp + ".pkl"),
    name="B",
    var_name="solar\ wind\ magnetic\ field\ from\ Wind",
    unit="B (nT)",
    spacecraft="wind",
    plotpoints=400,
    timestamp=timestamp,
)

plotz(
    tfiles=glob.glob("data/processed/wind/Bx_turb_" + timestamp + ".pkl"),
    rfiles=glob.glob("data/processed/wind/Bx_raw_" + timestamp + ".pkl"),
    name="Bx",
    var_name="solar\ wind\ magnetic\ field\ from\ Wind",
    unit="Bx (nT)",
    spacecraft="wind",
    plotpoints=400,
    timestamp=timestamp,
)

plotz(
    tfiles=glob.glob("data/processed/psp/B_R_turb_" + timestamp + ".pkl"),
    rfiles=glob.glob("data/processed/psp/B_R_raw_" + timestamp + ".pkl"),
    name="B_R",
    var_name="solar\ wind\ magnetic\ field\ from\ PSP",
    unit="B_R (nT)",
    spacecraft="psp",
    plotpoints=400,
    timestamp=timestamp,
)

plotz(
    tfiles=glob.glob("data/processed/wind/fbm_turb" + ".pkl"),
    rfiles=glob.glob("data/processed/wind/fbm_raw" + ".pkl"),
    name="fbm",
    var_name="multi-fractal\ fractional\ Brownian\ motion",
    unit="",
    spacecraft="",
    plotpoints=400,
    timestamp=timestamp,
)

# plotz(
#     tfiles=glob.glob("data/processed/wind/V_turb_" + timestamp + ".pkl"),
#     rfiles=glob.glob("data/processed/wind/V_raw_" + timestamp + ".pkl"),
#     name="V",
#     plotpoints=400,
#     timestamp=timestamp,
# )

# plotz(
#     tfiles=glob.glob("data/processed/wind/np_turb_" + timestamp + ".pkl"),
#     rfiles=glob.glob("data/processed/wind/np_raw_" + timestamp + ".pkl"),
#     name="np",
#     plotpoints=400,
#     timestamp=timestamp,
# )
