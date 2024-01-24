# This script reads in a time series file and its corresponding turbulent
# statistics and plots them
# It is a modified version of plotz_tq.py, which was written by Tulasi Parashar

# Note that it is designed to be able to loop over a dictionary of dictionaries
# of turbulent quantities of various time series for a single interval

# TO-DO: Simplify by not inputting a separate 'r' raw time series file, just access
# it from the turbulent statistics dictionary file

import glob
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import TurbAn.Analysis.Simulations.AnalysisFunctions as af
# have commented out import in TurbAn\Analysis\Simulations\AnalysisFunctions.py and TurbAn\Analysis\TimeSeries\Time_Series_Analysis.py

from pylab import rcParams

rcParams["figure.figsize"] = 14, 6


def plotz(tfiles, rfiles, name, plotpoints, timestamp):
    pdf = PdfPages("plots/stats_" + name + "_" + timestamp + ".pdf")
    for flt, flr in zip(tfiles, rfiles):
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
        ax0 = plt.subplot2grid((2, 4), (0, 0), colspan=4)
        ax1 = plt.subplot2grid((2, 4), (1, 0))
        ax2 = plt.subplot2grid((2, 4), (1, 1))
        ax3 = plt.subplot2grid((2, 4), (1, 2))
        ax4 = plt.subplot2grid((2, 4), (1, 3))

        # Plot the raw time series dr
        if len(names) == 3:
            ax0.plot(dr.index[::jumps], dr[names[0]][::jumps], ".-", label=names[0])
            ax0.plot(dr.index[::jumps], dr[names[1]][::jumps], ".-", label=names[1])
            ax0.plot(dr.index[::jumps], dr[names[2]][::jumps], ".-", label=names[2])
        else:
            ax0.plot(dr.index[::jumps], dr[names[0]][::jumps], ".-", label=names[0])
        ax0.set_xlabel("Time")
        ax0.set_title(
            f"Frequency- and lag-domain statistics for interval of ${name}$ beginning {str(dr.index[0])}"
        )
        ax0.legend()

        # Plot the power spectrum
        ax1.plot(
            d[name]["f_periodogram"], d[name]["power_periodogram"], alpha=0.5, c="C0"
        )
        ax1.plot(d[name]["f_periodogram"], d[name]["p_smooth"], label="PSD", c="C0")
        ax1.legend(loc="upper right")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("Frequency (Hz)")

        if not np.isnan(d[name]["qi"]):
            ax1.plot(
                d[name]["xi"],
                d[name]["pi"] * 3,
                c="black",
                ls="--",
                label="Inertial range power-law fit",
            )
            ax1.plot(
                d[name]["xk"],
                d[name]["pk"] * 3,
                c="black",
                ls="--",
                label="Kinetic range power-law fit",
            )
            ax1.text(d[name]["xi"][0] * 5, d[name]["pi"][0], "$f^{q_i}$")
            ax1.text(d[name]["xk"][0] * 2, d[name]["pk"][0], "$f^{q_k}$")

            if not np.isnan(d[name]["break_s"]):
                ax1.text(d[name]["break_s"] / 2, 4e-5, "$f_b$")
                ax1.axvline(d[name]["break_s"], color="black", ls="dotted")

                # Add box with timestamp and values of qi and qk
                textstr = "\n".join(
                    (
                        str(dr.index[0])[:-3]
                        + "-"
                        + "23:59",  # NOTE - this is a hacky way to get the end timestamp
                        r"$q_i=%.2f$" % (d[name]["qi"],),
                        r"$q_k=%.2f$" % (d[name]["qk"],),
                        r"$f_b=%.2f$" % (d[name]["break_s"],),  # ,
                        # r'$f_{{di}}=%.2f$' % (f_di, )
                    )
                )

                props = dict(boxstyle="round", facecolor="gray", alpha=0.2)
                # Place the text box. (x, y) position is in axis coordinates.
                ax1.text(
                    0.05,
                    0.1,
                    textstr,
                    transform=ax1.transAxes,
                    fontsize=8,
                    verticalalignment="bottom",
                    bbox=props,
                )

        # Plot the correlation function cr
        # (already divided by 3 and correlation scale calculated)

        # if len(names) == 3:
        # ax2.plot(d[name]['times'],d[name]['cr']/3.,label='ACF')
        # tcor=d[name]['times'][np.argmin(np.abs(d[name]['cr']/3.-1./np.e))]
        # else:
        ax2.plot(d[name]["times"], d[name]["cr"], label="ACF")
        # tcor=d[name]['times'][np.argmin(np.abs(d[name]['cr']-1./np.e))]
        tcor = d[name]["corr_scale_exp_trick"]
        # Correlation scale is available in the stats file, don't need to calculate here
        ax2.axvline(tcor, linestyle="--", c="black")
        ax2.text(tcor * 1.1, 0.8, r"$t_{{corr}}$ = {0:.02f} s".format(tcor))
        ax2.set_xlabel("$\Delta$ t (s)")
        ax2.legend()

        # Plot the structure function sfn
        ax3.loglog(d[name]["sfn"].iloc[:, 2], label="S$^{(2)}$")
        yyy = d[name]["sfn"].iloc[:, 2].values[58]
        af.pltpwrl(50, yyy * 2.0, xi=1, xf=500, alpha=2.0 / 3, label="2/3", ax=ax3)
        # yy2=d[name]['sfn'].iloc[:,2].values[1]*1.1
        # ax3.text(3,yy2,r'S$^{{(2)}}(dt_{{max}})$={0:.03e}'.format(d[name]['sfn'][0,-1]))
        ax3.set_xlabel("$\Delta$ t (s)")
        ax3.legend()

        # Plot the kurtosis sdk (not calculated component-wise currently)

        # if len(names) == 3:
        #    ax4.semilogx(d[names[0]]['tau'],d[names[0]]['sdk'],label='$\kappa_{'+names[0]+'}$')
        #    ax4.semilogx(d[names[1]]['tau'],d[names[1]]['sdk'],label='$\kappa_{'+names[1]+'}$')
        #    ax4.semilogx(d[names[2]]['tau'],d[names[2]]['sdk'],label='$\kappa_{'+names[2]+'}$')
        # else:
        ax4.semilogx(d[name]["sdk"], label="$\kappa_{" + name + "}$")
        ax4.set_xlabel("$\Delta$ t (s)")
        ax4.legend()

        # Plot the PDF
        # if len(names) == 3:
        #    ax4.semilogy(d[names[0]]['bn1'   ],d[names[0]]['pdf1'   ],label=r'$\Delta = 1 dt$')
        #    ax4.semilogy(d[names[0]]['bn10'  ],d[names[0]]['pdf10'  ],label=r'$\Delta = 10 dt$')
        #    ax4.semilogy(d[names[0]]['bn100' ],d[names[0]]['pdf100' ],label=r'$\Delta = 100 dt$')
        #    ax4.semilogy(d[names[0]]['bn1000'],d[names[0]]['pdf1000'],label=r'$\Delta = 1000 dt$')
        # else:
        #    ax4.semilogy(d[name]['bn1'   ],d[name]['pdf1'   ],label=r'$\Delta = 1 dt$')
        #    ax4.semilogy(d[name]['bn10'  ],d[name]['pdf10'  ],label=r'$\Delta = 10 dt$')
        #    ax4.semilogy(d[name]['bn100' ],d[name]['pdf100' ],label=r'$\Delta = 100 dt$')
        #    ax4.semilogy(d[name]['bn1000'],d[name]['pdf1000'],label=r'$\Delta = 1000 dt$')
        # ax4.legend()
        # ax4.set_xlabel(r'$\Delta$'+names[0]+r'$/\sigma_{\Delta '+names[0]+'}$')
        # ax4.set_ylabel('PDF')

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
tfiles = glob.glob("data/processed/B_turb" + timestamp + ".pkl")
# rfiles=sorted(glob.glob(TRB_DIR+'df_e1_8hr_*.p'))
rfiles = glob.glob("data/processed/B_raw" + timestamp + ".pkl")

# Future vars we might want to calculate stats for
# allnames=['vp','b','np_moment','np','wp_fit','va','b_v','zpp','zmp','sc']

plotz(
    tfiles=glob.glob("data/processed/B_turb_" + timestamp + ".pkl"),
    rfiles=glob.glob("data/processed/B_raw_" + timestamp + ".pkl"),
    name="B",
    plotpoints=400,
    timestamp=timestamp,
)

plotz(
    tfiles=glob.glob("data/processed/Bx_turb_" + timestamp + ".pkl"),
    rfiles=glob.glob("data/processed/Bx_raw_" + timestamp + ".pkl"),
    name="Bx",
    plotpoints=400,
    timestamp=timestamp,
)

plotz(
    tfiles=glob.glob("data/processed/V_turb_" + timestamp + ".pkl"),
    rfiles=glob.glob("data/processed/V_raw_" + timestamp + ".pkl"),
    name="V",
    plotpoints=400,
    timestamp=timestamp,
)

plotz(
    tfiles=glob.glob("data/processed/np_turb_" + timestamp + ".pkl"),
    rfiles=glob.glob("data/processed/np_raw_" + timestamp + ".pkl"),
    name="np",
    plotpoints=400,
    timestamp=timestamp,
)
