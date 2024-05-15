import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

# gfortran test.f90 -o test.exe
# ./test.exe

# gfortran -shared -o brazil.dll brazil.f90
# gfortran -shared brazil.f90 -o brazil.so

import ctypes

brazil = ctypes.CDLL("./brazil.dll")
fortlib = ctypes.CDLL("./brazil.so")

# f2py -c --fcompiler='gnu95' -m brazil brazil.f90

import fmodpy

myflib = fmodpy.fimport("brazil.f90")


def brzl(df, x, y, z, nx, ny, lims, **kwargs):
    from brazil import brazil

    h, zz = brazil(df[x], df[y], df[z], nx, ny, lims[x] + lims[y] + lims[z])
    zz[np.where(zz == zz.min())] = np.NaN
    zz[np.where(h < 10)] = np.NaN
    dx = (lims[x][1] - lims[x][0]) / 2 / nx
    dy = (lims[y][1] - lims[y][0]) / 2 / ny
    ext = [lims[x][0] + dx, lims[x][1] + dx, lims[y][0] + dy, lims[y][1] + dy]
    fig = plt.imshow(zz.T, origin="lower", aspect="auto", extent=ext, **kwargs)
    plt.colorbar(label=z)
    plt.xlim(lims[x])
    plt.ylim(lims[y])
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    return h, zz, fig


brzl(df, "vsw", "te", "n_e", 100, 100, lims)


def hilohist(df, cond, var, splits):
    dfs = {}
    for i in range(splits):
        lo = i / splits
        hi = (i + 1) / splits
        dfs[i] = df[
            (df[cond] > df[cond].quantile(lo)) & (df[cond] < df[cond].quantile(hi))
        ]
        # dfs[i][var].hist(histtype='step',bins=32,label='{:3.2f} < {} < {:3.2f}'.format(lo,cond,hi))
        sns.kdeplot(
            dfs[i][var], fill=True, label="{:3.2f} < {} < {:3.2f}".format(lo, cond, hi)
        )
    plt.xlabel(var)
    plt.legend()
    plt.show()
    return dfs, plt.gca()


mpl.rcParams["font.family"] = "serif"
# mpl.rcParams["font.family"] = "monospace"
# mpl.rcParams["font.monospace"] = ["FreeMono"]
mpl.rcParams["font.size"] = 16
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.major.size"] = 5.5
mpl.rcParams["xtick.minor.size"] = 3
mpl.rcParams["ytick.major.size"] = 5.5
mpl.rcParams["ytick.minor.size"] = 3

datadir = "C:/Users/spann/Documents/Research/Code repos/reynolds_scales_project/"
data = pd.read_csv(datadir + "wind_dataset_12H.csv")
data.Timestamp = pd.to_datetime(data.Timestamp)
data.set_index("Timestamp", inplace=True)
data.sort_index(inplace=True)
data.rename(
    {
        "V0": "vsw",
        "ne": "n_e",
        "np": "n_p",
        "nalpha": "n_a",
        "Talpha": "Ta",
        "Re_lt": "relt",
        "Re_tb": "retb",
        "Re_di": "redi",
        "lambda_t_raw": "tsr",
        "lambda_t": "ts",
        "lambda_c_e": "lce",
        "lambda_c_fit": "lcf",
        "lambda_c_int": "lci",
    },
    axis=1,
    inplace=True,
)
data["tpte"] = data.Tp / data.Te
data["tatp"] = data.Ta / data.Tp
data["dbob"] = data.db / data.B0
data["rhoi"] = 102 * np.sqrt(data.Tp) / data.B0
data["rhoe"] = 2.38 * np.sqrt(data.Te) / data.B0
data["betae"] = 0.403 * data.n_e * data.Te / data.B0**2
data["betai"] = 0.403 * data.n_e * data.Tp / data.B0**2
data["va"] = 21.8 * data.B0 / np.sqrt(data.n_e)
data["nane"] = data.n_a / data.n_e
data["abs_sc"] = data["sigma_c"].abs()

df = data["2004-05-01":]

for c in df.columns:
    df[c][df[c] > df[c].quantile(0.98)] = np.NaN
    df[c][df[c] < df[c].quantile(0.01)] = np.NaN

lims = {}
lims["n_e"] = [0.6, 20.0]
lims["n_p"] = [0.6, 20.0]
lims["Te"] = [5.0, 100.0]
lims["Tp"] = [3.0, 55.0]
lims["db"] = [1.0, 20.0]
lims["B0"] = [1.6, 30.0]
lims["rhoe"] = [0.5, 5.0]
lims["rhoi"] = [10.0, 200]
lims["de"] = [1.0, 30.0]
lims["dp"] = [10, 220]
lims["betae"] = [0.01, 5.0]
lims["betai"] = [0.01, 3.0]
lims["va"] = [10.0, 200]
lims["ld"] = [0.005, 0.04]
lims["tcf"] = [400, 8000]
lims["tce"] = [400, 8000]
lims["tci"] = [250, 6000]
lims["ttc"] = [1.0, 15.0]
lims["qi"] = [-2.5, -1.4]
lims["qk"] = [-4.5, -1.0]
lims["fb"] = [0.05, 1.0]
lims["vsw"] = [260.0, 800]
lims["p"] = [0.3, 8.0]
lims["relt"] = [3000, 2e6]
lims["redi"] = [5000, 1.2e6]
lims["retb"] = [5000, 6e5]
lims["lce"] = [2e5, 4e6]
lims["lci"] = [2e5, 4e6]
lims["tpte"] = [0.1, 10.0]
lims["ts"] = [500, 7000]
lims["sigma_c"] = [-1, 1]
lims["sigma_r"] = [-1, 1]
lims["cos_a"] = [-1, 1]
lims["ra"] = [0.005, 4.0]
lims["mat"] = [0.02, 2.5]
lims["mst"] = [0.1, 2.5]
lims["SN"] = [2, 220]
lims["n_a"] = [0.003, 0.3]
lims["nane"] = [0.0008, 0.24]
lims["tatp"] = [1.5, 22]
lims["dv"] = [5, 140]
lims["abs_sc"] = [0, 1]
lims["tb"] = [0.15, 4]
lims["ms"] = [3.3, 30]

for k in lims.keys():
    df[k][(df[k] < lims[k][0]) | (df[k] > lims[k][1])] = np.NaN

### Quick and dirty plots

# de vs ne
x = np.linspace(0.0, 17.0, 100)
sns.jointplot(x=df.n_e, y=df.de, kind="kde", fill=True, cmap="Purples")
plt.plot(x, 5.3 / np.sqrt(x), "--", color="orange")
plt.xlim(0.0, 17)
plt.ylim(1.0, 6.3)
plt.xlabel(r"$n_e$ [cm$^{-3}$]")
plt.ylabel(r"$d_e$ [km]")
plt.savefig("de_vs_ne.png", transparent=True)
plt.savefig("de_vs_ne.svg")
# rhoe vs B/sqrt(Te)
x = np.linspace(0.5, 4.0, 100)
sns.jointplot(
    x=df.B0 / np.sqrt(df.te), y=df.rhoe, kind="kde", fill=True, cmap="Purples"
)
plt.plot(x, 2.38 / x, "--", color="orange")
plt.xlim(0.5, 4.0)
plt.ylim(0.5, 4.0)
plt.ylabel(r"$\rho_e$ [km]")
plt.xlabel(r"$B/\sqrt{T_e}$ [nT/eV$^{1/2}$]")
plt.savefig("rhoe_vs_b.png", transparent=True, bbox_inches="tight")
plt.savefig("rhoe_vs_b.svg", bbox_inches="tight")
# Vsw vs tpte
x = np.linspace(100, 800, 100)
sns.jointplot(x=df.vsw, y=df.tpte, kind="kde", fill=True, cmap="Purples")
plt.xlabel(r"$V_{sw}$ [km/s]")
plt.ylabel(r"$T_p/T_e$")
plt.xlim(240, 710)
plt.ylim(0.0, 3.5)
plt.savefig("vsw_vs_tpte.png", transparent=True, bbox_inches="tight")
plt.savefig("vsw_vs_tpte.svg", bbox_inches="tight")
# qi vs qk
sns.jointplot(x=df.qi, y=df.qk, kind="kde", fill=True, cmap="Purples")
plt.xlabel(r"$q_i$")
plt.ylabel(r"$q_k$")
plt.xlim(-1.95, -1.48)
plt.ylim(-4.2, -1.0)
plt.savefig("qi_vs_qk.png", transparent=True, bbox_inches="tight")
plt.savefig("qi_vs_qk.svg", bbox_inches="tight")
# qi vs vsw
sns.jointplot(x=df.vsw, y=df.qi, kind="kde", fill=True, cmap="Purples")
plt.xlim(240, 770)
plt.ylim(-1.97, -1.38)
plt.xlabel(r"$V_{sw}$ [km/s]")
plt.ylabel(r"$q_i$")
plt.savefig("vsw_vs_qi.png", transparent=True, bbox_inches="tight")
plt.savefig("vsw_vs_qi.svg", bbox_inches="tight")
sns.jointplot(x=df.rhoe, y=df.tb, kind="kde", fill=True, cmap="Purples")
plt.xlabel(r"$\rho_e$ [km]")
plt.ylabel(r"$t_b$ [s]")
plt.xlim(0.0, 3.4)
plt.ylim(0.0, 2.0)
plt.savefig("rhoe_vs_tb.png", transparent=True, bbox_inches="tight")
plt.savefig("rhoe_vs_tb.svg", bbox_inches="tight")
sns.jointplot(x=df.tp, y=df.qk, kind="kde", fill=True, cmap="Purples")
plt.ylim(-4.1, -1.0)
plt.xlim(0, 44.1)
plt.xlabel(r"$T_p$ [eV]")
plt.ylabel(r"$q_k$")
plt.savefig("Tp_vs_qk.png", transparent=True, bbox_inches="tight")
plt.savefig("Tp_vs_qk.svg", bbox_inches="tight")
sns.jointplot(x=df.vsw, y=df.te, kind="kde", fill=True, cmap="Purples")
plt.xlim(240, 720)
plt.ylim(5.0, 22.0)
plt.xlabel(r"$V_{sw}$ [km/s]")
plt.ylabel(r"$T_e$ [eV]")
plt.savefig("vsw_vs_te.png", transparent=True, bbox_inches="tight")
plt.savefig("vsw_vs_te.svg", bbox_inches="tight")
sns.jointplot(x=df.vsw, y=df.tp, kind="kde", fill=True, cmap="Purples")
plt.xlim(240, 720)
plt.ylim(0.0, 42.0)
plt.xlabel(r"$V_{sw}$ [km/s]")
plt.ylabel(r"$T_p$ [eV]")
plt.savefig("vsw_vs_tp.png", transparent=True, bbox_inches="tight")
plt.savefig("vsw_vs_tp.svg", bbox_inches="tight")
sns.jointplot(x=df.tp, y=df.ts, kind="kde", fill=True, cmap="Purples")
plt.xlim(0, 44)
plt.ylim(0, 5300)
plt.xlabel(r"$T_p$ [eV]")
plt.ylabel(r"$\lambda_T$ [km]")
plt.savefig("tp_Vs_ts.png", transparent=True, bbox_inches="tight")
plt.savefig("tp_vs_ts.svg", bbox_inches="tight")
sns.jointplot(x=df.tp, y=df.te, kind="kde", fill=True, cmap="Purples")
plt.xlim(0, 44)
plt.ylim(5.0, 22.0)
plt.xlabel(r"$T_p$ [eV]")
plt.ylabel(r"$T_e$ [eV]")
plt.savefig("tp_Vs_te.png", transparent=True, bbox_inches="tight")
plt.savefig("tp_vs_te.svg", bbox_inches="tight")
### MATRIX PLOT
corrlist = [
    "vsw",
    "n_e",
    "Tp",
    "Te",
    "B0",
    "db",
    "dboB0",
    "dv",
    "betae",
    "betai",
    "sigma_r",
    "qi",
    "qk",
    "tb",
    "lce",
    "ts",
    "mst",
    "mat",
    "rhoi",
]
corrnames = [
    r"$V_{sw}$",
    r"$n_e$",
    r"$T_p$",
    r"$T_e$",
    r"$B_0$",
    r"$\delta B$",
    r"$\delta b/B_0$",
    r"$\delta v$",
    r"$\beta_e$",
    r"$\beta_i$",
    r"$\sigma_r$",
    r"$q_i$",
    r"$q_k$",
    r"$\lambda_c$",
    r"$\lambda_T$",
    r"$M_{st}$",
    r"$M_{at}$",
    r"$\rho_p$",
]
rnm = {}
for i, j in zip(corrlist, corrnames):
    rnm[i] = j

sns.set_theme(style="white")
mpl.rcParams["font.size"] = 9
plt.rc("text", usetex=True)
mpl.rcParams["xtick.minor.visible"] = False
mpl.rcParams["ytick.minor.visible"] = False
corrmat = df[rnm.keys()].rename(rnm, axis=1).corr()
mask = np.triu(np.ones_like(corrmat)) - np.eye(corrmat.shape[0])
sns.heatmap(
    corrmat,
    mask=mask,
    annot=True,
    cmap="Spectral",
    vmin=-1,
    vmax=1,
    fmt=".2f",
    cbar_kws={"pad": 0.01},
)
fig = plt.gcf()
fig.set_size_inches(8.5, 7.5)
fig.savefig("corrmat.pdf", bbox_inches="tight")
fig.show()

### LENGTHS
mpl.rcParams["font.size"] = 9
plt.rc("text", usetex=True)
mpl.rcParams["xtick.minor.size"] = 3
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
df["bs"] = df["vsw"] * df["tb"]
lens = ["lce", "ts", "bs", "dp", "rhop", "de", "rhoe", "ld"]
labs = [
    r"$\lambda_C$",
    r"$\lambda_T$",
    r"$\lambda_b$",
    r"$d_p$",
    r"$\rho_p$",
    r"$d_e$",
    r"$\rho_e$",
    r"$\lambda_D$",
]
locs = [550000, 3212, 210, 110, 50, 3, 1.5, 0.01]
for l in range(len(lens)):
    sns.histplot(df[lens[l]], element="step", log_scale=True)
    med = df[lens[l]].median()
    plt.text(locs[l], 680, labs[l])
plt.xlabel("[km]")
plt.ylim(0, 760)
fig = plt.gcf()
fig.set_size_inches(8.5, 1.5)
fig.savefig("sw_lengths.pdf", bbox_inches="tight", pad_inches=0.01)
fig.show()

### SMOOTH VARIABLES WITH SUNSPOTS
vrs = ["SN", "qi", "qk", "relt", "vsw", "dv", "db", "ts", "lce", "mst", "mat"]
lbl = [
    "SSN",
    r"$q_i$",
    r"$q_k$",
    r"$Re_{\lambda_T}$",
    r"$V_{sw}$",
    r"$\delta v$",
    r"$\delta b$",
    r"$\lambda_T$",
    r"$\lambda_C$",
    r"$M_{st}$",
    r"$M_{at}$",
]
temp = {}
# for k in range(len(vrs)): temp[lbl[k]]= df[vrs[k]].interpolate().rolling(700).apply(np.nanmean)
for k in range(len(vrs)):
    temp[lbl[k]] = df[vrs[k]].ewm(span=730).mean()
dfs = pd.DataFrame(temp)["2004-06-01":]
temp = None

mpl.rcParams["font.size"] = 9
plt.rc("text", usetex=True)
dfs.plot(subplots=True, figsize=(3.5, 9), legend=False)

counter = 0
for ax in plt.gcf().axes:
    ax.set_ylabel(lbl[counter])
    counter += 1
plt.savefig("smooth_vars.pdf", bbox_inches="tight", pad_inches=0.01)
plt.show()
sns.set_style("white")
corrmat = dfs.corr(method="spearman")
mask = np.triu(np.ones_like(corrmat)) - np.eye(corrmat.shape[0])
sns.heatmap(
    corrmat,
    annot=True,
    mask=mask,
    cmap="Spectral",
    vmin=-1,
    vmax=1,
    fmt=".2f",
    cbar_kws={"pad": 0.001},
)
fig = plt.gcf()
fig.set_size_inches(4.5, 4.0)
fig.savefig("SSN_corr.pdf", bbox_inches="tight", pad_inches=0.01)
fig.show()
