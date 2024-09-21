import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 

from os import listdir 
from os.path import isdir, join, isfile
from pathlib import Path

cmap = mpl.colormaps['magma']

fn_ode = "data/TMode-0.0035/ode.npz"
data_ode = np.load(fn_ode)
tau = data_ode["tau"]
a = data_ode["a"]
get_a = lambda x: np.interp(x, tau, a)
N = np.log(a)

dn_root = "data/TMode-0.0035/m3_2=0.0/f_ξ=0.0/"
# dn_root = "data/TMode-0.001-benchmark/m3_2=2.0/f_ξ=0.0/"
# dn1 = dn_root + "mᵪ=0.05011872336272722_R/"
# dn2 = dn_root + "mᵪ=0.316227766016838_R/"
# dn3 = dn_root + "mᵪ=1.9952623149688795_R/"
dn1 = dn_root + "mᵪ=2.0_R/"
"""

dn_root = "data/TMode-0.0035/nosugra/f_ξ=0.0/"
dn1 = dn_root + "mᵪ=0.05011872336272722/"
dn2 = dn_root + "mᵪ=1.5848931924611134/"
dn3 = dn_root + "mᵪ=5.011872336272722/"
"""

# dns = [dn1, dn2, dn3]
dns = [dn1]

for dn in dns:
    fns = [x for x in listdir(dn) if isfile(join(dn, x))]
    
    # read out mᵪ
    m = [x for x in dn.split("/") if "mᵪ=" in x]
    if len(m) == 1:
        m = float(m[0].replace("mᵪ=", "").replace("_R", "").replace("_I", ""))
    
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)

    fig2 = plt.figure()
    ax = fig2.add_subplot()

    k_array = [float(fn.replace("k=", "").replace(".npz", "")) for fn in fns]
    k_max, k_min = np.amax(k_array), np.amin(k_array)

    for fn in fns:
        k = float(fn.replace("k=", "").replace(".npz", ""))
        color = cmap( (np.log(k_max) - np.log(k))/(np.log(k_max) - np.log(k_min)) )
        path = join(dn, fn)
        try:
            data = np.load(path)
        except Exception:
            pass
        finally:
            f = data["f"]
            eta = data["eta"]
            err = data["err"]
            chi = data["chi"]
            # print(len(f), len(eta))

            ax.plot(np.interp(eta[::10], tau, N), np.real(f[::10]), label=rf"$k={k:.2e}$", color=color)
            ax1.plot(np.interp(eta[::10], tau, N), np.real(f[::10]), label=rf"$k={k:.2e}$", color=color)
            ax2.plot(np.interp(eta[::10], tau, N), np.abs(chi[::10])**2 * get_a(eta[::10]), label=rf"$k={k:.2e}$", color=color)
            ax3.plot(np.interp(eta[::10], tau, N), err[::10], label=rf"$k={k:.2e}$", color=color)

    # ax.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$N$")
    ax1.set_ylabel(r"$|\beta_k|^2$")
    ax1.set_ylim((1e-14, 1e0))
    # plt.legend(loc=1)
 
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$a |\chi_k|^2$")
    # ax2.set_ylim((1e4, 2e4))
   
    ax3.set_yscale("log")
    ax3.set_xlabel(r"$N$")
    ax3.set_ylabel(r"error")
    # plt.legend(loc=1)
    handles, labels = plt.gca().get_legend_handles_labels()
    k_array = [float(x.replace("$", "").replace("k=", "")) for x in labels]
    k_array, handles, labels = map(list, zip(*sorted(zip(k_array, handles, labels))))
    # plt.legend(handles, labels, loc=4)
   
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    out_fn = out_dn + "f_evo.pdf"
    fig.savefig(out_fn, bbox_inches="tight")
    plt.close(1)

    ax.set_yscale("log")
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$|\beta_k|^2$")
    ax.set_xlim((0, 4))
    ax.set_ylim((1e-6, 1e-2))
    fig2.savefig(out_dn + "beta_evo.pdf", bbox_inches="tight")
    plt.close(2)
