import numpy as np
import matplotlib.pyplot as plt 

from os import listdir 
from os.path import isdir, join, isfile
from pathlib import Path

# dn_root = data/TMode-0.001-benchmark/m3_2=0.0/f_ξ=0.0/
# dn1 = dn_root + "mᵪ=0.05011872336272722_I/"
# dn2 = dn_root + "mᵪ=2.6607250597988097_I/"
# dn3 = dn_root + "mᵪ=10.0_I/"

dn_root = "data/TMode-0.001-benchmark/nosugra/f_ξ=0.0/"
dn1 = dn_root + "mᵪ=0.05011872336272722/"
dn2 = dn_root + "mᵪ=0.7079457843841379/"
dn3 = dn_root + "mᵪ=10.0/"

dns = [dn1, dn2, dn3]

for dn in dns:
    fns = [x for x in listdir(dn) if isfile(join(dn, x))]
    
    # read out mᵪ
    m = [x for x in dn.split("/") if "mᵪ=" in x]
    if len(m) == 1:
        m = float(m[0].replace("mᵪ=", "").replace("_R", "").replace("_I", ""))
    
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    for fn in fns:
        k = float(fn.replace("k=", "").replace(".npz", ""))
        path = join(dn, fn)
        data = np.load(path)
        f = data["f"]
        eta = data["eta"]
        err = data["err"]
        # print(len(f), len(eta))
        ax1.plot(eta, f, label=rf"$k={k:.2e}$")
        ax2.plot(eta, err, label=rf"$k={k:.2e}$")

    # ax.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"$\eta$")
    ax1.set_ylabel(r"$|\beta_k|^2$")
    plt.legend()
    
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\eta$")
    ax2.set_ylabel(r"error")
    plt.legend()
   
    out_dn = dn.replace("data", "figs")
    Path(out_dn).mkdir(parents=True, exist_ok=True)
    out_fn = out_dn + "f_evo.pdf"
    plt.savefig(out_fn, bbox_inches="tight")
    plt.close()
