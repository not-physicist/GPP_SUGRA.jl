from matplotlib.transforms import Bbox
import numpy as np
import matplotlib as mpl
# mpl.use("Agg")
import matplotlib.pyplot as plt 
from os import listdir 
from os.path import isdir, isfile, join, exists
from pathlib import Path

ode_fn = "data/TMode-0.0035/ode.npz"
data = np.load(ode_fn)
# assume aₑ == 1
mᵩ = data["m_phi"]
k_array = np.array([1e-3, 1e-2, 1e-1]) * mᵩ
# print(k_array)

dn = "data/TMode-0.0035/m3_2=0.00/m_eff/"
fns = [x for x in listdir(dn)]
# print(fns)

fig, ax = plt.subplots()
colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple", "tab:brown"]

for i, fn in enumerate(fns):
    # print(i, fn)
    data = np.load(join(dn, fn))
    a = data["a"]
    N = np.log(a)
    m2_R = data["m2_R"]
    m2_I = data["m2_I"]
    tau = data["tau"]
    
    label = "$" + r"m_{\rm eff}, " + fn.replace(".npz", "").replace("m_chi=", "") + "$"
    ax.plot(N, np.sqrt(m2_R)/a, color=colors[i], label=label)
    ax.plot(N, np.sqrt(m2_I)/a, color=colors[i], ls='--')

for k in k_array:
    ax.plot(N, k/a, label=f"$k/a, k={k/mᵩ:.1e}$")

ax.set_xlabel("$N$")
# ax.set_ylabel(r"$m_{\rm eff}/a$")
ax.set_yscale("log")
# ax.set_xlim((2, 5))
# ax.set_xlim((2e5, tau[-1]/2))
# ax.set_ylim((1.2335e-5, 1.2345e-5))
ax.legend(loc=1)

out_dn = dn.replace("data", "figs").replace("m_eff/", "")
Path(out_dn).mkdir(parents=True, exist_ok=True)
out_fn = out_dn + "m_eff.pdf"
plt.savefig(out_fn, bbox_inches="tight")
plt.close()
