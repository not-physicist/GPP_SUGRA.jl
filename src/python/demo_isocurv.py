import numpy as np
import matplotlib.pyplot as plt 

color = ["tab:blue", "tab:orange"]

# fn = "data/TMode-0.001-benchmark/m3_2=0.0/f_ξ=0.0/mᵪ=0.05011872336272722_I.npz"
fn = "data/TMode-0.001-benchmark/nosugra/f_ξ=0.0/mᵪ=0.05011872336272722.npz"

data = np.load(fn)
k = data["k"]
Δ2 = data["Delta2"]
Δ2_k3 = data["Delta2_k3"]
plt.plot(k, Δ2, c=color[0])
plt.plot(k, Δ2_k3, ls="--", c=color[0])
plt.xscale("log")
plt.yscale("log")

# fn = "data/TMode-0.001-benchmark/m3_2=0.0/f_ξ=0.0/mᵪ=2.6607250597988097_I.npz"
fn = "data/TMode-0.001-benchmark/nosugra/f_ξ=0.0/mᵪ=2.6607250597988097.npz"
data = np.load(fn)
k = data["k"]
Δ2 = data["Delta2"]
Δ2_k3 = data["Delta2_k3"]
plt.plot(k, Δ2, c=color[1])
plt.plot(k, Δ2_k3, ls="--", c=color[1])

plt.plot(k, k**3, color="k", ls="--")
plt.savefig("isocurv.pdf")
