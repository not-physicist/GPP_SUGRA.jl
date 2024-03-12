import numpy as np 
import matplotlib.pyplot as plt 

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

fig, ax = plt.subplots(figsize=(10, 1.5))

t_list = [r"$t_{-\infty}$", "$t_{e}$", "$t_{p}$", "$t_{rh}$", "$t_0$"]
n_t = len(t_list)

era_list = ["dS", "MD", "MD", r"RD, MD, $\Lambda$D"]
eta_n = len(era_list)

mult = 3

ax.arrow(-0.5, 0, mult*(n_t-0.5), 0, width=0.002, head_length=0.2, head_width=0.05, color="k", overhang=0.5)

for (i, text) in enumerate(t_list):
    #  ax.annotate(text, [mult*i, 0.1], ha='center')
    ax.text(mult*i, 0.1, text, ha='center', fontsize=15)
    ax.scatter([mult*i], [0], color="k", s=10)

for (i, text) in enumerate(era_list):
    #  ax.annotate(text, [mult*(i+0.5), -0.2], ha='center')
    ax.text(mult*(i+0.5), -0.2, text, ha='center', fontsize=10)

#  ax.set_xlim((-0.5, mult*n_t+0.5))
ax.set_ylim((-0.5, 0.5))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

plt.savefig("therm_history.pdf", bbox_inches="tight")
plt.close()
