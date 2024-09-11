# NOT NECESSARY ANYMORE!
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 
import matplotlib as mpl

data = np.load("data/TMode-0.0035/ode.npz")
tau = data["tau"]
# print(tau[0], tau[-1])
app_a = data["app_a"]
a = data["a"]
H_end = data["H_end"]
a_end = data["a_end"]
N = np.log(data["a"])

# a = a/a_end 
# a_end = 1
# print(a_end, H_end)

# print(app_a[-1])

cmap = mpl.colormaps['magma']

# get_app_a = lambda x: np.interp(x, tau, app_a)
m_phi = 6.171250763232981e-6
# try small mass first 
m_chi = 5.0 * m_phi
# print(H_end, m_phi, m_chi)

def get_omega2(k, t):
    return k**2 + np.interp(t, tau, a)**2 * m_chi**2 - np.interp(t, tau, app_a)
    # return k**2 + np.interp(t, tau, a)**2 * m_chi**2

def get_diff_eq(t, y, k):
    # print(t)
    return [y[1], -get_omega2(k, t) * y[0]]

k_array = np.logspace(-3, 1, 10) * a_end * H_end
k_max = np.amax(k_array)
k_min = np.amin(k_array)
# print(m_chi, k_array)
# fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
fig, (ax2, ax3) = plt.subplots(ncols=2)

for k in k_array:
    omega0 = np.sqrt(get_omega2(k, tau[0])+0.0j)
    y0 = np.array([1/np.sqrt(2*omega0)+0.0j, -1.0j * omega0/np.sqrt(2*omega0)])
    # y0 = np.array([1/np.sqrt(2*k)+0.0j, -1.0j * k/np.sqrt(2*k)]) * np.exp(-1.0j * k *tau[0])
    # y0 = np.array([1/np.sqrt(2*k)+0.0j, -1.0j * np.sqrt(k/2)])
    t_span = [tau[0], tau[-1]]
    # print(t_span)

    sol = solve_ivp(get_diff_eq, t_span, y0, args=(k,), rtol=1e-5)
    # print(k, sol.status)

    omega_e = np.sqrt(get_omega2(k, tau[-1])+0.0j)
    beta2 = np.abs(omega_e * sol.y[0][:] - 1.0j * sol.y[1][:])**2 / (2*omega_e)
    # ax.scatter(k/(a_end * H_end), beta2*k**3, c="k")
    color = cmap( (np.log(k_max) - np.log(k))/(np.log(k_max) - np.log(k_min)) )
    if k == k_max or k == k_min:
        # print(sol.t, np.abs(sol.y[0])**2)
        # ax1.plot(np.interp(sol.t, tau, N), np.abs(sol.y[0])**2, label=f"$k={k:.2e}$", color=color)
        # ax2.plot(np.interp(sol.t, tau, N), beta2, label=f"$k={k:.2e}$", color=color)
        ax2.plot(np.interp(sol.t, tau, N), beta2, label=f"$k={k:.2e}$", color=color)
    else:
        # ax1.plot(np.interp(sol.t, tau, N), np.abs(sol.y[0])**2, color=color)
        # ax2.plot(np.interp(sol.t, tau, N), beta2, color=color)
        ax2.plot(np.interp(sol.t, tau, N), beta2, color=color)
        # print(beta2[-1])
        # ax3.scatter(k/a_end/H_end, beta2[-1]*(k/a_end/H_end)**3, color=color)
        ax3.scatter(k/a_end/H_end, beta2[-1], color=color)

    print(k, np.log(beta2[-1]))

# ax3.plot(k_array/a_end/H_end, (k_array/a_end/H_end)**(-3), color=color)
# ax1.set_xscale("log")
# ax1.set_yscale("log")
# ax1.set_xlabel(r"$\tau$")
# ax1.set_ylabel(r"$|\chi_k|^2$")
# ax1.legend(loc=2)

ax2.set_yscale("log")
ax2.set_xlabel(r"$N$")
ax2.set_ylabel(r"$|\beta_k|^2$")

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel(r"$k/a_e H_e$")
# ax3.set_ylabel(r"$|\beta_k|^2 k^3$")
ax3.set_ylabel(r"$|\beta_k|^2$")

plt.savefig("demo-pp.pdf")
plt.close()
