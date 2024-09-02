import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt 

data = np.load("data/TMode-0.01/ode.npz")
tau = data["tau"]
app_a = data["app_a"]
a = data["a"]
H_end = data["H_end"]
a_end = data["a_end"]

# get_app_a = lambda x: np.interp(x, tau, app_a)
m_phi = 6.171250763232981e-6
# try small mass first 
m_chi = 0.01 * m_phi

def get_omega2(k, t):
    return k**2 + np.interp(t, tau, a)**2 * m_chi**2 - np.interp(t, tau, app_a)

def get_diff_eq(t, y, k):
    # print(t)
    return [y[1], -get_omega2(k, t) * y[0]]

k_array = np.logspace(-3, 1, 20) * a_end * H_end
fig, ax = plt.subplots()

for k in k_array:
    omega0 = np.sqrt(get_omega2(k, tau[0])+0.0j)
    # y0 = [1/np.sqrt(2*omega0)+0.0j, -1.0j * omega0/np.sqrt(2*omega0)]
    y0 = np.array([1/np.sqrt(2*k)+0.0j, -1.0j * k/np.sqrt(2*k)]) * np.exp(-1.0j * k *tau[0])
    t_span = [tau[0], tau[-1]]

    sol = solve_ivp(get_diff_eq, t_span, y0, args=(k,), rtol=1e-5)
    # print(k, sol.status)

    omega_e = np.sqrt(get_omega2(k, tau[-1])+0.0j)
    beta2 = np.abs(omega_e * sol.y[0][-1] - 1.0j * sol.y[1][-1])**2 / (2*omega_e)
    ax.scatter(k/(a_end * H_end), beta2*k**3, c="k")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$k/a_e H_e$")
ax.set_ylabel(r"$|\beta|^2$")
plt.savefig("demo-pp.pdf", bbox_inches="tight")
plt.close()
