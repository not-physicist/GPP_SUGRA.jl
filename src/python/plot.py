import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob


def plot_all():
    """
    Plot all the spectra
    """
    model = HilltopInf(0.5, 6)
    m_ϕ = model.m_phi
    m_χ = np.array([0.2, 1, 2]) * m_ϕ
    ξ = np.array([1 / 6])
    #  ξ = np.array([1 / 6, 0])

    # first ξ = 1/6
    for i, ξ_i in enumerate(ξ):
        fig, ax = plt.subplots()

        # TODO: when two many lines, use list to store styles and iterate over
        # them
        k, f = get_spec(m_ϕ, m_χ[0], ξ_i)
        ax.plot(k, f, c='k', label=r"$m_\chi = 0.2 m_\phi$")
        #  k, f = get_spec(m_ϕ, m_χ[1], ξ_i)
        #  ax.plot(k, f, c='tab:cyan', ls="--", label=r"$m_\chi = m_\phi$")
        #  k, f = get_spec(m_ϕ, m_χ[2], ξ_i)
        #  ax.plot(k, f, c='tab:orange', ls="--", label=r"$m_\chi = 2 m_\phi$")

        ax.set_xlabel(r"$k/(a_{\rm end} m_\phi)$")
        ax.set_ylabel(r"$f_\chi$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim([1e-12, 1e-4])
        ax.legend()

        plt.savefig(f"figs/f_{i+1}.pdf", bbox_inches="tight")
        plt.close()


def plot_ode():
    data = np.load("data/ode.npz")
    a = data["a"]
    phi = data["phi"]
    τ = data["tau"]

    plt.plot(τ, phi)
    #  plt.plot(a, phi)
    plt.xlabel("$a$")
    plt.ylabel(r"$\phi / m_{\rm pl}$")
    #  plt.show()
    plt.savefig("figs/ode.pdf", bbox_inches="tight")


def plot_background():
    fn = "data/ode.npz"
    data = np.load(fn)
    tau = data['tau']
    phi = data['phi']
    phi_d = data['phi_d']
    a = data['a']
    app_a = data["app_a"]
    a_end = data["a_end"]
    err = data["err"]

    tau_end = np.interp(a_end, a, tau)
    phi_end = np.interp(tau_end, tau, phi)

    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0, 0].plot(tau, phi, c="k")
    ax[0, 0].plot([tau_end, tau_end], [np.amin(phi), np.amax(phi)], c="grey", ls="--")

    ax[0, 0].set_xlabel("$\eta$")
    ax[0, 0].set_ylabel("$\phi$")

    ax[0, 1].plot(tau, a, c="k")
    ax[0, 1].plot([tau_end, tau_end], [np.amin(a), np.amax(a)], c="grey", ls="--")

    ax[0, 1].set_xlabel("$\eta$")
    ax[0, 1].set_ylabel("$a/a_i$")

    ax[1, 0].plot(tau, phi, c="k")
    ax[1, 0].plot([tau_end, tau_end], [np.amin(phi), np.amax(phi)], c="grey", ls="--")

    ax[1, 0].set_xlabel("$\eta$")
    ax[1, 0].set_ylabel("$\phi$")
    ax[1, 0].set_xlim([-2e6, 8e6])
    ax[1, 0].set_ylim([0.4, 0.6])

    ax[1, 1].plot(tau, a, c="k")
    ax[1, 1].plot([tau_end, tau_end], [np.amin(a), np.amax(a)], c="grey", ls="--")

    ax[1, 1].set_xlabel("$\eta$")
    ax[1, 1].set_ylabel("$a/a_i$")
    ax[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("figs/background.pdf", bbox_inches="tight")


def _parse_slash_float(s):
    """
    parse a string into float, even when they have the format 1/5
    """
    if "/" in s:
        nums = [float(x) for x in s.split("/")]
        return nums[0] / nums[1]
    else:
        return float(s)


def plot_f():
    dn = "data/"
    # recursively find npz files
    result = [y for x in os.walk(dn) for y in glob(os.path.join(x[0], '*.npz'))]

    # remove ode file
    f_fns = [x for x in result if "ode.npz" not in x]
    ode_fn = [x for x in result if "ode.npz" in x]
    #  print(f_fns, ode_fn)

    f_xi_str = "f_ξ="
    f_xi = None
    m_chi_str = "mᵪ="
    m_chi = None

    fig, ax = plt.subplots()

    for fn_i in f_fns:
    # iterate over different file paths
        path_list = fn_i.split("/")
        for path_i in path_list:
        # iterate over segments of the path
            if f_xi_str in path_i:
                f_xi =_parse_slash_float(path_i.replace(f_xi_str, "").replace("_", "/"))
            elif m_chi_str in path_i:
                m_chi = float(path_i.replace(m_chi_str, "").replace(".npz", ""))
        
        if f_xi is not None and m_chi is not None:
            print(f_xi, m_chi)
            data = np.load(fn_i)
            f = data["f"]
            k = data["k"]
            print(f, k)
            ax.plot(k, f, label=rf"$f_\xi = {f_xi:.2f}, m_\chi = {m_chi}$")

    ax.set_xlabel("k")
    ax.set_ylabel("f")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig("data/figs/f.pdf", bbox_inches="tight")
    plt.close()

plot_background()
#  plot_f()
