import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from os import listdir 
from os.path import isdir, isfile, join
from glob import glob
from pathlib import Path

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


def plot_background(dn):
    fn = dn + "ode.npz"
    out_dn = "figs/" + dn.replace("data/", "")
    out_fn = out_dn + "background.pdf"
    Path(out_dn).mkdir(parents=True, exist_ok=True)

    data = np.load(fn)
    tau = data['tau']
    phi = data['phi']
    phi_d = data['phi_d']
    a = data['a']
    app_a = data["app_a"]
    a_end = data["a_end"]
    err = data["err"]
    H = data["H"]

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

    #  ax[1, 0].plot(tau, phi, c="k")
    #  ax[1, 0].plot([tau_end, tau_end], [np.amin(phi), np.amax(phi)], c="grey", ls="--")
    #
    #  ax[1, 0].set_xlabel("$\eta$")
    #  ax[1, 0].set_ylabel("$\phi$")
    #  ax[1, 0].set_xlim([-2e6, 8e6])
    #  ax[1, 0].set_ylim([0.4, 0.6])

    ax[1, 0].plot(tau, H, c="k")
    ax[1, 0].plot([tau_end, tau_end], [np.amin(H), np.amax(H)], c="grey", ls="--")
    ax[1, 0].set_xlabel("$\eta$")
    ax[1, 0].set_ylabel("$H$")
    ax[1, 0].set_yscale("log")

    ax[1, 1].plot(tau, a, c="k")
    ax[1, 1].plot([tau_end, tau_end], [np.amin(a), np.amax(a)], c="grey", ls="--")

    ax[1, 1].set_xlabel("$\eta$")
    ax[1, 1].set_ylabel("$a/a_i$")
    ax[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_fn, bbox_inches="tight")


def _parse_slash_float(s):
    """
    parse a string into float, even when they have the format 1/5
    """
    if "/" in s:
        nums = [float(x) for x in s.split("/")]
        return nums[0] / nums[1]
    else:
        return float(s)


def plot_f(dn):
    # file names for output plots
    out_dn = "figs/" + dn.replace("data/", "") 
    out_fn = out_dn + "f.pdf"
    Path(out_dn).mkdir(parents=True, exist_ok=True)

    # recursively find npz files
    result = [y for x in os.walk(dn) for y in glob(os.path.join(x[0], '*.npz'))]

    # remove ode file
    f_fns = [x for x in result if "ode.npz" not in x]
    ode_fn = [x for x in result if "ode.npz" in x]
    print(f_fns, ode_fn)

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
        # after reading f_xi and m_chi, the data is ready to be read
            #  print(f_xi, m_chi)
            data = np.load(fn_i)
            f = data["f"]
            k = data["k"]
            #  print(f, k)
            ax.plot(k, f, label=rf"$f_\xi = {f_xi:.2f}, m_\chi = {m_chi}$")

    ax.set_xlabel(r"$k/(a_e m_\phi)$")
    ax.set_ylabel(r"$f_\chi = |\beta|^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.legend()
    plt.savefig(out_fn, bbox_inches="tight")
    plt.close()


def plot_f_new(dn):
    """
    plot f's stored in directory dn 
    assumes the following file structure
    dn/
        - ode.npz
        - f_ξ=$xi/
            - mᵪ=$m_chi.npz
        - ...
    """
    f_xi_str = "f_ξ="
    m_chi_str = "mᵪ="

    # get only subdirectories (full path)
    dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
    # get values of xi from directory name
    ξs = [_parse_slash_float(x.replace(f_xi_str, "").replace("_", "/")) for x in dirs]

    for (ξ, d_i) in zip(ξs, dirs):
        fns = [x for x in listdir(join(dn, d_i)) if isfile(join(dn, d_i, x))]
        # remove other files, like integrated.npz
        fns = [x for x in fns if x.startswith("mᵪ=")]
        # now assumes everything in f_ξ folders are npz files for f
        ms = [float(x.replace(m_chi_str, "").replace(".npz", "")) for x in fns]
        # sort the lists together
        fns, ms = zip(*sorted(zip(fns, ms)))
        #  print(fns, ms)

        fig, ax = plt.subplots()

        for (i, (fn_i, ms_i)) in enumerate(zip(fns, ms)):
            data = np.load(join(dn, d_i, fn_i))
            f = data["f"]
            k = data["k"]
            #  print(f, k)

            cmap = mpl.colormaps['viridis'].reversed()
            color = cmap(ms_i/max(ms))
            #  print(color)
        
            ax.plot(k, f, label=rf"$m_\chi = {ms_i:.1f} m_\phi$", c=color)

        out_dn = "figs/" + dn.replace("data/", "") 
        out_fn = out_dn + f"f_ξ={ξ:.2f}.pdf"

        ax.set_xlabel(r"$k/(a_e m_\phi)$")
        ax.set_ylabel(r"$f_\chi = |\beta|^2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.legend()
        plt.savefig(out_fn, bbox_inches="tight")
        plt.close()


def plot_integrated(dn):
    f_xi_str = "f_ξ="
    m_chi_str = "mᵪ="

    # get only subdirectories (full path)
    dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
    # get values of xi from directory name
    ξs = [_parse_slash_float(x.replace(f_xi_str, "").replace("_", "/")) for x in dirs]

    for (ξ, d_i) in zip(ξs, dirs):
        fn = join(dn, d_i, "integrated.npz")
        data = np.load(fn)
        m = data["m_chi"]
        f0 = data["f0"]
        ρ = data["rho"]
        n = data["n"]
        #  print(f0, ρ, n)

        fig, ax = plt.subplots()
        ax.scatter(m, f0, label=r"$f_\chi(k\leftarrow 0) $")
        ax.scatter(m, ρ, label=r"$a^4 \rho_\chi / m_\phi^4$")
        ax.scatter(m, n, label=r"$a^3 n_\chi / m_\phi^3$")

        ax.set_xlabel(r"$m_\chi / m_\phi$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.legend()

        out_dn = "figs/" + dn.replace("data/", "") 
        out_fn = out_dn + f"integrated_ξ={ξ:.2f}.pdf"
        plt.savefig(out_fn, bbox_inches="tight")
        plt.close()
        


if __name__ == "__main__":
    # TMode
    dn = "data/TMode/"
    plot_background("data/TMode/")
    plot_f_new("data/TMode/")
    plot_integrated(dn)
    
    # SmallField
    dn = "data/SmallField/"
    #  plot_background("data/SmallField/")
    #  plot_f("data/SmallField/")
    #  plot_f_new("data/SmallField/")
    #  plot_integrated(dn)
