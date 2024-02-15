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


def plot_f_deprecated(dn):
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

    f_xi_prefix = "f_ξ="
    f_xi = None
    m_chi_prefix = "mᵪ="
    m_chi = None

    fig, ax = plt.subplots()

    for fn_i in f_fns:
    # iterate over different file paths
        path_list = fn_i.split("/")
        for path_i in path_list:
        # iterate over segments of the path
            if f_xi_prefix in path_i:
                f_xi =_parse_slash_float(path_i.replace(f_xi_prefix, "").replace("_", "/"))
            elif m_chi_prefix in path_i:
                m_chi = float(path_i.replace(m_chi_prefix, "").replace(".npz", ""))
        
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


def plot_f(dn, out_suffix=""):
    """
    plot f's stored in directory dn 
    assumes the following file structure
    dn/
        - ode.npz
        - f_ξ=$xi/
            - mᵪ=$m_chi.npz
        - ...
    """
    f_xi_prefix = "f_ξ="
    m_chi_prefix = "mᵪ="
    
    try:
        # get only subdirectories (full path)
        dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
        # get values of xi from directory name
        ξs = [_parse_slash_float(x.replace(f_xi_prefix, "").replace("_", "/")) for x in dirs]
        #  print(dirs, ξs)
    except ValueError:
        raise ValueError("Check the given directory!")

    for (ξ, d_i) in zip(ξs, dirs):
        fns = [x for x in listdir(join(dn, d_i)) if isfile(join(dn, d_i, x))]
        # remove other files, like integrated.npz
        fns = [x for x in fns if x.startswith("mᵪ=")]

        try:
            ms = [float(x.replace(m_chi_prefix, "").replace(".npz", "")) for x in fns]
        except ValueError:
            # then probably there is suffix in the file name, e.g. for specifying R and I parents
            _fns_wo_suffix = [x.replace("_R", "").replace("_I", "") for x in fns] 
            ms = [float(x.replace(m_chi_prefix, "").replace(".npz", "")) for x in _fns_wo_suffix]
            

        # sort the lists together
        fns, ms = zip(*sorted(zip(fns, ms)))
        # arrays to store if th file corresponds to R/I field
        # need two arrays since sometimes no distinct fields
        fn_R = [1 if "_R" in x else 0 for x in fns]
        fn_I = [1 if "_I" in x else 0 for x in fns]
        
        fig, ax = plt.subplots()

        cmap = mpl.colormaps['viridis'].reversed()
        cmap2 = mpl.colormaps['magma'].reversed()
        
        # plot each spectrum
        for (i, (fn_i, ms_i)) in enumerate(zip(fns, ms)):
            full_path = join(dn, d_i, fn_i)
            data = np.load(full_path)
            f = data["f"]
            k = data["k"]
            #  print(f, k)

            if fn_I[i] == 1:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.1f} m_\phi$, I", c=color, ls="--")
            elif fn_R[i] == 1:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.1f} m_\phi$, R", c=color)
            else:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.1f} m_\phi$", c=color)

        out_dn = "figs/" + dn.replace("data/", "") 
        Path(out_dn).mkdir(parents=True, exist_ok=True)
        out_fn = out_dn + f"f_ξ={ξ:.2f}" + out_suffix + ".pdf"

        ax.set_xlabel(r"$k/(a_e m_\phi)$")
        ax.set_ylabel(r"$f_\chi = |\beta|^2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        plt.legend()
        plt.savefig(out_fn, bbox_inches="tight")
        plt.close()


def plot_f_m3_2(dn):
    """
    plot f; iterate over m3_2 first
    """
    m3_2_prefix = "m3_2="
    try:
        # get only subdirectories (full path)
        dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
        # get values of xi from directory name
        m3_2s = [float(x.replace(m3_2_prefix, "")) for x in dirs]
        #  print(dirs, m3_2s)
    except ValueError:
        raise ValueError("Check the given directory!")
    
    for (x, y) in zip(dirs, m3_2s):
        dn_i = dn + x + "/"
        plot_f(dn_i)


def plot_integrated(dn):
    f_xi_prefix = "f_ξ="
    m_chi_prefix = "mᵪ="

    # get only subdirectories (full path)
    dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
    # get values of xi from directory name
    ξs = [_parse_slash_float(x.replace(f_xi_prefix, "").replace("_", "/")) for x in dirs]

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


def plot_m_eff(dn):
    fn = dn + "m_eff.npz"
    data = np.load(fn)
    τ = data["tau"]
    m2_I = data["m2_I"]
    m2_R = data["m2_R"]

    fig, ax = plt.subplots()
    ax.plot(τ, m2_R, label="R", c="k")
    ax.plot(τ, m2_I, label="I", c="grey", ls="--")

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$m^2_{\rm eff} / m_\chi^2$")

    plt.savefig(dn.replace("data", "figs") + "m2.pdf", bbox_inches="tight")

# TODO: fixed borked plot due to _L and _R
if __name__ == "__main__":
    # TMode
    dn = "data/TMode/"
    #  plot_background("data/TMode/")
    plot_f_m3_2("data/TMode/")
    #  plot_integrated(dn)
    #  plot_m_eff(dn)
    
    # SmallField
    dn = "data/SmallField/"
    #  plot_background("data/SmallField/")
    #  plot_f("data/SmallField/")
    #  plot_integrated(dn)
