import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from os import listdir 
from os.path import isdir, isfile, join
from glob import glob
from pathlib import Path

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

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


def _get_xi_dn(dn):
    """
    get all directories containing different ξ's and their numerical values
    """
    f_xi_prefix = "f_ξ="
    try:
        # get only subdirectories (full path)
        dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
        # get values of xi from directory name
        ξs = [_parse_slash_float(x.replace(f_xi_prefix, "").replace("_", "/")) for x in dirs]
        #  print(dirs, ξs)
    except ValueError:
        raise ValueError("Check the given directory!")
    return dirs, ξs


def _get_m_fn(dn, sparse=1):
    """
    get all file names for different mᵪ's and their numerical values
    taking into account some files can have _R and _I suffixes

    sparse=False
        shoulde be a number between 0 and 1 to define the "sampling rate"
        right now only works for files with _R and _I suffixes
    """
    m_chi_prefix = "mᵪ="
    fns = [x for x in listdir(dn) if isfile(join(dn, x))]
    # remove other files, like integrated.npz
    fns = [x for x in fns if x.startswith("mᵪ=")]

    try:
        ms = [float(x.replace(m_chi_prefix, "").replace(".npz", "")) for x in fns]
    except ValueError:
        # then probably there is suffix in the file name, e.g. for specifying R and I parents
        _fns_wo_suffix = [x.replace("_R", "").replace("_I", "") 
                              for x in fns] 
        ms = [float(x.replace(m_chi_prefix, "").replace(".npz", "")) for x in _fns_wo_suffix]

    # sort the lists together
    fns, ms = zip(*sorted(zip(fns, ms)))

    if sparse is not False:
        _skip = int(1/sparse)
        # have to ensure having both fields
        _fns_R = [x for x in fns if "_R" in x][::_skip]
        _fns_I = [x for x in fns if "_I" in x][::_skip]
        if len(_fns_R) != len(_fns_I):
            raise(ValueError("You probably want to clense the data directory and re-generate the data files."))
        #  print(_fns_R, "\n", _fns_I)
        fns = _fns_R + _fns_I

        # re-parse m and sort again
        _fns_wo_suffix = [x.replace("_R", "").replace("_I", "") 
                              for x in fns] 
        ms = [float(x.replace(m_chi_prefix, "").replace(".npz", "")) for x in _fns_wo_suffix]
        fns, ms = zip(*sorted(zip(fns, ms)))

    return fns, ms


def _get_integrated_fn(dn):
    int_prefix = "integrated"
    fns = [x for x in listdir(dn) if isfile(join(dn, x))]
    # remove other files, like integrated.npz
    fns = [x for x in fns if x.startswith(int_prefix)]
    fn_R = [1 if "_R" in x else 0 for x in fns]
    fn_I = [1 if "_I" in x else 0 for x in fns]
    return fns, fn_R, fn_I


def _get_m3_2_dir(dn):
    m3_2_prefix = "m3_2="
    try:
        # get only subdirectories (full path)
        dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
        # get values of xi from directory name
        m3_2s = [float(x.replace(m3_2_prefix, "")) for x in dirs]
        #  print(dirs, m3_2s)
    except ValueError:
        raise ValueError("Check the given directory!")
    dirs, m3_2s = zip(*sorted(zip(dirs, m3_2s)))
    return dirs, m3_2s


def plot_f(dn, out_suffix="", sparse=1):
    """
    plot f's stored in directory dn 
    assumes the following file structure
    dn/
        - ode.npz
        - f_ξ=$xi/
            - mᵪ=$m_chi.npz
        - ...

    sparse=False
        shoulde be a number between 0 and 1 to define the "sampling rate"
    """

    dirs, ξs = _get_xi_dn(dn) 
    
    for (ξ, d_i) in zip(ξs, dirs):
        fns, ms = _get_m_fn(join(dn, d_i), sparse=sparse)
        #  if not sparse:

        # arrays to store if th file corresponds to R/I field
        # need two arrays since sometimes no distinct fields
        # TODO: put the following part in _get_m_fn
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
                #  ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, I", c=color, ls="--")
                ax.plot(k, f, c=color, ls="--")
            elif fn_R[i] == 1:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, R", c=color)
            else:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$", c=color)

        out_dn = "figs/" + dn.replace("data/", "") 
        Path(out_dn).mkdir(parents=True, exist_ok=True)
        out_fn = out_dn + f"f_ξ={ξ:.2f}" + out_suffix + ".pdf"

        ax.set_xlabel(r"$k/(a_e m_\phi)$")
        ax.set_ylabel(r"$f_\chi = |\beta|^2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.legend()
        plt.savefig(out_fn, bbox_inches="tight")
        plt.close()


def plot_f_m3_2(dn, sparse=1):
    """
    plot f; iterate over m3_2 first
    """
    dirs, m3_2s = _get_m3_2_dir(dn)

    for (x, y) in zip(dirs, m3_2s):
        dn_i = dn + x + "/"
        plot_f(dn_i, sparse=sparse)


def plot_integrated(dn):
    dirs, ξs = _get_xi_dn(dn)

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


def _get_n(fn):
    """
    get m_chi and its eventual energy density from fn (npz file!)
    """
    data = np.load(fn)
    m = data["m_chi"]
    rho = data["n"] * m
    return m, rho


def plot_integrated_comp(dn, add=False):
    """
    plot different integrated data for comparison
    assume m3_2/f_ξ/integrated.npz structure (with _R and _I for fields)
    """
    dirs, m3_2s = _get_m3_2_dir(dn)

    fig, ax = plt.subplots()
    cmap = mpl.colormaps['viridis'].reversed()

    for (x, y) in zip(dirs, m3_2s):
        dn_i = dn + x + "/"
        ξ_dirs, ξs = _get_xi_dn(dn_i)
        ξ_dirs_full = [join(dn, x, i) for i in ξ_dirs]
        #  print(ξ_dirs_full, ξs)
        

        for ξ_dir_i in ξ_dirs_full:
            fns, fn_R, fn_I = _get_integrated_fn(ξ_dir_i)
            full_fns = [join(ξ_dir_i, x) for x in fns]
            #  print(full_fns, fn_R, fn_I)
            
            if add:
                full_fns_R = [x for x in full_fns if "_R" in x]
                full_fns_I = [x for x in full_fns if "_I" in x]
                for i, (fn_R, fn_I) in enumerate(zip(full_fns_R, full_fns_I)):
                    m_R, rho_R = _get_n(fn_R)
                    m_I, rho_I = _get_n(fn_I)
                    if np.array_equal(m_R, m_I):
                        m = m_R
                        rho = rho_R + rho_I

                        color = cmap(y/max(m3_2s))
                        ax.plot(m, rho, color=color)
                    else:
                        raise(ValueError("Something went wrong!"))
            else:
                # plot every integrated.npz individually
                for i, fn_i in enumerate(full_fns):
                    #  print(fn_i)
                    m, rho = _get_n(fn_i)

                    #  label=f"$m_{{3/2}}={y:.1f}$"
                    ls = "-"
                    if fn_R[i] == 1:
                        ls = "-"
                    elif fn_I[i] == 1:
                        ls = "--"
                    else:
                        pass
                    color = cmap(y/max(m3_2s))
                    ax.plot(m, rho, color=color, ls=ls)

    ax.set_xlabel(r"$m_\chi / m_\phi$")
    ax.set_ylabel(r"$a^3 n_\chi m_\chi / m_\phi^4$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #  plt.legend()
    
    out_fn = dn.replace("data", "figs") + "integrated_comp"
    if add:
        out_fn += "_add"
    out_fn += ".pdf"

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
    ax.set_ylabel(r"$m^2_{\rm eff} / (a m_\chi)^2$")

    plt.savefig(dn.replace("data", "figs") + "m2.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # TMode
    dn = "data/TMode/"
    #  plot_background("data/TMode/")
    plot_f_m3_2("data/TMode/", sparse=0.15)
    #  plot_integrated_comp(dn, add=True)
    #  plot_integrated_comp(dn)
    #  plot_m_eff(dn)
    
    # SmallField
    dn = "data/SmallField/"
    #  plot_background("data/SmallField/")
    #  plot_f("data/SmallField/")
    #  plot_integrated(dn)
