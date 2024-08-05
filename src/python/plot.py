import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
from os import listdir 
from os.path import isdir, isfile, join
from glob import glob
from pathlib import Path
from scipy.optimize import curve_fit
import shutil

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#####################################################################################
# Background
#####################################################################################
def read_ode(dn):
    fn = dn + "ode.npz"
    data = np.load(fn)
    tau = data['tau']
    phi = data['phi']
    phi_d = data['phi_d']
    a = data['a']
    app_a = data["app_a"]
    a_end = data["a_end"]
    H_end = data["H_end"]
    err = data["err"]
    H = data["H"]
    mᵩ = data["m_phi"]

    #  print(tau[1:100], tau[-100:-1])

    return tau, phi, phi_d, a, app_a, a_end, H_end, err, H, mᵩ


def plot_background(dn):
    fn = dn + "ode.npz"
    out_dn = "figs/" + dn.replace("data/", "")
    out_fn = out_dn + "background.pdf"
    Path(out_dn).mkdir(parents=True, exist_ok=True)

    tau, phi, phi_d, a, app_a, a_end, H_end, err, H, mᵩ= read_ode(dn)

    tau_end = np.interp(a_end, a, tau)
    phi_end = np.interp(tau_end, tau, phi)

    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0, 0].plot(tau, phi, c="k")
    ax[0, 0].plot([tau_end, tau_end], [np.amin(phi), np.amax(phi)], c="grey", ls="--")

    ax[0, 0].set_xlabel(r"$\eta$")
    ax[0, 0].set_ylabel(r"$\phi$")
    #  ax[0, 0].set_xlim((0, 2e4))
    #  ax[0, 0].set_ylim((-0.01, 0.01))

    ax[0, 1].plot(tau, a, c="k")
    ax[0, 1].plot([tau_end, tau_end], [np.amin(a), np.amax(a)], c="grey", ls="--")

    ax[0, 1].set_xlabel(r"$\eta$")
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
    ax[1, 0].set_xlabel(r"$\eta$")
    ax[1, 0].set_ylabel("$H$")
    ax[1, 0].set_yscale("log")
    #  ax[1, 0].set_xlim((tau[0], 0))
    #  ax[1, 0].set_ylim((1e-6, 4e-6))

    ax[1, 1].plot(tau, a, c="k")
    ax[1, 1].plot([tau_end, tau_end], [np.amin(a), np.amax(a)], c="grey", ls="--")

    ax[1, 1].set_xlabel(r"$\eta$")
    ax[1, 1].set_ylabel("$a/a_i$")
    ax[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_fn, bbox_inches="tight")

#####################################################################################
# helper functions
#####################################################################################

def _parse_slash_float(s):
    """
    parse a string into float, even when they have the format 1/5
    """
    if "/" in s:
        nums = [float(x) for x in s.split("/")]
        return nums[0] / nums[1]
    else:
        return float(s)


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

    if sparse < 1:
        _skip = int(1/sparse)
        # have to ensure having both fields
        _fns_R = [x for x in fns if "_R" in x][::_skip]
        _fns_I = [x for x in fns if "_I" in x][::_skip]
        if len(_fns_R) != len(_fns_I):
            print(_fns_R, _fns_I)
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


def _get_m3_2_dir(dn, exclude=[]):
    m3_2_prefix = "m3_2="
    try:
        # get only subdirectories (full path)
        dirs = [x for x in listdir(dn) if isdir(join(dn, x))]
        dirs = [x for x in dirs if x not in exclude]
        #  print(dirs)
        # get values of xi from directory name
        m3_2s = [float(x.replace(m3_2_prefix, "")) for x in dirs]
        #  print(dirs, m3_2s)
    except ValueError:
        raise ValueError("Check the given directory!")
    dirs, m3_2s = zip(*sorted(zip(dirs, m3_2s)))
    return dirs, m3_2s

#####################################################################################
# plotting spectrum 
#####################################################################################

# to be excluded
DIR_TO_EXCLUDE = ['m_eff']

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
        
        
        fig = plt.figure()
        ax = fig.add_subplot()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot()

        cmap = mpl.colormaps['viridis'].reversed()
        # cmap2 = mpl.colormaps['magma'].reversed()
        
        # plot each spectrum
        for (i, (fn_i, ms_i)) in enumerate(zip(fns, ms)):
            full_path = join(dn, d_i, fn_i)
            data = np.load(full_path)
            f = data["f"]
            k = data["k"]
            Delta2 = data["Delta2"]
            #  print(f, k)

            if fn_I[i] == 1:
                color = cmap(ms_i/max(ms))
                #  ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, I", c=color, ls="--")
                ax.plot(k, f, c=color, ls="--")
                ax2.plot(k, Delta2, c=color, ls="--")
            elif fn_R[i] == 1:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, R", c=color)
                ax2.plot(k, Delta2, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, R", c=color)
            else:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$", c=color)
                ax2.plot(k, Delta2, label=rf"$m_\chi = {ms_i:.2f} m_\phi$", c=color)

        out_dn = "figs/" + dn.replace("data/", "") 
        Path(out_dn).mkdir(parents=True, exist_ok=True)
        out_fn = out_dn + f"f_ξ={ξ:.2f}" + out_suffix + ".pdf"

        ax.set_xlabel(r"$k/(a_e m_\phi)$")
        ax.set_ylabel(r"$f_\chi = |\beta|^2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.legend()
        fig.savefig(out_fn, bbox_inches="tight")
        plt.close(1)

        ax2.plot(k, k**3, label=rf"$\sim k^3$", c="k", ls="--")
        ax2.set_xlabel(r"$k/(a_e m_\phi)$")
        ax2.set_ylabel(r"$\Delta_\delta^2$")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        fig2.legend()
        out_fn = out_dn + f"isocurv_ξ={ξ:.2f}" + out_suffix + ".pdf"
        fig2.savefig(out_fn, bbox_inches="tight")
        plt.close(2)


def plot_f_m3_2(dn, sparse=1):
    """
    plot f; iterate over m3_2 first
    """
    dirs, m3_2s = _get_m3_2_dir(dn, exclude=DIR_TO_EXCLUDE)

    for (x, y) in zip(dirs, m3_2s):
        dn_i = dn + x + "/"
        plot_f(dn_i, sparse=sparse)

#####################################################################################
# integrated quantities
#####################################################################################

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
    get m_chi and nᵪ * (mᵪ / mᵩ)
    """
    data = np.load(fn)
    m = data["m_chi"]
    nm = data["n"] * m
    return m, nm


def _power_law(x, a, b, n):
    x = np.array(x)
    return a*x**n + b


def get_ρ_s_Trh(nm, mᵩ, rho_p):
    """
    calculate ρ_{χ, 0} / (s₀ T_{rh})
    assume all inputs are in reduced planck unit
    except nm which  is nᵪ * (mᵪ / m_ϕ)
    Now uses the correct formula (with rho_p)
    """
    #  return 2*np.pi * nm * mᵩ / Hₑ**2 / aₑ**3
    return 3.0 / 4.0 * nm * mᵩ / rho_p


def linear_f(x, a):
    return a*x

def inverse_f(x, a, b):
    return a * x + b

def plot_integrated_comp(dn, rho_p, mᵩ, add=False):
    """
    plot different integrated data for comparison
    assume m3_2/f_ξ/integrated.npz structure (with _R and _I for fields)
    """
    dirs, m3_2s = _get_m3_2_dir(dn, exclude=DIR_TO_EXCLUDE)
    m3_2s = np.array(m3_2s)

    fig, ax = plt.subplots()
    cmap = mpl.colormaps['viridis'].reversed()

    slopes = np.array([])
    err_slopes = np.array([])

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
                # plot both fields together
                full_fns_R = [x for x in full_fns if "_R" in x]
                full_fns_I = [x for x in full_fns if "_I" in x]
                for i, (fn_R, fn_I) in enumerate(zip(full_fns_R, full_fns_I)):
                    m_R, nm_R = _get_n(fn_R)
                    m_I, nm_I = _get_n(fn_I)
                    if np.array_equal(m_R, m_I):
                        m = m_R
                        ρ_s_T = get_ρ_s_Trh(nm_R + nm_I, mᵩ, rho_p)
                        print(dn_i, ρ_s_T[0])

                        color = cmap(y/max(m3_2s))
                        label = rf"$m_{{3/2}}={y:.1f}m_\phi$"
                        ax.plot(m, ρ_s_T, color=color, label=label)

                        #  trying to fit the first part
                        popt, pcov = curve_fit(linear_f, m[m<0.1], ρ_s_T[m<0.1])
                        perr = np.sqrt(np.diag(pcov))
                        #  print(y, popt, perr)
                        slopes = np.append(slopes, popt[0])
                        err_slopes = np.append(err_slopes, perr[0])

                        ax.plot(m, linear_f(m, *popt), alpha=0.5, color=color, ls="--")
                    else:
                        raise(ValueError("Something went wrong!"))
            else:
                # plot every integrated.npz individually
                for i, fn_i in enumerate(full_fns):
                    #  print(fn_i)
                    m, nm = _get_n(fn_i)
                    label = ''
                    ls = "-"
                    if fn_R[i] == 1:
                        ls = "-"
                        label = rf"$m_{{3/2}}={y:.1f}m_\phi$"
                    elif fn_I[i] == 1:
                        ls = "--"
                    else:
                        pass
                    color = cmap(y/max(m3_2s))
                    ρ_s_T = get_ρ_s_Trh(nm, mᵩ, rho_p)
                    ax.plot(m, ρ_s_T, color=color, ls=ls, label=label)
    
    # add linear part for visual guidance
    if add:
        #  ax.plot([0.1, 1], _power_law([0.1, 1], 1e-17, 0, 1), color="grey", ls="--", label="linear")
        pass

    ax.set_xlabel(r"$m_\chi / m_\phi$")
    ax.set_ylabel(r"$\rho_\chi/(s_0 T_{\rm rh})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    #  ax.set_xlim((0.05, 0.1))
    #  ax.set_ylim((3e-14, 5e-14))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='lower right')
    
    out_fn = dn.replace("data", "figs") + "integrated_comp"
    if add:
        out_fn += "_add"
    out_fn += ".pdf"

    plt.savefig(out_fn, bbox_inches="tight")
    plt.close()

    print("Data are", m3_2s, slopes, err_slopes)
    popt, perr = curve_fit(lambda x, a, b: a*x+b, m3_2s, slopes, sigma=err_slopes, absolute_sigma=True)
    print(popt, perr)
    print("Fitted:", m3_2s, inverse_f(m3_2s, *popt))
    fig, ax = plt.subplots()

    ax.plot(m3_2s, slopes, c="k", ls="-")
    ax.plot(m3_2s, inverse_f(m3_2s, *popt), c="k", ls="--")

    plt.savefig(out_fn.replace("comp", "slope"), bbox_inches="tight")
    plt.close()

#####################################################################################
# effective mass squared
#####################################################################################

def _draw_m(fn, ax):
    #  fn = dn + "m_eff.npz"
    data = np.load(fn)
    τ = data["tau"]
    m2_I = data["m2_I"]
    m2_R = data["m2_R"]

    #  fig, ax = plt.subplots()
    ax.plot(τ, m2_R, label="R", c="k")
    ax.plot(τ, m2_I, label="I", c="grey", ls="--")

    #  ax.set_xlabel(r"$\tau$")
    #  ax.set_ylabel(r"$m^2_{\rm eff} / (a m_\chi)^2$")

    #  plt.savefig(dn.replace("data", "figs") + "m2.pdf", bbox_inches="tight")


def plot_m_eff(dn):
    _, _, _, a, _, _, _, _, _, _ = read_ode(dn)
    dirs, m3_2s = _get_m3_2_dir(dn + "m_eff/")
    cmap = mpl.colormaps['magma'].reversed()

    for (d, m) in zip(dirs, m3_2s):
        fns, ms = _get_m_fn(join(dn + "m_eff/", d))
        fig, (ax1, ax2) = plt.subplots(ncols=2)
        
        for (x, y) in zip(fns, ms):
            data = np.load(join(dn, "m_eff/", d, x))
            τ = data["tau"]
            m2_I = data["m2_I"]
            m2_R = data["m2_R"]

            color = cmap(y/max(ms))
            ax1.plot(τ, m2_R, c=color, label=rf"$m_\chi={y:.2f}m_\phi$")
            ax1.plot(τ, m2_I, c=color, ls="--")

            ax2.plot(τ, m2_R/a**2, c=color, label=rf"$m_\chi={y:.2f}m_\phi$")
            ax2.plot(τ, m2_I/a**2, c=color, ls="--")

        ax1.set_xlabel(r"$\tau$")
        ax1.set_ylabel(r"$m^2_{\rm eff} / (m_\phi)^2$")
        ax1.set_yscale("log")
        ax2.set_xlabel(r"$\tau$")
        ax2.set_ylabel(r"$m^2_{\rm eff} / (a m_\phi)^2$")
        ax2.set_yscale("log")

        ax1.legend()

        plt.savefig(dn.replace("data", "figs") + f"m2_m3_2={m:.1f}.pdf", bbox_inches="tight")
        plt.close()


def cp_model_data(dn):
    Path(dn.replace("data", "figs")).mkdir(parents=True, exist_ok=True)
    src = dn+"model.dat"
    dest = src.replace("data", "figs")
    shutil.copy(src, dest)


if __name__ == "__main__":
    # TMode
    dn = "data/TMode-0.001-benchmark/"
    _, _, _, a, _, a_e, H_e, _, H, mᵩ = read_ode(dn)
    rho_p = 3 * H[-1]**2 * a[-1]**3
    #  print(a[50000])
    #  rho_p = a[50000]**3
    # cp_model_data(dn)
    # plot_background(dn)
    plot_f_m3_2(dn, sparse=0.5)
    #  plot_integrated_comp(dn, rho_p, mᵩ, add=True)
    #  plot_integrated_comp(dn, aₑ, Hₑ, mᵩ)
    #  plot_m_eff(dn)
    
    # SmallField
    # dn = "data/SmallField/"
    #  plot_background("data/SmallField/")
    #  plot_f("data/SmallField/")
    #  plot_integrated(dn)
