import numpy as np
from scipy.ndimage import gaussian_filter1d

import os
from os import listdir 
from os.path import isdir, isfile, join, exists
from glob import glob
from pathlib import Path
from scipy.optimize import curve_fit
import shutil

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

m_pl = 2.44e18 # reduced planck

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

    # print(tau[1:10] - tau[2:11], tau[-10:-1] - tau[-11:-2])

    return tau, phi, phi_d, a, app_a, a_end, H_end, err, H, mᵩ


def plot_background(dn):
    fn = dn + "ode.npz"
    out_dn = "figs/" + dn.replace("data/", "")
    out_fn = out_dn + "background.pdf"
    out_fn2 = out_dn + "background_draft.pdf"
    Path(out_dn).mkdir(parents=True, exist_ok=True)

    tau, phi, phi_d, a, app_a, a_end, H_end, err, H, mᵩ= read_ode(dn)
    N = np.log(a)
    # print(H[1:10], H[-10:-1])

    tau_end = np.interp(a_end, a, tau)
    phi_end = np.interp(tau_end, tau, phi)

    fig, ax = plt.subplots(ncols=2, nrows=2)
    ax[0, 0].plot(N, phi, c="k")
    # ax[0, 0].plot([tau_end, tau_end], [np.amin(phi), np.amax(phi)], c="grey", ls="--")

    ax[0, 0].set_xlabel(r"$N$")
    ax[0, 0].set_ylabel(r"$\phi/m_{\rm pl}$")
    
    ax[0, 1].plot(N, H*m_pl, c="k")
    ax[0, 1].set_xlabel(r"$N$")
    ax[0, 1].set_ylabel("$H / GeV$")
    ax[0, 1].set_yscale("log")

    ax[1, 0].plot(N, err, c="k")
    ax[1, 0].set_xlabel(r"$N$")
    ax[1, 0].set_ylabel("error")
    ax[1, 0].set_yscale("log")

    ax[1, 1].plot(N, -6*app_a/a**2*m_pl**2, c="k")
    ax[1, 1].set_xlabel(r"$N$")
    ax[1, 1].set_ylabel("$R/GeV^2$")

    plt.tight_layout()
    plt.savefig(out_fn, bbox_inches="tight")
    
    ax[1, 0].clear()
    ax[1, 0].plot(N, -6*app_a/a**2*m_pl**2, c="k")
    ax[1, 0].set_xlabel(r"$N$")
    ax[1, 0].set_ylabel("$R/GeV^2$")

    ax[1, 1].set_xlim((2, 6))
    ax[1, 1].set_ylim((-5e25, 6e25))

    plt.tight_layout()
    plt.savefig(out_fn2, bbox_inches="tight")
    plt.close()

    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    ax1.plot(N, -6*app_a/a**2*m_pl**2, c="k")
    ax1.set_xlabel(r"$N$")
    ax1.set_ylabel("$R/GeV^2$")

    ax2.plot(N, -6*app_a/a**2*m_pl**2, c="k")
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel("$R/GeV^2$")
    ax2.set_xlim((1, 4))
    ax2.set_ylim((-1e27, 1e26))
    
    # print(app_a[-100:-1])
    ax3.plot(N, -6*app_a/a**2*m_pl**2, c="k")
    ax3.set_xlabel(r"$N$")
    ax3.set_ylabel("$R/GeV^2$")
    ax3.set_xlim((6.9, np.max(N)))
    ax3.set_ylim((-1e21, 1e21))

    plt.tight_layout()
    plt.savefig(out_dn + "app_a.pdf", bbox_inches="tight")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.plot(N, H, c="k", label="$H$")
    # HUbble from derivative
    H_deriv = np.diff(a)/np.diff(tau)/a[:-1]**2
    ax1.plot(N[:-1], H_deriv, c="tab:blue", ls="--", label="$a'/a^2$")
    ax1.set_xlabel(r"$N$")
    ax1.legend(loc=3)

    ax2.plot(N[:-1], (H[:-1]-H_deriv)/H[:-1], c="k")
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$\Delta H / H$")
    plt.savefig(out_dn + "H.pdf", bbox_inches="tight")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.plot(N, app_a, c="k", label="from 2nd friedman")
    ap = np.diff(a)/np.diff(tau)
    app = np.diff(ap) / np.diff(tau[:-1])
    ax1.plot(N[:-2], app / a[:-2], c="tab:blue", alpha=0.4, label="from deriv.")
    ax1.legend(loc=2)
    ax1.set_xlabel(r"$N$")
    ax1.set_ylabel(r"$a''/a$")
    ax2.plot(N[:-2], (app_a[:-2] - app / a[:-2])/app_a[:-2], c="k")
    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$\Delta (a''/a) / (a''/a)$")

    plt.savefig(out_dn + "app_a_comp.pdf", bbox_inches="tight")
    plt.close()

    try:
        data = np.load(dn + "ode.npz")
        phi_d_sr = data["phi_d_sr"]
    except KeyError:
        print("No keys for phi_d_sr detected. SKIPPING")
    finally:
        fig, ax = plt.subplots()
        ax.plot(N, phi_d/(a*H), color="k", label="num")
        # ax.plot(N, phi_d_sr/(a*H), color="blue", label="SR", ls="--")
        # print(phi_d/(a*H)[0], phi_d_sr/(a*H))
        # ax.set_xlim(-0.2, 0.1)
        ax.set_ylim(-5, 5)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$d \phi / dN$")
        plt.tight_layout()
        plt.savefig(out_dn + "phi_d.pdf", bbox_inches="tight")

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
        dirs = [x for x in listdir(dn) if isdir(join(dn, x)) and f_xi_prefix in x]
        # get values of xi from directory name
        ξs = [_parse_slash_float(x.replace(f_xi_prefix, "").replace("_", "/")) for x in dirs]
        #  print(dirs, ξs)
    except ValueError:
        raise ValueError("Check the given directory!")
    return dirs, ξs


def _get_m_fn(dn, sparse=1.0):
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
        _fns_wo_suffix = [x.replace("_R", "").replace("_I", "").replace("_nosugra", "")
                              for x in fns] 
        ms = [float(x.replace(m_chi_prefix, "").replace(".npz", "")) for x in _fns_wo_suffix]
    
    # sort the lists together
    # print(fns, ms)
    try: 
        fns, ms = zip(*sorted(zip(fns, ms)))
    except ValueError:
        print("ValueError", fns, ms)
        return ValueError
    
    
    _skip = int(1/sparse)
    if sparse < 1 and "nosugra" not in dn:
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
    elif sparse < 1 and "nosugra" in dn:
        _fns_nosugra = [x for x in fns][::_skip]
        # print(_fns_nosugra)
        ms = [float(x.replace(m_chi_prefix, "").replace(".npz", "")) for x in _fns_nosugra]
        fns, ms = zip(*sorted(zip(_fns_nosugra, ms)))

        return fns, ms
    
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
        # print(dirs)
        # get values of xi from directory name
        m3_2s = ["0.0" if x == "nosugra" else float(x.replace(m3_2_prefix, "")) for x in dirs]
        #  print(dirs, m3_2s)
    except ValueError:
        raise ValueError("Check the given directory!", dn)

    dirs, m3_2s = zip(*sorted(zip(dirs, m3_2s)))
    dirs = list(dirs)
    m3_2s = list(m3_2s)

    if exists(dn + "nosugra"):
        dirs.append("nosugra")
        m3_2s.append(0.0)

    # print(dirs)
    return dirs, m3_2s

#####################################################################################
# plotting spectrum 
#####################################################################################

# to be excluded; nosugra will be plotted separately
DIR_TO_EXCLUDE = ['m_eff', "nosugra"]

def plot_f(dn, out_suffix="", sparse=1.0):
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
        # print(fns, ms)
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

        fig3 = plt.figure()
        ax3 = fig3.add_subplot()

        fig4 = plt.figure()
        ax4 = fig4.add_subplot()

        cmap = mpl.colormaps['viridis'].reversed()
        # cmap2 = mpl.colormaps['magma'].reversed()
        
        # plot each spectrum
        for (i, (fn_i, ms_i)) in enumerate(zip(fns, ms)):
            full_path = join(dn, d_i, fn_i)
            data = np.load(full_path)
            f = data["f"]
            k = data["k"]
            Delta2 = data["Delta2"]

            # get spectral index 
            itp_Delta2 = lambda x: np.interp(x, k, Delta2)
            n_Delta = (np.log10(itp_Delta2(0.01)) - np.log10(itp_Delta2(0.1)))/(np.log10(0.01) - np.log10(0.1))
            # print(n_Delta)

            Delta2_beta = data["Delta2_beta"]
            # print(len(Delta2), len(Delta2_beta))
            # print(dn, d_i, ms_i, Delta2, Delta2_beta)
            err = np.abs(data["err"])
            # print(ms_i, k, f)
            # print(k.shape, err.shape)
            # print(ms_i, np.amin(err))

            # ax3.scatter(ms_i, np.amax(err), c="k", label="maximal")
            # ax3.scatter(ms_i, np.amin(err), c="k", label="minimal", marker="x")

            if fn_I[i] == 1:
                color = cmap(ms_i/max(ms))
                #  ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, I", c=color, ls="--")
                ax.plot(k, f, c=color, ls="--")
                # ax2.plot(k, Delta2, c=color, ls="--")
                ax3.plot(k, err, c=color)
                ax4.plot(k, f*k**3, c=color, ls="--")

            elif fn_R[i] == 1:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, R", c=color)
                ax2.plot(k, Delta2, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, R", c=color)
                # ax2.plot(k, Delta2_beta, label=rf"$m_\chi = {ms_i:.2f} m_\phi, R, \beta^4$", c=color, ls="--")
                ax3.plot(k, err, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, R", c=color)
                ax4.plot(k, f*k**3, label=rf"$m_\chi = {ms_i:.2f} m_\phi$, R", c=color)
            else:
                color = cmap(ms_i/max(ms))
                ax.plot(k, f, label=rf"$m_\chi = {ms_i:.2f} m_\phi$", c=color)
                ax2.plot(k, Delta2, label=rf"$m_\chi = {ms_i:.2f} m_\phi$", c=color)
                # ax2.plot(k, Delta2_beta, label=rf"$m_\chi = {ms_i:.2f} m_\phi, \beta^4$", c=color, ls="--")
                ax3.plot(k, err, label=rf"$m_\chi = {ms_i:.2f} m_\phi$", c=color)
                ax4.plot(k, f*k**3, label=rf"$m_\chi = {ms_i:.2f} m_\phi$", c=color)

        out_dn = "figs/" + dn.replace("data/", "") 
        Path(out_dn).mkdir(parents=True, exist_ok=True)
        out_fn = out_dn + f"f_ξ={ξ:.2f}" + out_suffix + ".pdf"

        ax.set_xlabel(r"$k/(a_e H_e)$")
        ax.set_ylabel(r"$f_\chi = |\beta_k|^2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.legend()
        fig.savefig(out_fn, bbox_inches="tight")
        plt.close(1)

        ax2.plot(k, k**3*1e-2, label=rf"$\sim k^3$", c="k", alpha=0.3)
        ax2.set_xlabel(r"$k/(a_e H_e)$")
        ax2.set_ylabel(r"$\Delta_\delta^2$")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim((4e-2, 1e0))
        # ax2.set_ylim((1e-5, 1e+5))
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        fig2.legend(bbox_to_anchor=(0.35, 0.9))
        out_fn = out_dn + f"isocurv_ξ={ξ:.2f}" + out_suffix + ".pdf"
        fig2.savefig(out_fn, bbox_inches="tight")
        plt.close(2)

        # ax3.set_xlabel(r"$m_\chi/m_\phi$")
        ax3.set_xlabel(r"$k / (a_e H_e)$")
        ax3.set_ylabel(r"$|\alpha|^2 - |\beta|^2 - 1$")
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # fig3.legend()
        out_fn = out_dn + f"error_ξ={ξ:.2f}" + out_suffix + ".pdf"
        fig3.savefig(out_fn, bbox_inches="tight")
        plt.close(3)

        ax4.set_xlabel(r"$k/(a_e H_e)$")
        ax4.set_ylabel(r"$ |\beta_k|^2 \cdot (k/ a_e H_e)^3$")
        ax4.set_xscale("log")
        ax4.set_yscale("log")
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        fig4.legend()
        fig4.savefig(out_dn + f"f_k3_ξ={ξ:.2f}" + out_suffix + ".pdf", bbox_inches="tight")
        plt.close(4)


def plot_f_m3_2(dn, sparse=1):
    """
    plot f; iterate over m3_2 first
    """
    dirs, m3_2s = _get_m3_2_dir(dn, exclude=DIR_TO_EXCLUDE)

    for (x, y) in zip(dirs, m3_2s):
        dn_i = dn + x + "/"
        plot_f(dn_i, sparse=sparse)


def comp_div_num_den(k, f):
    """
    compute (IR) divergent number density by extrapolation and setting cutoff at CMB scale
    """
    return np.trapezoid(f*k**2, k)


def plot_integrated_nosugra(dn, mᵩ, Hₑ):
    """
    dn should be the directory of a specific r
    """
    # assume only one \xi
    Path(dn.replace("data", "figs") + "nosugra/").mkdir(parents=True, exist_ok=True)
    dn_nosugra_full = dn + "nosugra/f_ξ=0.0"
    fns, ms = _get_m_fn(dn_nosugra_full)
    # print(fns, ms)
    cmap = mpl.colormaps['viridis'].reversed()
    print("m_phi = ", mᵩ, "H_e = ", Hₑ)

    fig1 = plt.figure()
    ax = fig1.add_subplot()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    lines_array = []
    ρ_s0_T_array = []
    for (i, (fn_i, ms_i)) in enumerate(zip(fns, ms)):
        print("----")
        print(r"m_phi = ", ms_i)
        fn_full = join(dn_nosugra_full, fn_i)
        data = np.load(fn_full)
        color = cmap(ms_i/max(ms))
        # print(data)
        f = data["f"]
        k = data["k"]
        
        k_new, f_new, n = get_extra_f(k, f, 0.1, 60)

        # number density; in the unit of k
        num_den = comp_div_num_den(k_new, f_new)
        ρ_s0_T = get_ρ_s_Trh(num_den*ms_i, mᵩ, Hₑ)
        print(n, np.log10(num_den), ρ_s0_T)
        ρ_s0_T_array.append(ρ_s0_T)
        
        # save the "line"s for slowing less label handles
        label = rf"$m_\chi = {ms_i:.2f} m_\phi$"
        line,  = ax.plot(k_new, f_new, color=color, label=label)
        ax2.plot(k, f, color=color, label=label)
        lines_array.append(line)
    
    ax.set_xlabel(r"$k/(a_e H_e)$")
    ax.set_ylabel(r"$f_\chi = |\beta|^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_xlim((1e-4,1e1))
    # ax.set_ylim((1e0,1e8))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # don't need all the labels
    # fig.legend(handles=lines_array[::2])
    fig1.legend(handles=lines_array)
    out_fn = dn_nosugra_full.replace("data", "figs").replace("/f_ξ=0.0", "") + "/f_extra.pdf"
    fig1.savefig(out_fn, bbox_inches="tight")
    plt.close(1)

    ax2.set_xlabel(r"$k/(a_e H_e)$")
    ax2.set_ylabel(r"$f_\chi = |\beta|^2$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    # ax.set_xlim((1e-4,1e1))
    # ax.set_ylim((1e0,1e8))
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # don't need all the labels
    # fig.legend(handles=lines_array[::2])
    fig2.legend(handles=lines_array)
    out_fn = dn_nosugra_full.replace("data", "figs").replace("/f_ξ=0.0", "") + "/f.pdf"
    fig2.savefig(out_fn, bbox_inches="tight")
    plt.close(2)

    fig, ax = plt.subplots()
    ax.plot(ms, ρ_s0_T_array, color="k")
    ax.set_xlabel(r"$m_\chi / m_\phi$")
    ax.set_ylabel(r"$\rho / s_0 T_{\rm rh}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    out_fn = dn.replace("data", "figs") + "/integrated_nosugra.pdf"
    fig.savefig(out_fn, bbox_inches="tight")
    plt.close()

    return ms, ρ_s0_T_array


def get_extra_f(k, f, mask_upper = 0.1, N_cmb = 60):
    """
    extrapolate the f into cmb scale
    mask_upper: upper limit of the range where power index is calculated
    """
    # mask = (k > 0.01) & (k < 0.1)
    mask = k < mask_upper
    n = (np.log10(f[mask][0]) - np.log10(f[mask][-1])) / (np.log10(k[mask][0]) - np.log10(k[mask][-1])) 
    # print(f"mᵪ={ms_i:.3f},\tpower={n:.3f}")
    
    # N_cmb = 60
    k_cmb = np.exp(-N_cmb) 
    # print(k_cmb, np.log10(k_cmb) - np.log10(k[0]))
    # f_cmb = 10**((np.log10(k_cmb) - np.log10(k[0])) * n) * k[0]
    # print(f_cmb)

    k_new = np.logspace(np.log10(k_cmb), np.log10(k[0]), base=10, num=10)
    f_new = 10**((np.log10(k_new) - np.log10(k[0])) * n) * f[0]

    k_new = np.concatenate((k_new[:-1], k))
    f_new = np.concatenate((f_new[:-1], f))
    # print(k_new, f_new)
    return k_new, f_new, n

#####################################################################################
# integrated quantities
#####################################################################################

# def plot_integrated(dn):
#     dirs, ξs = _get_xi_dn(dn)
#
#     for (ξ, d_i) in zip(ξs, dirs):
#         fn = join(dn, d_i, "integrated.npz")
#         data = np.load(fn)
#         m = data["m_chi"]
#         f0 = data["f0"]
#         ρ = data["rho"]
#         n = data["n"]
#         # print(m, ρ, n)
#
#         fig, ax = plt.subplots()
#         ax.scatter(m, f0, label=r"$f_\chi(k\leftarrow 0) $")
#         ax.scatter(m, ρ, label=r"$a^4 \rho_\chi / m_\phi^4$")
#         ax.scatter(m, n, label=r"$a^3 n_\chi / m_\phi^3$")
#
#         ax.set_xlabel(r"$m_\chi / m_\phi$")
#         ax.set_xscale("log")
#         ax.set_yscale("log")
#         plt.legend()
#
#         out_dn = "figs/" + dn.replace("data/", "") 
#         out_fn = out_dn + f"integrated_ξ={ξ:.2f}.pdf"
#         plt.savefig(out_fn, bbox_inches="tight")
#         plt.close()

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


def get_ρ_s_Trh(nm, mᵩ, Hₑ):
    """
    calculate ρ_{χ, 0} / (s₀ T_{rh})
    
    m should be in mᵩ-unit 
    n should be in (aₑHₑ)^3-unit
    """
    #  return 2*np.pi * nm * mᵩ / Hₑ**2 / aₑ**3
    return nm * mᵩ * Hₑ / 4.0


def linear_f(x, a):
    return a*x

def inverse_f(x, a, b):
    return a * x + b

def plot_integrated_comp(dn, aₑ, Hₑ, mᵩ, add=False, nosugra=None):
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
        if y == 0.0:
            color = cmap(0.0)
        else:
            max_m3_2 = max(m3_2s)
            # exclude m3_2 = 0
            min_m3_2 = min([x for x in m3_2s if x != 0.0])
            full_range = np.log10(max_m3_2) - np.log10(min_m3_2)
            # print(max_m3_2, min_m3_2, full_range)
            color = cmap(1-abs(np.log10(y))/full_range)

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
                        ρ_s_T = get_ρ_s_Trh((nm_R + nm_I)/(aₑ*Hₑ)**3, mᵩ, Hₑ)
                        # print(nm_R + nm_I)
                        # print(dn_i, "rho/s/T = ", ρ_s_T[0])

                        label = rf"$m_{{3/2}}={y:.2f}m_\phi$"
                        ax.plot(m, ρ_s_T, color=color, label=label)
                        
                        '''
                        #  trying to fit the first part
                        popt, pcov = curve_fit(linear_f, m[m<0.1], ρ_s_T[m<0.1])
                        perr = np.sqrt(np.diag(pcov))
                        #  print(y, popt, perr)
                        slopes = np.append(slopes, popt[0])
                        err_slopes = np.append(err_slopes, perr[0])

                        ax.plot(m, linear_f(m, *popt), alpha=0.5, color=color, ls="--")
                        '''
                    else:
                        raise(ValueError("Something went wrong!"))
            else:
                # plot every integrated.npz individually
                for i, fn_i in enumerate(full_fns):
                    # print(fn_i)
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
                    ρ_s_T = get_ρ_s_Trh(nm, mᵩ, Hₑ)
                    ax.plot(m, ρ_s_T, color=color, ls=ls, label=label)
    
    # add linear part for visual guidance
    if add:
        #  ax.plot([0.1, 1], _power_law([0.1, 1], 1e-17, 0, 1), color="grey", ls="--", label="linear")
        pass
    
    if nosugra is not None:
        ms, ρ_s_T = nosugra
        ax.plot(ms, ρ_s_T, color="k", label="no SUGRA")

    ax.set_xlabel(r"$m_\chi / m_\phi$")
    ax.set_ylabel(r"$\rho_\chi/(s_0 T_{\rm rh})$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim((0.03, 2))
    ax.set_ylim((5e-17, 2e-5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(loc='upper right')
    
    out_fn = dn.replace("data", "figs") + "integrated_comp"
    if add:
        out_fn += "_add"
    out_fn += ".pdf"

    plt.savefig(out_fn, bbox_inches="tight")
    plt.close()
    
    '''
    print("Data are", m3_2s, slopes, err_slopes)
    popt, perr = curve_fit(lambda x, a, b: a*x+b, m3_2s, slopes, sigma=err_slopes, absolute_sigma=True)
    print(popt, perr)
    print("Fitted:", m3_2s, inverse_f(m3_2s, *popt))
    fig, ax = plt.subplots()

    ax.plot(m3_2s, slopes, c="k", ls="-")
    ax.plot(m3_2s, inverse_f(m3_2s, *popt), c="k", ls="--")

    plt.savefig(out_fn.replace("comp", "slope"), bbox_inches="tight")
    plt.close()
    '''

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


def plot_m_eff(dn, a, mᵩ, which=None):
    dirs, m3_2s = _get_m3_2_dir(dn)
    # print(dirs, m3_2s)
    cmap = mpl.colormaps['magma'].reversed()

    for (d, m) in zip(dirs, m3_2s):
        if d in which:
            full_path_d = join(join(dn, d), "m_eff")
            # print(d, dn, full_path)
            fns, ms = _get_m_fn(full_path_d)
            fig, ax = plt.subplots()
        
            for (x, y) in zip(fns, ms):
                full_path_f = join(full_path_d, x)
                print(full_path_f)
                data = np.load(full_path_f)
                a = data["a"]
                N = np.log(a)
                m2_I = data["m2_I"]
                m2_R = data["m2_R"]

                color = cmap(y/max(ms))
                ax.plot(N, m2_R / a**2 / mᵩ**2, c=color, label=rf"$m_\chi={y:.2f}m_\phi$")
                ax.plot(N, m2_I / a**2 / mᵩ**2, c=color, ls="--")

            ax.set_xlabel(r"$N$")
            ax.set_ylabel(r"$m^2_{\tilde{\chi}, \rm {eff}} / (a ^2 m_\phi^2 )$")
            ax.set_yscale("log")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.legend(loc="lower left")

            plt.savefig(full_path_d.replace("data", "figs") + ".pdf", bbox_inches="tight")
            plt.close()


def cp_model_data(dn):
    Path(dn.replace("data", "figs")).mkdir(parents=True, exist_ok=True)
    src = dn+"model.dat"
    dest = src.replace("data", "figs")
    shutil.copy(src, dest)


if __name__ == "__main__":
    # TMode
    # dn = "data/TMode-0.0001/"
    # dn = "data/TMode-0.001-iso/"
    dn = "data/TMode-0.0035/"
    # dn = "data/TMode-0.001/"
    _, _, _, a, _, aₑ, Hₑ, _, H, mᵩ = read_ode(dn)
    # rho_p = 3 * H[-1]**2 * a[-1]**3
    #  print(a[50000])
    #  rho_p = a[50000]**3
    # cp_model_data(dn)
    # plot_background(dn)
    # plot_f_m3_2(dn, sparse=0.4)
    # ms, ρ = plot_integrated_nosugra(dn, mᵩ, Hₑ)
    # plot_integrated_comp(dn, aₑ, Hₑ, mᵩ, add=True, nosugra=(ms, ρ))
    plot_m_eff(dn, a, mᵩ, which=["m3_2=0.00", "m3_2=0.10"])
    
    # SmallField
    # dn = "data/SmallField/"
    #  plot_background("data/SmallField/")
    #  plot_f("data/SmallField/")
    #  plot_integrated(dn)
