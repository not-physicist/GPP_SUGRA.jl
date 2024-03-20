module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, ProgressBars, LinearInterpolations
#  using JLD2
#  using Infiltritor
#  using Interpolations

using ..Commons

"""
get ω functions ready from ODE solution
mᵩ: mass of inflaton 
mᵪ: mass of the produced particle

One is responsible to make sure ode.τ and m2_eff has the same dimension. Now ode data isshould be automatically trimmed.

LinearInterpolations uses ~ half as much as memories, a bit faster also.
"""
function init_func(k::Real, ode::ODEData, m2_eff::Vector)
    τ = ode.τ

    # NOTE: don't take the last element as upper bound of the integration;
    # since the derivatives "sacrifies" some elements
    # otherwise problematic with the ODE solver
    # and possibly outside of the interpolation range
    t_span = [τ[1], τ[end-2]]
    
    # get sampled ω values and interpolate
    ω = (k^2 .+ m2_eff).^(1/2)

    # try linear interpolation first
    #  get_ω = Interpolations.interpolate((τ,), ω, Gridded(Linear()))
    get_ω = LinearInterpolations.Interpolate(τ, ω)

    #  a simple consistency check
    #  @show "original: %f, interpolated: %f" ω[1] get_ω(τ[1])

    dω = diff(ω) ./ diff(τ) 
    #  get_dω = Interpolations.interpolate((τ[1:end-1],), dω, Gridded(Linear()))
    get_dω = LinearInterpolations.Interpolate(τ[1:end-1], dω)
    
    # cumulative integration
    Ω = cumul_integrate(τ, ω)
    #  get_Ω = Interpolations.interpolate((τ,), Ω, Gridded(Linear()))
    get_Ω = LinearInterpolations.Interpolate(τ, Ω)

    return get_ω, get_dω, get_Ω, t_span
end

"""
Defines the differential equation to solve
"""
function get_diff_eq(u, p, t)
    ω = p[1]
    dω = p[2]
    Ω = p[3]

    α = u[1]
    β = u[2]

    dω_2ω = dω(t) / (2 * ω(t))
    e = exp(+2.0im * Ω(t))
    dydt = dω_2ω .* SA[e * β, conj(e) * α]
    return dydt
end

"""
Solve the differential equations for GPP
dtmax not used, but keep just in case
NOTE: max_err is now deprecated!
"""
function solve_diff(k::Real, ode::ODEData, m2_eff::Vector, dtmax::Real=false)
    ω, dω, Ω, t_span = init_func(k, ode, m2_eff)
    p = SA[ω, dω, Ω]

    u₀ = SA[1.0 + 0.0im, 0.0 + 0.0im]

    prob = ODEProblem(get_diff_eq, u₀, t_span, p)
    #  adaptive algorithm depends on relative tolerance
    sol = solve(prob, RK4(), reltol=1e-7, abstol=1e-9, save_everystep=false, maxiters=1e7)

    res = sol.u[end]
    f = abs(res[2])^2
    max_err = abs(abs(res[1])^2 - abs(res[2])^2 - 1) 

    return f, max_err
end 

function solve_diff(k::Vector, ode::ODEData, m2_eff::Vector, dtmax::Real=false)
    f = zeros(size(k)) 
    err = zeros(size(k)) 
    #  @show f, err
    
    Threads.@threads for i in 1:length(k)
        res = solve_diff(k[i], ode, m2_eff, dtmax)
        #  @show typeof(res), res
        f[i] = res[1]
        err[i] = res[2]
    end
    
    return f, err
end

"""
compute comoving energy (a⁴ρ) given k, m2, and f arrays
"""
function get_com_energy(k::Vector, f::Vector, m2::Real)
    ω = sqrt.(k .^2 .+ m2)
    integrand = k.^2 .* ω .* f ./ (4*π^2)
    return integrate(k, integrand)
end

function get_com_number(k::Vector, f::Vector)
    integrand = k.^2 .* f ./ (4*π^2) 
    return integrate(k, integrand)
end

"""
save the spectra for various parameters (given as arguments);
use multi-threading, remember use e.g. julia -n auto
direct_out -> if return the results instead of save to npz

results data structure:
- ModelName
    - f_ξ=ξ
        m_χ=m_χ.npz

"""
function save_each(data_dir::String, mᵩ::Real, ode::ODEData, 
                   k::Vector, mᵪ::Vector, ξ::Vector, 
                   get_m2_eff::Function;
                   dtmax::Real=false, 
                   direct_out::Bool=false,
                   fn_suffix::String="")
    #  println("Computing spectra using ", Threads.nthreads(), " cores")

    # interate over the model parameters
    for ξᵢ in ξ
        ρs = zeros(size(mᵪ))
        ns = zeros(size(mᵪ))
        f0s = zeros(size(mᵪ))
        ξ_dirᵢ = data_dir * "f_ξ=$ξᵢ/"

        for (i, mᵪᵢ) in ProgressBar(enumerate(mᵪ))
            #  only want to compute this once for one set of parameters
            m2_eff = get_m2_eff(ode, mᵪᵢ, ξᵢ)
            
            f, err = solve_diff(k, ode, m2_eff, dtmax)
            
            # take the ρ at the end, use last m2_eff
            ρs[i] = get_com_energy(k, f, m2_eff[end])
            ns[i] = get_com_number(k, f)
            f0s[i] = f[1]

            if direct_out
                return f
            else
                mkpath(ξ_dirᵢ)
                npzwrite("$(ξ_dirᵢ)mᵪ=$(mᵪᵢ/mᵩ)$fn_suffix.npz",
                         Dict("k"=>k/(ode.aₑ*mᵩ), "f"=>f, "err"=>err))
                #  save("$(ξ_dirᵢ)mᵪ=$(mᵪᵢ/mᵩ)$fn_suffix.npz",
                         #  Dict("k"=>k/(ode.aₑ*mᵩ), "f"=>f, "err"=>err))
            end
        end
        # k is in planck unit
        # want ρ and n in planck unit as well
        # add all other factors in the plotting
        npzwrite("$(ξ_dirᵢ)integrated$fn_suffix.npz",
                 Dict("m_chi" =>mᵪ / mᵩ, "f0"=>f0s, "rho"=>ρs, "n"=>ns))
        #  save("$(ξ_dirᵢ)integrated$fn_suffix.jld2",
                 #  Dict("m_chi" =>mᵪ / mᵩ, "f0"=>f0s, "rho"=>ρs, "n"=>ns))
    end
end

"""
adding one more iteration over m3_2
results data structure:
- ModelName
    - m3_2=m3_2
        - f_ξ=ξ
            m_χ=m_χ.npz
"""
function save_each(data_dir::String, mᵩ::Real, ode::ODEData, 
                   k::Vector, mᵪ::Vector, ξ::Vector, 
                   m3_2::Vector,
                   get_m2_eff::Function;
                   dtmax::Real=false, 
                   direct_out::Bool=false,
                   fn_suffix::String="")
    for x in ProgressBar(m3_2)
        m3_2_dir = data_dir * "m3_2=$(x/mᵩ)/"
        m2_eff_R(ode, mᵪ, ξ) = get_m2_eff(ode, mᵪ, ξ, x)
        save_each(m3_2_dir, mᵩ, ode, k, mᵪ, ξ, m2_eff_R, 
                  dtmax=dtmax, direct_out=direct_out, fn_suffix=fn_suffix)
    end
end

end
