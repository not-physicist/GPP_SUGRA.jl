module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, ProgressBars, LinearInterpolations
#  using JLD2
#  using Infiltritor
#  using Interpolations

using ..Commons

# type of interpolator...
const INTERPOLATOR_TYPE = LinearInterpolations.Interpolate{typeof(LinearInterpolations.combine), Tuple{Vector{Float64}}, Vector{Float64}, Symbol}

"""
get ω functions ready from ODE solution
mᵩ: mass of inflaton 
mᵪ: mass of the produced particle

One is responsible to make sure ode.τ and m2_eff has the same dimension. Now ode data isshould be automatically trimmed.

LinearInterpolations uses ~ half as much as memories, a bit faster also.
"""
function init_Ω(k::Real, ode::ODEData, m2::INTERPOLATOR_TYPE)
    # get sampled ω values and interpolate
    # TODO: maybe one can pass m2 vector instead of interpolator; maybe faster
    ω = (k^2 .+ m2.(ode.τ)).^(1/2)
    # cumulative integration
    Ω = cumul_integrate(ode.τ, ω)
    get_Ω = LinearInterpolations.Interpolate(ode.τ, Ω)

    return get_Ω
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
function solve_diff(k::Real, ode::ODEData, m2::INTERPOLATOR_TYPE, dm2::INTERPOLATOR_TYPE)
    Ω = init_Ω(k, ode, m2)
    ω = x -> sqrt(k^2 + m2(x))
    dω = x -> dm2(x) / (2*ω(x))
    #  @show typeof(ω) typeof(dω) typeof(Ω)
    p = (ω, dω, Ω)
    t_span = [ode.τ[1], ode.τ[end-2]]

    u₀ = SA[1.0 + 0.0im, 0.0 + 0.0im]

    prob = ODEProblem(get_diff_eq, u₀, t_span, p)
    #  adaptive algorithm depends on relative tolerance
    sol = solve(prob, RK4(), reltol=1e-7, abstol=1e-9, save_everystep=false, maxiters=1e7)

    res = sol.u[end]
    f = abs(res[2])^2
    max_err = abs(abs(res[1])^2 - abs(res[2])^2 - 1) 

    return f, max_err
end 

function solve_diff(k::Vector, ode::ODEData, m2::INTERPOLATOR_TYPE, dm2::INTERPOLATOR_TYPE)
    f = zeros(size(k)) 
    err = zeros(size(k)) 
    
    Threads.@threads for i in 1:length(k)
        res = solve_diff(k[i], ode, m2, dm2)
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
                   direct_out::Bool=false,
                   fn_suffix::String="")
    # interate over the model parameters
    for ξᵢ in ξ
        ρs = zeros(size(mᵪ))
        ns = zeros(size(mᵪ))
        f0s = zeros(size(mᵪ))
        ξ_dirᵢ = data_dir * "f_ξ=$ξᵢ/"
        
        iter = ProgressBar(enumerate(mᵪ))
        for (i, mᵪᵢ) in iter
            set_description(iter, string(@sprintf("mᵪ:%.2f", mᵪᵢ)))
            #  only want to compute this once for one set of parameters
            m2_eff = get_m2_eff(ode, mᵪᵢ, ξᵢ)
            get_m2 = LinearInterpolations.Interpolate(ode.τ, m2_eff)
            # dm2 = d(m^2)/dτ
            get_dm2 = LinearInterpolations.Interpolate(ode.τ[1:end-1], diff(m2_eff) ./ diff(ode.τ))
            
            f, err = solve_diff(k, ode, get_m2, get_dm2)
            
            # take the ρ at the end, use last m2_eff
            ρs[i] = get_com_energy(k, f, m2_eff[end-2])
            ns[i] = get_com_number(k, f)
            f0s[i] = f[1]

            if direct_out
                return f
            else
                mkpath(ξ_dirᵢ)
                npzwrite("$(ξ_dirᵢ)mᵪ=$(mᵪᵢ/mᵩ)$fn_suffix.npz",
                         Dict("k"=>k/(ode.aₑ*mᵩ), "f"=>f, "err"=>err))
            end
        end
        # k is in planck unit
        # want ρ and n in planck unit as well
        # add all other factors in the plotting
        npzwrite("$(ξ_dirᵢ)integrated$fn_suffix.npz",
                 Dict("m_chi" =>mᵪ / mᵩ, "f0"=>f0s, "rho"=>ρs, "n"=>ns))
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
                   direct_out::Bool=false,
                   fn_suffix::String="")
    iter = ProgressBar(m3_2)
    for x in iter
        set_description(iter, string(@sprintf("m_32:%.2f", x)))
        m3_2_dir = data_dir * "m3_2=$(x/mᵩ)/"
        m2_eff_R(ode, mᵪ, ξ) = get_m2_eff(ode, mᵪ, ξ, x)
        save_each(m3_2_dir, mᵩ, ode, k, mᵪ, ξ, m2_eff_R, 
                  direct_out=direct_out, fn_suffix=fn_suffix)
    end
end

end
