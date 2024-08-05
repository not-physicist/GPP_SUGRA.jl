module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, ProgressBars, LinearInterpolations, Printf
#  using JLD2
#  using Infiltritor
 using Interpolations

using ..Commons
using ..Isocurvature

# type of interpolator...
# const INTERP_TYPE = LinearInterpolations.Interpolate{typeof(LinearInterpolations.combine), Tuple{Vector{Float64}}, Vector{Float64}, Symbol}
#  const INTERP_TYPE = Interpolations.GriddedInterpolation{Float64, 1, Vector{Float64}, Interpolations.Gridded{Interpolations.Linear{Interpolations.Throw{Interpolations.OnGrid}}}, Tuple{Vector{Float64}}}

"""
get ω functions ready from ODE solution
mᵩ: mass of inflaton 
mᵪ: mass of the produced particle

One is responsible to make sure ode.τ and m2_eff has the same dimension. Now ode data isshould be automatically trimmed.

LinearInterpolations uses ~ half as much as memories, a bit faster also.
"""
function init_Ω(k::Real, τ::Vector, m2::Vector)
    # get sampled ω values and interpolate
    ω = @. (k^2 + m2)^(1/2)
    # cumulative integration
    Ω = cumul_integrate(τ, ω)
    get_Ω = LinearInterpolations.Interpolate(τ, Ω, extrapolate=LinearInterpolations.Constant(0.0))
    # get_Ω = Interpolations.interpolate((τ,), Ω, Gridded(Linear()))

    return get_Ω
end

"""
get the parameters (interpolators) for diff_eq
"""
function get_p(k::Real, get_m2::T, get_dm2::T, Ω::T) where {T <: LinearInterpolations.Interpolate}
# function get_p(k::Real, get_m2::T, get_dm2::T, Ω::T) where {T <: Interpolations.GriddedInterpolation}
    ω = x -> sqrt(k^2 + get_m2(x))
    dω = x -> get_dm2(x) / (2*ω(x))
    #  @show typeof(ω) typeof(dω) typeof(Ω)
    return ω, dω, Ω
end

"""
Defines the differential equation to solve
"""
function get_diff_eq(u::SVector, p::Tuple, t::Real)
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
"""
function solve_diff(k::Real, t_span::Vector, get_m2::T, get_dm2::T, Ω::T) where {T <: LinearInterpolations.Interpolate}
# function solve_diff(k::Real, t_span::Vector, get_m2::T, get_dm2::T, Ω::T) where {T <: Interpolations.GriddedInterpolation}
    p = get_p(k, get_m2, get_dm2, Ω)
    # t_span = [τ[1], τ[end-2]]

    u₀ = SA[1.0 + 0.0im, 0.0 + 0.0im]
    
    # false: out of place function for ODEs
    prob = ODEProblem{false}(get_diff_eq, u₀, t_span, p)
    #  adaptive algorithm depends on relative tolerance
    sol = solve(prob, RK4(), reltol=1e-7, abstol=1e-9, save_everystep=false, maxiters=1e8)

    αₑ = sol[1, end]
    βₑ = sol[2, end]
    ωₑ = p[1](sol.t[end])
    
    δt = (sol.t[end] - sol.t[end-1])
    δαₑ = (sol[1, end] - sol[1, end-1] ) / δt
    δβₑ = (sol[2, end] - sol[2, end-1] ) / δt
    δωₑ = (p[1](sol.t[end]) - p[1](sol.t[end-1]) ) / δt

    return αₑ, βₑ, ωₑ, δαₑ, δβₑ, δωₑ
end 

"""
parallelized version for multiple momentum k 
"""
function solve_diff(k::Vector, t_span::Vector, get_m2::T, get_dm2::T, Ω::Vector{T}) where {T <: LinearInterpolations.Interpolate}
# function solve_diff(k::Vector, t_span::Vector, get_m2::T, get_dm2::T, Ω::Vector{T}) where {T <: Interpolations.GriddedInterpolation}
    α = zeros(ComplexF64, size(k)) 
    β = zeros(ComplexF64, size(k)) 
    ω = zeros(ComplexF64, size(k)) 
    δα = zeros(ComplexF64, size(k)) 
    δβ = zeros(ComplexF64, size(k)) 
    δω = zeros(ComplexF64, size(k)) 

    #  err = zeros(size(k))
    
    @inbounds Threads.@threads for i in eachindex(k)
        @inbounds res = solve_diff(k[i], t_span, get_m2, get_dm2, Ω[i])
        #  @show typeof(res), res
        @inbounds α[i] = res[1]
        @inbounds β[i] = res[2]
        @inbounds ω[i] = res[3]
        @inbounds δα[i] = res[4]
        @inbounds δβ[i] = res[5]
        @inbounds δω[i] = res[6]
    end
    
    #  return f, err
    return α, β, ω, δα, δβ, δω
end

###########################################################################
# test ensemble problem solver, seems slower...

# function test_ensemble(k::Vector, ode::ODEData, get_m2_eff::Function, mᵪ::Real, ξ::Real)
#     m2_eff = get_m2_eff(ode, mᵪ, ξ)
#     get_m2 = LinearInterpolations.Interpolate(ode.τ, m2_eff)
#     get_dm2 = LinearInterpolations.Interpolate(ode.τ[1:end-1], diff(m2_eff) ./ diff(ode.τ))
#
#     @show solve_diff_ensemble(k, ode.τ, m2_eff, get_m2, get_dm2)[:, 1]
#     #  @time solve_diff(k, ode.τ, m2_eff, get_m2, get_dm2)
#     return true
# end

###########################################################################
#
"""
comoving energy (a⁴ρ) given k, m2, and f arrays
"""
function get_com_energy(k::Vector, f::Vector, ω::Vector)
    # ω = sqrt.(@. k^2 + m2)
    integrand = @. k^2 * ω * f / (4*π^2)
    return integrate(k, integrand)
end

"""
comoving number density (a³n) given k, m2, and f arrays
"""
function get_com_number(k::Vector, f::Vector)
    integrand = @. k^2 * f / (4*π^2) 
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
                   k::Vector, mᵪ::SVector, ξ::SVector, 
                   get_m2_eff::Function;
                   direct_out::Bool=false,
                   fn_suffix::String="")
    # interate over the model parameters
    for ξᵢ in ξ
        ρs = zeros(MVector{size(mᵪ)[1]})
        ns = zeros(MVector{size(mᵪ)[1]})
        f0s = zeros(MVector{size(mᵪ)[1]})
        ξ_dirᵢ = data_dir * "f_ξ=$ξᵢ/"
        
        iter = ProgressBar(eachindex(mᵪ))
        @inbounds for i in iter
            @inbounds mᵪᵢ = mᵪ[i]
            set_description(iter, ("mᵪ: $(@sprintf("%.2f", mᵪᵢ))"))
            
            fn_out = "$(ξ_dirᵢ)mᵪ=$(mᵪᵢ/mᵩ)$fn_suffix.npz"
            # skip if file already exists (skip the outer loop as well!)
            # if isfile(fn_out)
            #     println("File exists: ", fn_out, " SKIPPING")
            #     return 
            # end
            
            #  only want to compute this once for one set of parameters
            m2_eff = get_m2_eff(ode, mᵪᵢ, ξᵢ)
            get_m2 = LinearInterpolations.Interpolate(ode.τ, m2_eff, extrapolate=LinearInterpolations.Constant(0.0))
            # get_m2 = Interpolations.interpolate((ode.τ,), m2_eff, Gridded(Linear()))
            get_dm2 = LinearInterpolations.Interpolate(ode.τ[1:end-1], diff(m2_eff) ./ diff(ode.τ), extrapolate=LinearInterpolations.Constant(0.0))
            # get_dm2 = Interpolations.interpolate((ode.τ[1:end-1],), diff(m2_eff) ./ diff(ode.τ), Gridded(Linear()))
            # want to caculate Ω here already, instead of in the inner loop
            Ω = Array{Interpolate, 1}(undef, size(k))
            # Ω = Array{Interpolations.GriddedInterpolation, 1}(undef, size(k))
            @inbounds Threads.@threads for i in eachindex(k)
                @inbounds Ω[i] = init_Ω(k[i], ode.τ, m2_eff)
            end
            # @show typeof(Ω)
                
            t_span = [ode.τ[1], ode.τ[end-2]]
            α, β, ω, δα, δβ, δω = solve_diff(k, t_span, get_m2, get_dm2, Ω)
            # err = @. abs2(α) - abs2(β)
            # @show err
            f = abs2.(β)

            # take the ρ at the end, use last m2_eff
            @inbounds ρs[i] = get_com_energy(k, f, ω)
            @inbounds ns[i] = get_com_number(k, f)
            @inbounds f0s[i] = f[1]
            
            # for isocurvature calculation, need Ω only at the end 
            # interpolate for k
            Ω_new = [x(ode.τ[end-2]) for x in Ω]
            # @show typeof(Ω_new)
            Δ2 = get_Δ2_χ(k, α, β, Ω_new, ω, ρs[i], ode.a[end], mᵪᵢ^2, mᵩ)
            # Δ2 = get_Δ2(k, α, β, Ω_new, ρs[i], ode.a[end], m2_eff[end])

            if direct_out
                return f
            else
                mkpath(ξ_dirᵢ)
                # k is in planck unit
                npzwrite(fn_out, Dict("k"=>k/(ode.aₑ*mᵩ), "f"=>f, "Delta2"=>Δ2))
            end
        end
        # k is in planck unit
        # want ρ and n in planck unit as well
        # add all other factors in the plotting
        #  @show ξ_dirᵢ fn_suffix
        #  @show typeof(mᵪ / mᵩ) typeof(f0s) typeof(ρs) typeof(ns)
        npzwrite("$(ξ_dirᵢ)integrated$fn_suffix.npz",
                Dict("m_chi" => [mᵪ / mᵩ ...], "f0"=>[f0s...], "rho"=>[ρs...], "n"=>[ns...]))
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
                   k::Vector, mᵪ::SVector, ξ::SVector, 
                   m3_2::SVector,
                   get_m2_eff::Function;
                   direct_out::Bool=false,
                   fn_suffix::String="")
    iter = ProgressBar(eachindex(m3_2))
    for i in iter
        @inbounds m = m3_2[i]
        set_description(iter, "m_32: $(@sprintf("%.2f", m))",)
        m3_2_dir = data_dir * "m3_2=$(m/mᵩ)/"
        m2_eff_R(ode, mᵪ, ξ) = get_m2_eff(ode, mᵪ, ξ, m)
        save_each(m3_2_dir, mᵩ, ode, k, mᵪ, ξ, m2_eff_R, 
                  direct_out=direct_out, fn_suffix=fn_suffix)
    end
end

end
