module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, ProgressBars, LinearInterpolations, Printf
#  using JLD2
#  using Infiltritor
 using Interpolations

using ..Commons
using ..Isocurvature


"""
get ω functions ready from ODE solution
mᵩ: mass of inflaton 
mᵪ: mass of the produced particle

One is responsible to make sure ode.τ and m2_eff has the same dimension. Now ode data isshould be automatically trimmed.

LinearInterpolations uses ~ half as much as memories, a bit faster also.
"""
function init_Ω(k::Real, τ::Vector, m2::Vector)
    # get sampled ω values and interpolate
    # allow complex ω, or negative ω^2
    ω = @. sqrt(Complex(k^2 + m2))
    # cumulative integration
    Ω = cumul_integrate(τ, ω)
    get_Ω = LinearInterpolations.Interpolate(τ, Ω, extrapolate=LinearInterpolations.Constant(0.0))
    # get_Ω = Interpolations.interpolate((τ,), Ω, Gridded(Linear()))

    return get_Ω
end

"""
get the parameters (interpolators) for diff_eq
"""
function get_p(k::Real, get_m2::T, get_dm2::T, Ω::U) where {T , U <: LinearInterpolations.Interpolate}
# function get_p(k::Real, get_m2::T, get_dm2::T, Ω::T) where {T <: Interpolations.GriddedInterpolation}
    ω = x -> sqrt(Complex(k^2 + get_m2(x)))
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
function solve_diff(k::Real, t_span::Vector, get_m2::T, get_dm2::T, Ω::U) where {T, U <: LinearInterpolations.Interpolate}
# function solve_diff(k::Real, t_span::Vector, get_m2::T, get_dm2::T, Ω::T) where {T <: Interpolations.GriddedInterpolation}
    p = get_p(k, get_m2, get_dm2, Ω)
    # @show p[1](t_span[1])
    # t_span = [τ[1], τ[end-2]]

    u₀ = SA[1.0 + 0.0im, 0.0 + 0.0im]
    
    # false: out of place function for ODEs
    prob = ODEProblem{false}(get_diff_eq, u₀, t_span, p)
    #  adaptive algorithm depends on relative tolerance
    # sol = solve(prob, Tsit5(), reltol=1e-7, abstol=1e-9, save_everystep=false, maxiters=1e8)
    sol = solve(prob, Rosenbrock23(autodiff=false), reltol=1e-7, abstol=1e-9, save_everystep=false, maxiters=1e8)

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
# solve mode functions directly

function get_diff_eq_mode(u, ω2, t)
    χ = u[1]
    ∂χ = u[2]

    return @SVector [∂χ, -ω2(t)*χ]
end

function solve_diff_mode(k::Real, t_span::Vector, get_m2::T) where {T <: LinearInterpolations.Interpolate}
    get_ω2 = x -> k^2 + get_m2(x)
    ω₀ = sqrt(get_ω2(t_span[1])+0.0im)
    ωₑ = sqrt(get_ω2(t_span[2])+0.0im)

    # u₀ = @SVector [1/sqrt(2*k), -1.0im*k/sqrt(2*k)] 
    u₀ = @SVector [1/sqrt(2*ω₀), -1.0im*ω₀/sqrt(2*ω₀)] 

    prob = ODEProblem{false}(get_diff_eq_mode, u₀, t_span, get_ω2)
    sol = solve(prob, Vern9(), reltol=1e-9, abstol=1e-12, save_everystep=false, maxiters=1e8)
    # sol = solve(prob, Rosenbrock23(autodiff=false), reltol=1e-6, abstol=1e-9, save_everystep=false, maxiters=1e8)

    χₑ = sol[1, end]
    ∂χₑ = sol[2, end]

    # wronskian
    err = 1 + 1.0im * (χₑ*conj(∂χₑ) - conj(χₑ) * ∂χₑ)

    return χₑ, ∂χₑ, ωₑ, err
end

"""
f = |β|^2 from mode functions χₖ, ∂χₖ
"""
function _get_f(ω, χ, ∂χ)
    return abs2(ω * χ - 1.0im * ∂χ)/(2*ω)
end

function solve_diff_mode(k::Vector, t_span::Vector, get_m2::T) where {T <: LinearInterpolations.Interpolate}
    χ = zeros(ComplexF64, size(k))
    ∂χ = zeros(ComplexF64, size(k))
    ω = zeros(ComplexF64, size(k))
    f = zeros(size(k))
    err = zeros(size(k))

    @inbounds Threads.@threads for i in eachindex(k)
        @inbounds res = solve_diff_mode(k[i], t_span, get_m2)
        @inbounds χ[i], ∂χ[i], ω[i], err[i] = res
        @inbounds f[i] = _get_f(ω[i], χ[i], ∂χ[i])
    end
    
    # @show f
    return f, ω, χ, ∂χ, err
end

"""
save mode functions at all times
"""
function solve_diff_mode_all(k::Real, t_span::Vector, get_m2::T) where {T <: LinearInterpolations.Interpolate}
    get_ω2 = x -> k^2 + get_m2(x)

    # u₀ = @SVector [1/sqrt(2*k), -1.0im*k/sqrt(2*k)] 
    ω₀ = sqrt(get_ω2(t_span[1])+0.0im)
    u₀ = @SVector [1/sqrt(2*ω₀), -1.0im*ω₀/sqrt(2*ω₀)] 

    prob = ODEProblem{false}(get_diff_eq_mode, u₀, t_span, get_ω2)
    sol = solve(prob, Tsit5(), reltol=1e-9, abstol=1e-12, save_everystep=true, maxiters=1e8)
    # sol = solve(prob, Rosenbrock23(autodiff=false), reltol=1e-6, abstol=1e-9, save_everystep=true, maxiters=1e8)
    ω = [sqrt(get_ω2(x) + 0.0im) for x in sol.t]
    return sol.t, sol[1, :], sol[2, :], ω
end

function solve_diff_mode_all(k::Vector, t_span::Vector, get_m2::T, dn::String) where {T <: LinearInterpolations.Interpolate}
    ω = zeros(ComplexF64, size(k))
    f = zeros(size(k))
    χ_k = zeros(ComplexF64, size(k))
    ∂χ_k = zeros(ComplexF64, size(k))
    err_k = zeros(size(k))

    @inbounds Threads.@threads for i in eachindex(k)
        @inbounds res = solve_diff_mode_all(k[i], t_span, get_m2)
        # @show res[4][1:10]
        # η_all = res[1]
        χ = res[2]
        ∂χ = res[3]
        # ω_all = res[4]
        # @show sizeof(res[1]), sizeof(res[2])
        
        f_all = @. _get_f(res[4], res[2], res[3])
        err = abs.(1 .+ 1.0im .* (χ.*conj.(∂χ) .- conj.(χ) .* ∂χ))
        
        fn = dn * "/k=$(k[i]).npz"
        @show fn
        mkpath(dn)
        npzwrite(fn, Dict("eta"=>res[1], "f"=>f_all, "err"=>err, "chi"=>χ))

        f[i] = f_all[end]
        ω[i] = res[4][end]
        err_k[i] = findmax(err)[1]
        χ_k[i] = χ[end]
        ∂χ_k[i] = ∂χ[end]
    end
    
    # @show f
    return f, ω, χ_k, ∂χ_k, err_k
end

###########################################################################
"""
comoving energy (a⁴ρ) given k, m2, and f arrays
"""
function get_com_energy(k::Vector, f::Vector, ω::Vector)
    # @show k, f, ω
    integrand = @. k^2 * ω * f / (2*π^2)
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

get_m2_eff should take three parameters: ode, mᵪ, ξ
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
            get_m2 = LinearInterpolations.Interpolate(ode.τ, m2_eff)
            # get_m2 = Interpolations.interpolate((ode.τ,), m2_eff, Gridded(Linear()))
            # get_dm2 = LinearInterpolations.Interpolate(ode.τ[1:end-1], diff(m2_eff) ./ diff(ode.τ), extrapolate=LinearInterpolations.Constant(0.0))
            # get_dm2 = Interpolations.interpolate((ode.τ[1:end-1],), diff(m2_eff) ./ diff(ode.τ), Gridded(Linear()))
            
            # want to caculate Ω here already, instead of in the inner loop
            # Ω = Array{Interpolate, 1}(undef, size(k))
            # Ω = Array{Interpolations.GriddedInterpolation, 1}(undef, size(k))
            # @show k[1]^2, k[end]^2, m2_eff[1], m2_eff[10]
            # @inbounds Threads.@threads for i in eachindex(k)
            #     @inbounds Ω[i] = init_Ω(k[i], ode.τ, m2_eff)
            # end
            # @show typeof(Ω)
                
            t_span = [ode.τ[1], ode.τ[end-2]]
            # α, β, ω, δα, δβ, δω = solve_diff(k, t_span, get_m2, get_dm2, Ω)
            # err = @. abs2(α) - abs2(β)
            # @show err
            # f = abs2.(β)
            
            # f, ω, χ, ∂χ, err = solve_diff_mode(k, t_span, get_m2)
            f, ω, χ, ∂χ, err = solve_diff_mode_all(k, t_span, get_m2, "$(ξ_dirᵢ)mᵪ=$(mᵪᵢ/mᵩ)$fn_suffix")
            # @show f

            # take the ρ at the end, use last m2_eff
            @inbounds ρs[i] = get_com_energy(k, f, ω)
            @inbounds ns[i] = get_com_number(k, f)
            @inbounds f0s[i] = f[1]
            
            # for isocurvature calculation, need Ω only at the end 
            # interpolate for k
            # Ω_new = [x(ode.τ[end-2]) for x in Ω]
            # @show typeof(Ω_new)
            # Δ2 = get_Δ2_dis(k, α, β, Ω_new, ρs[i], ode.a[end], mᵪᵢ^2)
            # Δ2 = get_Δ2(k, α, β, Ω_new, ρs[i], ode.a[end], m2_eff[end])
            Δ2 = get_Δ2_χ(k, χ, ∂χ, ρs[i], ode.a[end], mᵪᵢ)
            Δ2_β = get_Δ2_only_β(k, f, ρs[i], ode.a[end], mᵪᵢ)

            if direct_out
                return f
            else
                mkpath(ξ_dirᵢ)
                # k is in planck unit
                npzwrite(fn_out, Dict("k"=>k/(ode.aₑ*mᵩ), "f"=>f, "Delta2"=>Δ2, "Delta2_beta"=>Δ2_β, "err"=>err))
                # npzwrite(fn_out, Dict("k"=>k/(ode.aₑ*mᵩ), "f"=>f))
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
