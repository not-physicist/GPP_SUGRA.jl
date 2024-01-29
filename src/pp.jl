module PPs

using StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration, Interpolations, Folds
#  using Infiltritor

using ..Commons

"""
get ω functions ready from ODE solution
mᵩ: mass of inflaton 
mᵪ: mass of the produced particle
"""
function init_func(k::Real, mᵩ::Real, ode::ODEData, m2_eff::Vector)
    #  τ, ϕ, dϕ, a, app_a, err, aₑ = read_ode()
    #  odedata = read_ode()
    τ = ode.τ
    aₑ = ode.aₑ

    # NOTE: don't take the last element as upper bound of the integration;
    # since the derivatives "sacrifies" some elements
    # otherwise problematic with the ODE solver
    t_span = [τ[1], τ[end-3]]
    k *= aₑ * mᵩ
    
    # get sampled ω values and interpolate
    #  mₑ² = a[1:end-2].^2 .* mᵪ^2 - (1.0 - 6.0 * ξ) .* app_a
    mₑ² = m2_eff
    ω² = k^2 .+ mₑ²
    ω = (ω²).^(1/2)
    # try linear interpolation first
    get_ω = interpolate((τ[1:end-2],), ω, Gridded(Linear()))
    #  a simple consistency check
    #  @show "original: %f, interpolated: %f" ω[1] get_ω(τ[1])

    #  dω = diff(τ[1:end-2], ω)
    dω = diff(ω) ./ diff(τ[1:end-2]) 
    get_dω = interpolate((τ[1:end-3],), dω, Gridded(Linear()))
    
    # cumulative integration
    Ω = cumul_integrate(τ[1:end-2], ω)
    get_Ω = interpolate((τ[1:end-2],), Ω, Gridded(Linear()))

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
"""
function solve_diff(k::Real, mᵩ::Real, ode::ODEData, m2_eff::Vector)
    ω, dω, Ω, t_span = init_func(k, mᵩ, ode, m2_eff)
    p = SA[ω, dω, Ω]

    u₀ = SA[1.0 + 0.0im, 0.0 + 0.0im]

    prob = ODEProblem(get_diff_eq, u₀, t_span, p)
    sol = solve(prob, DP8(), dtmax=1e2)
    #  sol = solve(prob, Rodas5P(autodiff=false), dtmax=1e2)
        
    f = abs(sol.u[end][2])^2
    max_err = maximum([abs(abs(x[1])^2 - abs(x[2])^2 - 1) for x in sol.u])

    return f, max_err
    #=
    # not efficient memoery usage; may cause problem
    # (locate memories for too many useless arrays)
    #  τ = sol.t
    α = [x[1] for x in sol.u]
    β = [x[2] for x in sol.u]

    err = [abs(x)^2 for x in α] .- [abs(x)^2 for x in β] .- 1
    err = [abs(x) for x in err]

    f = abs(β[end])^2
    max_err = maximum(err) 

    # "free" the memory
    # usually not necessary; but seems to be for the AMD system
    # slows down the function
    sol = nothing
    α = nothing
    β = nothing
    err = nothing
    GC.gc(true)
    =#
end 

"""
save the spectra for various parameters (given as arguments);
use multi-threading, remember use e.g. julia -n auto
direct_out -> if return the results instead of save to npz
"""
function save_each(mᵩ::Real, ode::ODEData, k::Vector, mᵪ::Vector,
                   ξ::Vector, ξ_dir::Vector, get_m2_eff::Function,
                   direct_out::Bool=false)
    println("Computing spectra using ", Threads.nthreads(), " cores")

    # interate over the model parameters
    for (ξᵢ, ξ_dirᵢ) in zip(ξ, ξ_dir)
        for mᵪᵢ in mᵪ
            #  @printf "ξ = %f, mᵪ = %f \t" ξᵢ mᵪ_i/mᵩ
            #  only want to compute this once for one set of parameters
            m2_eff = get_m2_eff(ode, mᵪᵢ, ξᵢ)
            
            # Folds.collect is the multi-threaded version of collect
            res = @time Folds.collect(solve_diff(x, mᵩ, ode, m2_eff) for x in k)
            # maybe some optimization is possible here...
            f = [x[1] for x in res]
            err = [x[2] for x in res]
            #  @infiltrate
            
            if direct_out
                return f
            else
                if !isdir(ξ_dirᵢ)
                    mkdir(ξ_dirᵢ) 
                end
                npzwrite("$(ξ_dirᵢ)mᵪ=$(mᵪᵢ/mᵩ).npz", Dict("k"=>k, "f"=>f, "err"=>err))
            end
        end
    end
end

end
