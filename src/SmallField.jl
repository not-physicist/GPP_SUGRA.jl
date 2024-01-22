include("common.jl")
include("ode.jl")
include("pp.jl")

"""
Model for small field inflation potential
"""
module SmallFields

using Interpolations, Folds, StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration
#  using Infiltrator

using ..ODEs
using ..Commons
using ..PPs


function get_M(v::Real, n::Int, Nₑ::Real)
    P = 2.2e-9  # observed magnitude of power spectrum
    M4 = P * 12 * pi^2 / (2 * n * ((n - 2) * Nₑ)^(n - 1))^(2 / (n - 2))
    M4 *= (v^n)^(2 / (n - 2))
    M = M4^(1 / 4)
    return M
end


function get_mᵩ(v::Real, n::Int, Nₑ::Real)
    M = get_M(v, n, Nₑ)
    mᵩ = √(2) * n * M^2 / v
    return mᵩ
end


function get_ϕₑ(v::Real, n::Int)
    return v * (v/(√2 * n))^(1 / (n-1))
end


function get_m2_eff(ode::ODEData, mᵪ::Real, ξ::Real)
    mₑ² = ode.a[1:end-2].^2 .* mᵪ^2 - (1.0 - 6.0 * ξ) .* ode.app_a
    return mₑ²
end

"""
inflation potential model parameters, in reduced Planck units
"""
struct SmallField{T<:Real, N<:Int}
    v::T
    n::N
    Nₑ::T
    M::T
    mᵩ::T
    ϕₑ::T
end
# To compute the derived quantities automatically
SmallField(v, n, Nₑ) = SmallField(v, n, Nₑ, get_M(v, n, Nₑ), get_mᵩ(v, n, Nₑ), get_ϕₑ(v, n))

"""
get the inflation potential
"""
function get_V(x::Float64, model::SmallField)
    # property destructuring; requires at least julia 1.7
    (;v, n, Nₑ, M, mᵩ, ϕₑ) = model
    return M^4 * (1 - (x/v)^n)^2
end

"""
get the derivative of inflation potential
"""
function get_dV(x::Float64, model::SmallField)
    (;v, n, Nₑ, M, mᵩ, ϕₑ) = model
    return M^4 / v * 2.0 * n * (x/v)^(n-1) * (-1.0 + (x/v)^n)
end

"""
"inflationary" scale; may well be model depedent
"""
function get_Hinf(model::SmallField)
    return √(get_V(0.0, model) / 3.0)
end

"""
get scale factor and conformal time at the end of inflation
"""
function get_end(ϕ::Vector, a::Vector, τ::Vector, model::SmallField)
    # to ensure ϕ is monotonic increasing; 0.4 should be fine for (v=0.5) now
    # TODO: can be improved by checking sign of dϕ
    (;v, n, Nₑ, M, mᵩ, ϕₑ) = model
    τ = τ[ϕ .< 0.4]
    a = a[ϕ .< 0.4]
    ϕ = ϕ[ϕ .< 0.4]

    itp = interpolate((ϕ,), τ, Gridded(Linear()))
    τₑ = itp(ϕₑ)

    itp = interpolate((τ,), a, Gridded(Linear()))
    aₑ = itp(τₑ)
    #  print("$ϕₑ, $τₑ, $aₑ \n")
    return τₑ, aₑ
end

"""
Calculate the background quantities and save to data/ode.npz
"""
function save_ode()
    model = SmallField(0.5, 6, 60.0)
    (;v, n, Nₑ, M, mᵩ, ϕₑ) = model
    
    # initial conditions
    u₀ = SA[0.4*ϕₑ, 0.0, 1.0]
    τᵢ = - 1 / get_Hinf(model)
    tspan = (τᵢ, -τᵢ) 
    
    # initiate parameter (functions)
    _get_V(x) = get_V(x, model)
    _get_dV(x) = get_dV(x, model)
    p = (_get_V, _get_dV)
    
    τ, ϕ, dϕ, a, ap, app, app_a = @time ODEs.solve_ode(u₀, tspan, p)

    err = ODEs.get_err(app, a[1:end-2], ϕ[1:end-2], dϕ[1:end-2], _get_V)
    τₑ, aₑ = get_end(ϕ, a, τ, model)
    
    if !isdir("data")
        mkdir("data") 
    end
    npzwrite("data/ode.npz", Dict("tau"=>τ, "phi"=>ϕ, "phi_d"=>dϕ, "a"=>a, "app_a"=>app_a, "err"=>err, "a_end"=>aₑ))
    return true
end

"""
save the spectra for various parameters;
use multi-threading, remember use e.g. julia -n auto
"""
function save_f()
    println("Computing spectra using ", Threads.nthreads(), " cores")
    model = SmallField(0.5, 6, 60.0)
    ode = read_ode()
    (;v, n, Nₑ, M, mᵩ, ϕₑ) = model

    k = Commons.logspace(-1, 1, 100)
    mᵪ = [0.2, 0.5, 1.0, 2.0] .* mᵩ
    ξ = [1.0 / 6.0, 0.0]
    #  mᵪ = [0.2] .* mᵩ
    #  ξ = [1.0 / 6.0]
    ξ_dir = ["data/f_ξ=1_6/", "data/f_ξ=0/" ]

    # interate over the model parameters
    for (i, ξᵢ) in enumerate(ξ)
        for mᵪ_i in mᵪ
            #  @printf "ξ = %f, mᵪ = %f \t" ξᵢ mᵪ_i/mᵩ
            m2_eff = get_m2_eff(ode, mᵪ_i, ξᵢ)
            
            # Folds.collect is the multi-threaded version of collect
            res = @time Folds.collect(PPs.solve_diff(x, mᵩ, ode, m2_eff) for x in k)
            # maybe some optimization is possible here...
            f = [x[1] for x in res]
            err = [x[2] for x in res]
            #  @infiltrate

            if !isdir(ξ_dir[i])
                mkdir(ξ_dir[i]) 
            end
            npzwrite("$(ξ_dir[i])mᵪ=$(mᵪ_i/mᵩ).npz", Dict("k"=>k, "f"=>f, "err"=>err))
        end
    end
end

"""
save the spectra for one set of parameters; just for testing
"""
function test_save_f()
    model = SmallField(0.5, 6, 60.0)
    (;v, n, Nₑ, M, mᵩ, ϕₑ) = model
    ode = read_ode()

    k = Commons.logspace(-1, 1, 5)
    mᵪ = 1.0 .* mᵩ
    ξ = 1.0 / 6.0
    ξ_dir = "data/f_ξ=1_6/"
    
    m2_eff = get_m2_eff(ode, mᵪ, ξ)

    res = @time Folds.collect(PPs.solve_diff(x, mᵩ, ode, m2_eff) for x in k)
    f = [x[1] for x in res]
    err = [x[2] for x in res]
    #  @show f err

    if !isdir(ξ_dir)
        mkdir(ξ_dir) 
    end
    npzwrite("$(ξ_dir)mᵪ=$(mᵪ/mᵩ).npz", Dict("k"=>k, "f"=>f, "err"=>err))

    # approximate the true values
    if isapprox(f, [1.7891387330706488e-6, 1.3922591876374687e-7, 1.2272875358428686e-7, 2.1205741410295525e-10, 2.2278489765847522e-11], rtol=1e-3)
        f = 
        return true
    else
        return false
    end
end

end
