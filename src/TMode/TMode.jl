"""
Model for T-mode α attractor inflation potential
"""
module TModes

# submodules
include("ModelData.jl")
using .ModelDatas

using ..ODEs
using ..Commons
using ..PPs

using StaticArrays, NPZ

# global constant
const MODEL_NAME="TMode"
const MODEL_DATA_DIR="data/$MODEL_NAME/"

"""
the f-function in superpotential
"""
function get_V(ϕ::Real, model::TMode)
    x = ϕ / (sqrt(6) * model.α)
    return model.V₀ * tanh(x)^(2*model.n)
end

function get_dV(ϕ::Real, model::TMode)
    x = ϕ / (sqrt(6) * model.α)
    return sqrt(2/3) / model.α * model.V₀ * sech(x)^2 * tanh(x)^(2*model.n-1)
end

"""
define inflationary scale like this
"""
function get_Hinf(model::TMode)
    return √(model.V₀ / 3.0)
end

"""
the f-function in super potential
"""
function get_f(ϕ::Vector, model::TMode, m3_2::Real)
    x = ϕ ./ (sqrt(6 * model.α))
    if model.n == 1
        return sqrt(3*model.α*model.V₀) .* log.(cosh.(x)) .+ m3_2/sqrt(3)
    else
        throw(ErrorException("This n is yet to be implemented!"))
    end
end

"""
effective mass squared of the real field
"""
function get_m2_eff_R(ode::ODEData, mᵪ::Real, ξ::Real, f::Vector)
    m2 = ode.a.^2 .* (mᵪ^2 .+ ode.H .^2 .+ 2 .* f.^2 .+ mᵪ.*f)
    return m2
end

function save_ode(data_dir::String=MODEL_DATA_DIR)
    mkpath(data_dir)

    model = TMode(1, 0.965, 0.001)
    @show model

    # initial conditions
    u₀ = SA[1.5 * model.ϕₑ, 0.0, 1.0]
    τᵢ = - 1 / get_Hinf(model)
    tspan = (τᵢ, -τᵢ)

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    p = (_V, _dV)
    
    τ, ϕ, dϕ, a, ap, app, app_a, H, err = @time ODEs.solve_ode(u₀, tspan, p, 10)

    τₑ, aₑ = get_end(ϕ, dϕ, a, τ, model.ϕₑ)    
    
    mkpath(data_dir)
    npzwrite(data_dir * "ode.npz", Dict("tau"=>τ, "phi"=>ϕ, "phi_d"=>dϕ, "a"=>a, "app_a"=>app_a, "err"=>err, "a_end"=>aₑ, "H"=>H))
    return true
end

function save_f(data_dir::String=MODEL_DATA_DIR)
    model = TMode(1, 0.965, 0.001)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    k = logspace(-2, 2, 100) * ode.aₑ * model.mᵩ
    mᵪ = [0.2, 0.5, 1.0, 2.0, 5.0] .* mᵩ

    ξ = [0.0]
    m3_2 = [0.0, 1.0]

    f = get_f(ode.ϕ, model, 0.0)
    m2_eff(ode, mᵪ, ξ) = get_m2_eff_R(ode, mᵪ, ξ, f)
    PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, m3_2, m2_eff, 1.0)
    return true
end

function test_save_f(data_dir::String="data/SmallField/")
    model = ModelDatas.TMode(1, 0.965, 0.001)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    k = logspace(-1, 1, 5) * ode.aₑ * model.mᵩ
    mᵪ = [1.0] .* mᵩ
    ξ = [1.0 / 6.0]
    ξ_dir = ["data/f_ξ=0/"]
    f = get_f(ode.ϕ, model, 0.0)
    m2_eff(ode, mᵪ, ξ) = get_m2_eff_R(ode, mᵪ, ξ, f)

    f = PPs.save_each(mᵩ, ode, k, mᵪ, ξ, ξ_dir, m2_eff, 10, true)
    @show f
    return true
end

end
