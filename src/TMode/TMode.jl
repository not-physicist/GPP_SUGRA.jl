"""
Model for T-mode α attractor inflation potential
"""
module TModes

include("ModelData.jl")
using .ModelDatas

using ..ODEs
using ..Commons
using ..PPs

using StaticArrays, NPZ

"""
inflationary potential parameter; in reduced Planck unit
"""

function get_f(Φ::Real, model::TMode, m3_2::Real)
    x = Φ/(sqrt(3) * model.α)
    if model.n == 1
        return sqrt(3)*model.α*sqrt(model.V₀)*log(cosh(x)) + m3_2/sqrt(3)
    else
        throw(ErrorException("This n is yet to be implemented!"))
    end
end

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

function save_ode()
    model = TMode(1, 0.965, 0.001)
    @show model

    # initial conditions
    u₀ = SA[2.0 * model.ϕₑ, 0.0, 1.0]
    τᵢ = - 1 / get_Hinf(model)
    tspan = (τᵢ, -τᵢ)

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    p = (_V, _dV)
    
    τ, ϕ, dϕ, a, ap, app, app_a, err = @time ODEs.solve_ode(u₀, tspan, p)

    τₑ, aₑ = get_end(ϕ, dϕ, a, τ, model.ϕₑ)    

    if !isdir("data")
        mkdir("data") 
    end
    npzwrite("data/ode.npz", Dict("tau"=>τ, "phi"=>ϕ, "phi_d"=>dϕ, "a"=>a, "app_a"=>app_a, "err"=>err, "a_end"=>aₑ))
    return true
end

function test()
    model = ModelDatas.TMode(1, 0.9649, 0.001)
    @show model
    ModelDatas.get_ϕ_end(model)
    return true
end

end
