"""
Model for T-mode α attractor inflation potential
"""
module TModes

include("ModelData.jl")
using .ModelDatas

using ..ODEs
using ..Commons
using ..PPs

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


function save_ode()
    # TODO: put realistic numbers
    model = TMode(1.0, 1.0, 1)

    # initial conditions
    u₀ = SA[1.0, 0.0, 1.0]
    τᵢ = - 1 / Hinf
    tspan = (τᵢ, -τᵢ)

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    p = (_V, _dV)
end

function test()
    model = ModelDatas.TMode(1, 0.95, 0.001)
    @show model
    return true
end
end
