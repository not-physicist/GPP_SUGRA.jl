"""
Model for small field inflation potential
"""
module SmallFields

using Interpolations, StaticArrays, OrdinaryDiffEq, NPZ, NumericalIntegration
#  using Infiltrator

using ..ODEs
using ..Commons
using ..PPs

# global constant
const MODEL_NAME="SmallField"
const MODEL_DATA_DIR="data/$MODEL_NAME/"


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
    mₑ² = ode.a.^2 .* mᵪ^2 - (1.0 - 6.0 * ξ) .* ode.app_a
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
Calculate the background quantities and save to data/ode.npz
"""
function save_ode(data_dir::String=MODEL_DATA_DIR)
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
    
    τ, ϕ, dϕ, a, ap, app, app_a, H, err = @time ODEs.solve_ode(u₀, tspan, p, 1e2)

    τₑ, aₑ = get_end(ϕ, dϕ, a, τ, ϕₑ)
    τₑ, Hₑ = get_end(ϕ, dϕ, H, τ, ϕₑ)
    
    mkpath(data_dir)
    npzwrite(data_dir * "ode.npz", Dict("tau"=>τ, "phi"=>ϕ, "phi_d"=>dϕ, "a"=>a, "app_a"=>app_a, "err"=>err, "a_end"=>aₑ, "H"=>H, "H_end"=>Hₑ, "m_phi"=>model.mᵩ))
    return true
end

function save_f(data_dir::String=MODEL_DATA_DIR)
    model = SmallField(0.5, 6, 60.0)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    k = logspace(-3, 1, 400) * ode.aₑ * model.mᵩ
    mᵪ = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0] .* mᵩ
    ξ = [1.0 / 6.0, 0.0]
    ξ_dir = ["1_6/", "0/"]
    ξ_dir = data_dir * "f_ξ=" .* ξ_dir 

    PPs.save_each(mᵩ, ode, k, mᵪ, ξ, ξ_dir, get_m2_eff, 1e2)
end

"""
save the spectra for one set of parameters; just for testing
"""
function test_save_f(data_dir::String=MODEL_DATA_DIR)
    model = SmallField(0.5, 6, 60.0)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    k = logspace(-1, 1, 5) * ode.aₑ * model.mᵩ
    #  @show mᵩ
    mᵪ = [1.0] .* mᵩ
    ξ = [1.0 / 6.0]
    #  ξ_dir = ["data/f_ξ=1_6/"]
    
    f = PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, get_m2_eff, direct_out=true)
    
    # approximate the true values
    if isapprox(f, [1.547728392833625e-6, 1.4363698510177691e-7, 1.1700695189080433e-7, 1.2696232419956884e-10, 2.849093626043739e-14], rtol=1e-2)

        @show f
        return true
    else
        @show f
        return false
    end
end

end
