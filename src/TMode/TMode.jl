"""
Model for T-mode α attractor inflation potential
"""
module TModes

# submodules
include("ModelData.jl")
using .ModelDatas

using ..EOMs
using ..Commons
using ..PPs

using StaticArrays, NPZ, Logging
# using JLD2

# global constant
const MODEL_NAME="TMode"
# not complete dir!
const MODEL_DATA_DIR="data/$MODEL_NAME-"

function get_V(ϕ::Real, model::TMode)
    x = ϕ / (sqrt(6) * model.α)
    return model.V₀ * tanh(x)^(2*model.n)
end

function get_dV(ϕ::Real, model::TMode)
    x = ϕ / (sqrt(6) * model.α)
    return sqrt(2/3) / model.α * model.V₀ * sech(x)^2 * tanh(x)^(2*model.n-1)
end

"""
dϕ = dϕ/dτ at slow roll trajectory in conformal time
"""
function get_dϕ_SR(ϕ::Real, model::TMode, a::Real=1.0)
    return - a * get_dV(ϕ, model) / sqrt(3 * get_V(ϕ, model))
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
    x = @. ϕ / (sqrt(6 * model.α))
    if model.n == 1
        return @. sqrt(3*model.α*model.V₀) * log(cosh(x)) + m3_2/sqrt(3)
    else
        throw(ArgumentError("n=$(model.n) is yet to be implemented!"))
    end
end

"""
effective mass squared of the real field
NOTE: conformal coupling not implemented yet!
"""
function get_m2_eff_R(ode::ODEData, mᵪ::Real, ξ::Real, f::Vector)
    if ξ != 0
        throw(ArgumentError("Conformal coupling not implemented yet!"))
    end
    m2 = @. ode.a^2 * (mᵪ^2 + ode.H ^2 + f^2 - mᵪ*f)
    return m2
end

"""
effective mass squared of the imaginary field
NOTE: conformal coupling not implemented yet!
"""
function get_m2_eff_I(ode::ODEData, mᵪ::Real, ξ::Real, f::Vector)
    if ξ != 0
        throw(ArgumentError("Conformal coupling not implemented yet!"))
    end
    m2 = @. ode.a^2 * (mᵪ^2 + ode.H ^2 + f^2 + mᵪ*f)
    return m2
end
get_m2_eff_R(ode, model, ξ, m3_2, mᵪ) = get_m2_eff_R(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2)) / (model.mᵩ^2)
get_m2_eff_I(ode, model, ξ, m3_2, mᵪ) = get_m2_eff_I(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2)) / (model.mᵩ^2)


function save_eom(ϕᵢ::Float64, r::Float64=0.001, data_dir::String=MODEL_DATA_DIR*"$r/")
    mkpath(data_dir)

    model = TMode(1, 0.965, r, ϕᵢ)
    @info dump_struct(model)
    #  @info model
    save_model_data(model, data_dir * "model.dat")

    # initial conditions
    ϕᵢ *= model.ϕₑ
    dϕᵢ = get_dϕ_SR(ϕᵢ, model)
    @debug ϕᵢ, dϕᵢ
    u₀ = SA[ϕᵢ, dϕᵢ, 1.0]
    #  τᵢ = - 1 / get_Hinf(model)
    #  tspan = (τᵢ, -τᵢ)

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    p = (_V, _dV)
    
    τ, ϕ, dϕ, a, ap, app, app_a, H, err, aₑ, Hₑ = EOMs.solve_eom(u₀, p)

    mkpath(data_dir)
    npzwrite(data_dir * "ode.npz", Dict("tau"=>τ, "phi"=>ϕ, "phi_d"=>dϕ, "a"=>a, "app_a"=>app_a, "err"=>err, "a_end"=>aₑ, "H"=>H, "H_end"=>Hₑ, "m_phi"=>model.mᵩ))
    return true
end

#=
function test_ode(data_dir::String=MODEL_DATA_DIR)
    mkpath(data_dir)
    
    r = 0.001
    ϕᵢ = 1.7
    model = TMode(1, 0.965, r, ϕᵢ)
    @show model
    save_model_data(model, data_dir * "model.dat")
    
    ϕᵢ *=  model.ϕₑ
    Hᵢ = sqrt(get_V(ϕᵢ, model) / 3.0)
    # initial dϕdt = dϕdτ
    dϕᵢ = get_dϕ_SR(ϕᵢ, model) / Hᵢ
    #  @show ϕᵢ, dϕᵢ
    u₀ = SA[ϕᵢ, dϕᵢ, 1.0]

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    p = (_V, _dV)
    
    τ, ϕ, dϕ, a, ap, app, app_a, H, err = EOMs.solve_eom(u₀, p)

    mkpath(data_dir)
    npzwrite(data_dir * "ode.npz", Dict("tau"=>τ, "phi"=>ϕ, "phi_d"=>dϕ, "a"=>a, "app_a"=>app_a, "err"=>err, "H"=>H, "m_phi"=>model.mᵩ, "a_end" => 1.0, "H_end" => H[1]))

end

function save_m_eff(data_dir::String=MODEL_DATA_DIR*"$r/")
    model = TMode(1, 0.965, 0.001)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)
    #  a = ode.a

    ξ = 0.0
    m3_2 = [0.0, 1.0] * mᵩ
    mᵪ = logspace(-1.3, 0.7, 10).* mᵩ
    
    for x in m3_2
        dn = "$(data_dir)m_eff/m3_2=$(x/mᵩ)/"
        mkpath(dn)
        for y in mᵪ
            fn = dn * "mᵪ=$(y/mᵩ).npz"
            m_R = get_m2_eff_R(ode, model, ξ, x, y)
            m_I = get_m2_eff_I(ode, model, ξ, x, y)
            npzwrite(fn, Dict("tau" => ode.τ, "m2_R" => m_R, "m2_I" => m_I))
        end
    end
end
=#

function save_f(r::Float64=0.001, data_dir::String=MODEL_DATA_DIR*"$r/";
                num_mᵪ::Int=20, num_m32::Int=5, num_k::Int=100)
    model = TMode(1, 0.965, r, NaN)
    @info dump_struct(model)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    k = logspace(-2.0, 2.0, num_k) * ode.aₑ * model.mᵩ 
    #  mᵪ = SA[logspace(-1.3, 0.7, num_mᵪ).* mᵩ ...]
    mᵪ = SA[logspace(-1.3, 1.0, num_mᵪ).* mᵩ ...]

    ξ = SA[0.0]
    #  m3_2 = [0.0, logspace(-2, log10(2.0), num_m32-1)...] * mᵩ
    m3_2 = SA[collect(range(0.0, 2.0; length=num_m32)) * mᵩ ...]

    m2_eff_R(ode, mᵪ, ξ, m3_2) = get_m2_eff_R(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2))
    PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, m3_2, m2_eff_R, fn_suffix="_R")

    m2_eff_I(ode, mᵪ, ξ, m3_2) = get_m2_eff_I(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2))
    PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, m3_2, m2_eff_I, fn_suffix="_I")
    return true
end
# IMPORTANT: need to run save_eom(1.6, 0.001) before running the benchmarks
dn_bm = "data/TMode-0.001-benchmark"
save_eom_benchmark() = save_eom(1.6, 0.001, dn_bm)
save_f_benchmark() = save_f(0.001, num_mᵪ=5, num_m32=3, num_k=10, dn_bm)
save_f_benchmark2() = save_f(0.001, num_mᵪ=5, num_m32=3, num_k=100, dn_bm)

function save_f_single()
    r = 0.001
    data_dir = MODEL_DATA_DIR

    model = TMode(1, 0.965, r, NaN)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)
    
    num_k = 10
    k = logspace(-2.0, 2.0, num_k) * ode.aₑ * model.mᵩ 
    m2_eff_R(ode, mᵪ, ξ) = get_m2_eff_R(ode, mᵪ, ξ, get_f(ode.ϕ, model, 0.0))
    PPs.test_ensemble(k, ode, m2_eff_R, mᵩ, 0.0)
end

function test_save_f(data_dir::String=MODEL_DATA_DIR)
    model = ModelDatas.TMode(1, 0.965, 0.001, 1.7)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    k = logspace(-1, 1, 5) * ode.aₑ * model.mᵩ
    mᵪ = [1.0] .* mᵩ
    ξ = [0.0]
    m3_2 = 0.0
    #  ξ_dir = ["data/f_ξ=0/"]
    f = get_f(ode.ϕ, model, m3_2)
    m2_eff(ode, mᵪ, ξ) = get_m2_eff_R(ode, mᵪ, ξ, f)
    
    f = PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, m2_eff, direct_out=true)
    if isapprox(f,[0.0016289868135007956, 0.0014188781457536808, 0.0011877475734364273, 0.00010408116931268722, 1.5360952986516037e-6], rtol=1e-2)
        return true
    else
        @show f
        return false
    end
end
end
