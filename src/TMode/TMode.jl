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
    x = ϕ / (sqrt(6 * model.α))
    return model.V₀ * tanh(x)^(2*model.n)
end

function get_dV(ϕ::Real, model::TMode)
    x = ϕ / (sqrt(6 * model.α))
    return 2*model.n / sqrt(6*model.α) * model.V₀ * sech(x)^2 * tanh(x)^(2*model.n-1)
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
# get_m2_eff_R(ode, model, ξ, m3_2, mᵪ) = get_m2_eff_R(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2)) / (model.mᵩ^2)
# get_m2_eff_I(ode, model, ξ, m3_2, mᵪ) = get_m2_eff_I(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2)) / (model.mᵩ^2)

"""
effective mass squared WITHOUT sugra-correction
"""
function get_m2_no_sugra(ode::ODEData, mᵪ::Real, ξ::Real)
    m2 = @. ode.a^2 * mᵪ^2 - (1-6*ξ) * ode.app_a
    return m2
end

"""
ϕᵢ: in unit of ϕₑ (field value at end of slow roll)
init_time_mul: initial (conformal) time multiplicant; needs to make the simulation run longer for large r 
"""
function save_eom(ϕᵢ::Float64, r::Float64=0.001, data_dir::String=MODEL_DATA_DIR*"$r/", max_a::Float64=1e4)
    mkpath(data_dir)

    model = TMode(1, 0.965, r, ϕᵢ)
    @info dump_struct(model)
    @info data_dir
    save_model_data(model, data_dir * "model.dat")

    # initial conditions
    ϕᵢ *= model.ϕₑ
    dϕdτᵢ = get_dϕ_SR(ϕᵢ, model)
    Hinf = sqrt(get_V(ϕᵢ,model)/3)
    # initial a = 1
    dϕdNᵢ = dϕdτᵢ/Hinf  

    @info "Initial conditions are: ", ϕᵢ, dϕdNᵢ
    u₀ = SA[ϕᵢ, dϕdNᵢ, 1.0]

    # parameters
    _V(x) = get_V(x, model)
    _dV(x) = get_dV(x, model)
    p = (_V, _dV)
    
    # println("\nEFOLDS")
    τ, ϕ, dϕ, a, app_a, H, err, aₑ, Hₑ = EOMs.solve_eom(u₀, p, max_a)
    
    # println("\nConformal")
    # tspan = (-1/Hinf, 1/Hinf)
    # u₀ = SA[ϕᵢ, dϕdτᵢ, 1.0]
    # τ, ϕ, dϕ, a, app_a, H, err, aₑ, Hₑ = EOMs.solve_eom_conf_only(u₀, p, tspan)
    
    dϕ_SR = [get_dϕ_SR(x, model, y) for (x, y) in zip(ϕ, a)]

    mkpath(data_dir)
    npzwrite(data_dir * "ode.npz", Dict("tau"=>τ, "phi"=>ϕ, "phi_d"=>dϕ, "a"=>a, "app_a"=>app_a, "err"=>err, "a_end"=>aₑ, "H"=>H, "H_end"=>Hₑ, "m_phi"=>model.mᵩ, "phi_d_sr"=>dϕ_SR))
    return true
end

function save_f(r::Float64=0.001, data_dir::String=MODEL_DATA_DIR*"$r/";
                num_mᵪ::Int=20, num_k::Int=100)
    model = TMode(1, 0.965, r, NaN)
    @info dump_struct(model)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    # k = logspace(-2.0, 2.0, num_k) * ode.aₑ * mᵩ 
    k = logspace(-2.0, 2.0, num_k) * ode.aₑ * ode.Hₑ
    # mᵪ = SA[3.0 * mᵩ]
    mᵪ = SA[logspace(-2.0, log10(3.0), num_mᵪ).* mᵩ ...]
    ξ = SA[0.0]
    # m3_2 = SA[collect(range(0.0, 2.0; length=num_m32)) * mᵩ ...]
    # m3_2 = SA[0.0] .* mᵩ
    m3_2 = SA[0.0, 0.01, 0.1, 0.2] .* mᵩ
    
    m2_eff_R(ode, mᵪ, ξ, m3_2) = get_m2_eff_R(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2))
    PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, m3_2, m2_eff_R, fn_suffix="_R")
     
    m2_eff_I(ode, mᵪ, ξ, m3_2) = get_m2_eff_I(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2))
    PPs.save_each(data_dir, mᵩ, ode, k, mᵪ, ξ, m3_2, m2_eff_I, fn_suffix="_I")

    # PPs.save_each(data_dir * "nosugra/", mᵩ, ode, k, mᵪ, ξ, get_m2_no_sugra, solve_mode=true)
    return true
end
const dn_bm = "data/TMode-0.001-benchmark/"
save_eom_benchmark() = save_eom(1.7, 0.001, dn_bm)
save_f_benchmark() = save_f(0.001, num_mᵪ=5, num_m32=3, num_k=10, dn_bm)
save_f_benchmark2() = save_f(0.001, num_mᵪ=5, num_m32=3, num_k=100, dn_bm)

function save_m_eff(r::Float64=0.001, data_dir::String=MODEL_DATA_DIR*"$r/";
                    num_mᵪ::Int=5, num_m32::Int=1)
    model = TMode(1, 0.965, r, NaN)
    @info dump_struct(model)
    mᵩ = model.mᵩ
    ode = read_ode(data_dir)

    ξ = 0.0
    mᵪ = SA[logspace(-1.3, 0.3, num_mᵪ).* mᵩ ...]
    # mᵪ = SA[2.0 * mᵩ]
    m3_2 = SA[collect(range(0.0, 2.0; length=num_m32)) * mᵩ ...]
    # m3_2 = SA[0.0]

    m2_eff_R(mᵪ, m3_2) = get_m2_eff_R(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2))
    m2_eff_I(mᵪ, m3_2) = get_m2_eff_I(ode, mᵪ, ξ, get_f(ode.ϕ, model, m3_2))
    
    @info "aₑm_ϕ = $(ode.aₑ*mᵩ)" 
    for m3_2_i in m3_2 
        m3_2_dir = data_dir * "m3_2=$(m3_2_i/mᵩ)/"
        mkpath(m3_2_dir)
        # @info out_dir
        for mᵪᵢ in mᵪ
            # @info "Saving effective mass..."
            @info "mᵪ=$mᵪᵢ\tm3_2=$m3_2_i"
            
            f = get_f(ode.ϕ, model, m3_2_i)
            m2_R = m2_eff_R(mᵪᵢ, m3_2_i)
            m2_I = m2_eff_I(mᵪᵢ, m3_2_i)
            
            mkpath(m3_2_dir * "m_eff")
            out_fn = m3_2_dir * "m_eff/m_chi=$(mᵪᵢ/mᵩ).npz"
            npzwrite(out_fn, Dict("tau"=>ode.τ, "a"=>ode.a, "m2_R"=>m2_R, "m2_I"=>m2_I, "f"=>f))
        end
    end
    return true
end

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
