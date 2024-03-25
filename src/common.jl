"""
Some convenient function to share among files/modules
"""
module Commons

using NPZ, NumericalIntegration, LinearInterpolations
# using Interpolations, JLD2

export logspace, read_ode, ODEData, get_end, LinearInterpolations, dump_struct

"""
returns an array whose elements are even spaced on logarithmic scale
"""
function logspace(start::Float64, stop::Float64, num::Integer)::Vector{Float64}
    return 10.0 .^ (range(start, stop, num))
end

"""
struct to store the ODE data;
note that they may have different length (due to the derivatives)
"""
struct ODEData{V<:Vector, F<:Real}
    τ::V
    ϕ::V
    dϕ::V
    a::V
    app_a::V
    H::V
    err::V

    aₑ::F
    Hₑ::F
end

"""
read ODE solution stored in data/ode.npz
"""
function read_ode(data_dir::String)::ODEData
    # maybe a try catch clause here; not sure if necessary
    fn = data_dir * "ode.npz"
    data = npzread(fn)
    #  fn = data_dir * "ode.jld2"
    #  data = load(fn)

    τ = data["tau"]::Vector{Float64}
    ϕ = data["phi"]::Vector{Float64}
    dϕ = data["phi_d"]::Vector{Float64}
    a = data["a"]::Vector{Float64}
    app_a = data["app_a"]::Vector{Float64}
    H = data["H"]::Vector{Float64}
    err = data["err"]::Vector{Float64}
    aₑ = data["a_end"]::Float64
    Hₑ = data["H_end"]::Float64
    return ODEData(τ, ϕ, dϕ, a, app_a, H, err, aₑ, Hₑ)
end

"""
get scale factor and conformal time at the end of inflation
can actually replace scale factor witn any other quantity
"""
function get_end(ϕ::Vector, dϕ::Vector, a::Vector, τ::Vector, ϕₑ::Real)
    # generate an appropriate mask
    flag = true  # whether dϕ has changed sign 
    i = 1  # index for while
    mask = zeros(Int, size(ϕ))
    while flag == true && i < size(ϕ)[1]
        mask[i] = 1
        i += 1
        if dϕ[i] * dϕ[2] < 0
            # terminate after sign change
            # for whatever reason dϕ[1] is always 0 
            flag = false
        end
    end
    mask = BitVector(mask)
    
    # depending small/large field model, the field value array can be descending or ascending
    try
        itp = LinearInterpolations.Interpolate((ϕ[mask],), τ[mask])
    catch
        itp = LinearInterpolations.Interpolate((reverse(ϕ[mask]),), reverse(τ[mask]))
    end
    τₑ = itp(ϕₑ)
    
    itp = LinearInterpolations.Interpolate((τ[mask],), a[mask])
    aₑ = itp(τₑ)
    return τₑ, aₑ
end


function get_t(τ::Vector, a::Vector)
    t = cumul_integrate(τ, a)
    return t
end

function get_ϵ₁(ode::ODEData)
    dH = diff(ode.H) ./ diff(ode.τ) ./ ode.a[1:end-1]
    return - dH ./ ode.H[1:end-1] .^ 2
    #  return 1 / 2  * (ode.dϕ ./ ode.H ./ ode.a ).^ 2
end

"""
Simple dump for struct, but instead of output to stdout, return a string for Logging
"""
function dump_struct(s)
    out = "Fields of $(typeof(s)): \n"
    for i in fieldnames(typeof(s))
        out *= "$i" * " = " * string(getfield(s, i)) * "\n"
    end
    return out
end

end
