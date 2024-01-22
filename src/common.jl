"""
Some convenient function to share among files/modules
"""
module Commons

using NPZ, Interpolations

export logspace
export read_ode
export ODEData

"""
returns an array whose elements are even spaced on logarithmic scale
"""
function logspace(start, stop, num::Integer)
    return 10 .^ (range(start, stop, num))
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
    err::V

    aₑ::F
end


"""
read ODE solution stored in data/ode.npz
"""
function read_ode(fn::String="data/ode.npz")
    # maybe a try catch clause here; not sure if necessary
    data = npzread(fn)
    τ = data["tau"]
    ϕ = data["phi"]
    dϕ = data["phi_d"]
    a = data["a"]
    app_a = data["app_a"]
    err = data["err"]
    aₑ = data["a_end"]
    return ODEData(τ, ϕ, dϕ, a, app_a, err, aₑ)
end

end
