"""
Some convenient function to share among files/modules
"""
module Commons

using NPZ, Interpolations

export logspace
export read_ode
export ODEData
export get_end
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

"""
get scale factor and conformal time at the end of inflation
"""
function get_end(ϕ::Vector, dϕ::Vector, a::Vector, τ::Vector, ϕₑ::Real)
    # generate an appropriate mask
    flag = true  # if dϕ hasn't changed sign 
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
        itp = interpolate((ϕ[mask],), τ[mask], Gridded(Linear()))
    catch
        itp = interpolate((reverse(ϕ[mask]),), reverse(τ[mask]), Gridded(Linear()))
    end
    τₑ = itp(ϕₑ)
    
    itp = interpolate((τ[mask],), a[mask], Gridded(Linear()))
    aₑ = itp(τₑ)
    return τₑ, aₑ
end


end
