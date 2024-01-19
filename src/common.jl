"""
Some convenient function to share among files/modules
"""
module Commons

using NPZ

export logspace
export read_ode

"""
simple derivative of two sampled data;
note that diff exists in Base already!
"""
#  function diff(t::Vector, x::Vector)
#      # TODO: rename this function; use the Base.diff here!
#      dx = (x[2:end] - x[1:end-1]) ./ (t[2:end] - t[1:end-1])
#      return dx
#  end

"""
returns an array whose elements are even spaced on logarithmic scale
"""
function logspace(start, stop, num::Integer)
    return 10 .^ (range(start, stop, num))
end

"""
read ODE solution stored in data/ode.npz
"""
function read_ode(fn::String="data/ode.npz")
    # maybe a try catch clause here; not sure
    data = npzread(fn)
    τ = data["tau"]
    ϕ = data["phi"]
    dϕ = data["phi_d"]
    a = data["a"]
    app_a = data["app_a"]
    err = data["err"]
    aₑ = data["a_end"]
    return τ, ϕ, dϕ, a, app_a, err, aₑ
end

end
