"""
Some convenient function to share among files/modules
"""
module Commons

using NPZ, NumericalIntegration, LinearInterpolations
# using Interpolations, JLD2

export logspace, read_ode, ODEData, get_end, LinearInterpolations, dump_struct, double_trap

"""
returns an array whose elements are even spaced on logarithmic scale
"""
function logspace(start, stop, num::Integer)
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

    τ = data["tau"]
    ϕ = data["phi"]
    dϕ = data["phi_d"]
    a = data["a"]
    app_a = data["app_a"]
    H = data["H"]
    err = data["err"]
    aₑ = data["a_end"]
    Hₑ = data["H_end"]
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

"""
compute double integrals numerically using trapzoidal rule
similar multiquad.jl convention
∫_{x₁}^{x₂} ∫_{y₁(x)}^{y₂(x)} f(y, x) dy dx

Here, f should take two Int64 variables as indices
"""
function double_trap(f::Function, x1::Real, x2::Real, y1::Function, y2::Function, x::Vector, y::Vector)
    """
    integrate over y first
    """
    function get_inner_int(i::Int64)
        i_start = findfirst(z -> z>y1(x[i]), y)
        i_end = findlast(z -> z<y2(x[i]), y)
        # @show i_start, i_end
        return integrate(y[i_start:i_end], [f(z, i) for z in i_start:i_end])
    end
    
    i_start2 = findfirst(z -> z > x1, x)
    i_end2 = findlast(z -> z < x2, x)
    # @show i_start, i_end
    return integrate(x[i_start2:i_end2], [get_inner_int(z) for z in i_start2:i_end2])
end

end
