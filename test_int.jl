using Pkg, MultiQuad, QuadGK, NumericalIntegration, LinearInterpolations

"""
compute double integrals numerically using trapzoidal rule
similar multiquad.jl convention
∫_{x₁}^{x₂} ∫_{y₁(x)}^{y₂(x)} f(y, x) dy dx

Here, f should take two float variables as indices
"""
function double_trap(f::Function, x1::Real, x2::Real, y1::Function, y2::Function, x::Vector, y::Vector)
    """
    integrate over y first
    """
    function get_inner_int(i::Int64)
        i_start = findfirst(z -> z >= y1(x[i]), y)
        i_end = findlast(z -> z <= y2(x[i]), y)
        # @show y1(x[i]), y2(x[i])
        # @show i_start, i_end, y[i_start:i_end]

        # it could happen that i_start and i_end are so close to each other, the array would be empty
        integral = i_start >= i_end ? 0.0 : integrate(y[i_start:i_end], [f(z, i) for z in i_start:i_end])
        # i_start >= i_end ? nothing : @show [f(x[z], y[i]) for z in i_start:i_end]
        # @show integral
        return integral
    end

    i_start2 = findfirst(z -> z >= x1, x)
    i_end2 = findlast(z -> z <= x2, x)
    # @show i_start2, i_end2
    return integrate(x[i_start2:i_end2], [get_inner_int(z) for z in i_start2:i_end2])
end

"""
returns an array whose elements are even spaced on logarithmic scale
"""
function logspace(start, stop, num::Integer)
    return 10.0 .^ (range(start, stop, num))
end

N = 20
k = 0.1

#######################################################################
# simulate flat f in the IR
function f(y, x)
    # @show y, x
    if x < 1 && x > 0 && y < 1 && y > 0
        return x*y
    else 
        return 0.0 
    end
end

res = @time dblquad(f, 0.0, 1.0, z->abs(k - z), z->k + z, rtol=1e-5)[1]
@show res
# @show [(abs(k-x), k+x) for x in range(0, 1, 10)]
# @show [quad(y -> f(y, x), abs(k-x), k+x)[1] for x in range(0, 1, 10)]

#######################################################################

X = logspace(-1, 0, N)
# X = collect(0:1:N)
# Y = logspace(-2, 0, N)
Y = X
f_array = [x > 0 && x < 1 ? 1.0 : 0.0 for x in X]
# @show f_array

# interpolate into more data points
X_more = logspace(-1, 0, N*50)
# X_more = collect(0:1:N*50)
Y_more = X_more
f_array_more = [interpolate(X, f_array, x, extrapolate=LinearInterpolations.Constant(1.0)) for x in X_more]
# @show f_array_more

function f_disc(i, j)
    return X_more[i] * Y_more[j] * f_array_more[i] * f_array_more[j]
end

res2 = @time double_trap(f_disc, 0.0, 1.0, z->abs(k - z), z->k + z, X_more, Y_more)
@show res2
