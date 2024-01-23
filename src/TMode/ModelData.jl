"""
module for generating struct storing model parameters
"""
module ModelDatas

using SimpleNonlinearSolve

export TMode

struct TMode{T<:Real, N<:Int} 
    n::N
    nₛ::T
    r::T 
    # derivated quantities
    V₀::T
    α::T 
end
TMode(n, nₛ, r) = TMode(n, nₛ, r, get_V₀(n, nₛ, r), get_α(n, nₛ, r))

"""
compute field value corresponding to the CMB pivot scale 
"""
function get_ϕ_cmb(n::Int, nₛ::Real, α::Real)
    function _f_aux(u, p)
        p = 2 * n
        λ = 1 / (sqrt(6) * α)
        x = u * λ
        δ = 1 - nₛ
        return cosh.(x) .^ 2 .- 1 / (2*δ) * (δ + 4*p*λ^2 + sqrt(δ^2 + 4 * p^2 * λ^2 * δ + 16 * p^2 * λ^4))
    end

    u₀ = (1.0, 5.0)
    prob = IntervalNonlinearProblem(_f_aux, u₀)
    sol = solve(prob, ITP())
    sol = only(sol)
    #  println("Root finding result for ϕ_cmb is $sol")
    @show sol
    return sol
end

"""
first slow roll parameter
"""
function get_ϵV(ϕ::Real, n::Int, α::Real)
    x = ϕ / (sqrt(6) * α)
    dV_V = 2 * n / (cosh(x) * sinh(x)) / (sqrt(6) * α)
    return dV_V^2 / 2
end

function get_V₀(n::Int, nₛ::Real, r::Real)
    A = 2.2e-9
    α = get_α(n, nₛ, r)
    ϕ_cmb = get_ϕ_cmb(n, nₛ, α)
    ϵ = get_ϵV(ϕ_cmb, n, α)
    @show ϵ
    x = ϕ_cmb / (sqrt(6) * α)
    V₀ = A * 24 * pi^2 * ϵ / tanh(x)^(2*n)
    @show V₀
    return V₀
end

function get_α(n::Int, nₛ::Real, r::Real)
    δ = 1 - nₛ
    return 8 * 2*n * sqrt(2*r) / sqrt((2*n)^2 * (8*δ - r)^2 - 4*r^2) / sqrt(6)
end

end
