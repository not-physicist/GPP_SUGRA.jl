#  include("common.jl")
module ODEs

using StaticArrays, OrdinaryDiffEq
#  import ..Commons

"""
the friedman equation written in conformal time
"""
function friedmann_eq(u, p, t)
    #  (;v, n, Nₑ, M, mᵩ, ϕₑ) = model
    ϕ = u[1]
    dϕ = u[2]
    a = u[3]

    get_V = p[1]
    get_dV = p[2]

    H2 = 1 / 3 * (dϕ^2 / 2 + a^2 * get_V(ϕ))
    H = √(H2)

    SA[dϕ, -2 * H * dϕ - a^2 * get_dV(ϕ), a * H]
end

"""
use the second Friedmann equation to get the error of the solution
assume the sizes of the vectors: n-2, n, n, n
"""
function get_err(app::Vector, a::Vector, ϕ::Vector, dϕ::Vector, get_V::Function)
    V = get_V.(ϕ)
    #  @show size(app) size(a) size(ϕ) size(dϕ) size(V)
    abs.(app./a[1:end-2] - (4*a[1:end-2].^2 .* V[1:end-2] - dϕ[1:end-2].^2)/6)
end

"""
solve ODE given the parameters and initial conditions
"""
function solve_ode(u₀::SVector{3, Float64}, 
                   tspan::Tuple{Float64, Float64}, 
                   p::Tuple{Function, Function},
                   dtmax::Real=1e2)

    # define and solve ODE
    prob = ODEProblem(friedmann_eq, u₀, tspan, p)
    sol = solve(prob, DP8(), dtmax=dtmax)

    # somehow, static array has weirdness in indexing...
    τ = sol.t
    ϕ = [x[1] for x in sol.u]
    dϕ = [x[2] for x in sol.u]
    a = [x[3] for x in sol.u]
    #  ap = Commons.diff(τ, a)
    ap = diff(a) ./ diff(τ) 
    #  app = Commons.diff(τ[1:end-1], ap)
    app = diff(ap) ./ diff(τ[1:end-1]) 
    app_a = app ./ a[1:end-2]
    err = get_err(app, a, ϕ, dϕ, p[1])

    return τ, ϕ, dϕ, a, ap, app, app_a, err
end

end
