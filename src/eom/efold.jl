"""
Solving EOM in number of efolds
"""
module EFolds

using StaticArrays, OrdinaryDiffEq, NumericalIntegration

#  using ..Helpers: get_others

"""
squared Hubble parameter computed in number of efolds
"""
function get_H2(ϕ::Real, dϕdN::Real, V::Function)
    d = 1 - 1/6 * dϕdN^2
    return V(ϕ) / 3.0 / d
end

"""
the friedman equation written in number of efolds

H is the normal Hubble
"""
function friedmann_eq_efold(u, p, t)
    ϕ = u[1]
    dϕdN = u[2]
    a = u[3]

    get_V = p[1]
    get_dV = p[2]

    H2 = get_H2(ϕ, dϕdN, get_V)
    
    return SA[dϕdN, - dϕdN ^ 3 - 3 * dϕdN - get_dV(ϕ) / get_H2(ϕ, dϕdN, get_V), a]
end

"""
assume u₀ in efold units
"""
function solve_eom(u₀::SVector{3, Float64}, 
                         p::Tuple{Function, Function})
    # defines when to terminate integrator (at ϵ1 = 0.5)
    condition(u, t, integrator) = u[2]^2 / (2) <= 0.1
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition,affect!)

    prob = ODEProblem(friedmann_eq_efold, u₀, [0.0, 10.0], p)
    # dtmax setting is required to ensure the following differentiation behaves well enough
    sol = solve(prob, RK4(), reltol=1e-6, abstol=1e-8, callback=cb, dtmax=0.001)
    
    N = sol.t
    ϕ = [x[1] for x in sol.u]
    dϕdN = [x[2] for x in sol.u]
    a = [x[3] for x in sol.u]

    # the last two element seems to be duplicate; something to do with the termination
    N, ϕ, dϕdN, a = N[1:end-1], ϕ[1:end-1], dϕdN[1:end-1], a[1:end-1]
    println("Number of efolds in inflation: $(N[end] - N[1])")
    
    # due to numerical nature, very small negative number can be produced for H^2
    H = sqrt.(max.(0, get_H2.(ϕ, dϕdN, p[1])))
    τ = cumul_integrate(N, 1 ./ (a .* H))
    dϕdτ = a .* H .* dϕdN
    #  @show H[1:5:100]
    #  @show (diff(H)/diff(τ))[1:10]
    
    ap = diff(a) ./ diff(τ)
    H_τ = ap ./ (a[1:end-1] .^2)
    #  @show H_τ[1:5:100]

    return τ, ϕ, dϕdτ, a
end
end
