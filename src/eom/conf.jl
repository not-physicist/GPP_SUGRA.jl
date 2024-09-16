"""
solving EOM in conformal time
"""
module Conformals

using StaticArrays, OrdinaryDiffEq

#  using ..Helpers: get_others

"""
the friedman equation written in conformal time

H is the conformal Hubble
"""
function friedmann_eq(u, p, t)
    ϕ = u[1]
    dϕ = u[2]
    a = u[3]

    get_V = p[1]
    get_dV = p[2]
    
    # normal Hubble
    H = sqrt(1 / 3 * (dϕ^2 / (2*a^2) + get_V(ϕ)))

    return SA[dϕ, -2 * a * H * dϕ - a^2 * get_dV(ϕ), a^2 * H]
end

"""
solve ODE given the parameters and initial conditions
"""
function solve_eom(u₀::SVector{3, Float64}, 
                   tspan::Tuple{Float64, Float64}, 
                   p::Tuple{Function, Function},
                   cb)
    prob = ODEProblem(friedmann_eq, u₀, tspan, p)
    sol = solve(prob, RK4(), maxiters=1e8, reltol=1e-12, abstol=1e-12, save_start=false, callback=cb)

    # @show sol
    
    τ = sol.t
    ϕ = sol[1, :]
    dϕdτ = sol[2, :]
    a = sol[3, :]

    # normal Hubble
    H = @. sqrt(1 / 3 * (dϕdτ^2 / 2 + a^2 * p[1](ϕ)))/a
    # @show H[1:10]
    # @show τ[1:10], a[1:10]
    @show u₀, ϕ[1:10], dϕdτ[1:10]

    return τ, ϕ, dϕdτ, a, H
end

end
