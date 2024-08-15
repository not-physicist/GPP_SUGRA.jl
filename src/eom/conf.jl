"""
solving EOM in conformal time
"""
module Conformals

using StaticArrays, OrdinaryDiffEq, TerminalLoggers

#  using ..Helpers: get_others

"""
the friedman equation written in conformal time

H is the conformal Hubble
"""
function friedmann_eq(u, p, t)
    #  (;v, n, Nₑ, M, mᵩ, ϕₑ) = model
    ϕ = u[1]
    dϕ = u[2]
    a = u[3]

    get_V = p[1]
    get_dV = p[2]
    
    # conformal Hubble
    H = √(1 / 3 * (dϕ^2 / 2 + a^2 * get_V(ϕ)))

    return SA[dϕ, -2 * H * dϕ - a^2 * get_dV(ϕ), a * H]
end

"""
solve ODE given the parameters and initial conditions
"""
function solve_eom(u₀::SVector{3, Float64}, 
                   tspan::Tuple{Float64, Float64}, 
                   p::Tuple{Function, Function})
    prob = ODEProblem(friedmann_eq, u₀, tspan, p)
    sol = solve(prob, Tsit5(), maxiters=1e8, reltol=1e-9, abstol=1e-12, save_start=false, progress = true)
    #  sol = solve(prob, RK4(), maxiters=1e8, reltol=1e-9, abstol=1e-10, save_start=false)
    
    τ= sol.t
    ϕ = sol[1, :]
    dϕdτ = sol[2, :]
    a = sol[3, :]
       
    return τ, ϕ, dϕdτ, a
end

end
