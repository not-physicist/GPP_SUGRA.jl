"""
Solve background ODE in conformal time
"""
module ODEs

using StaticArrays, OrdinaryDiffEq, NumericalIntegration, LinearInterpolations
#  import ..Commons

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

    SA[dϕ, -2 * H * dϕ - a^2 * get_dV(ϕ), a * H]
end

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
    
    SA[dϕdN, -3 * dϕdN - get_dV(ϕ) / get_H2(ϕ, dϕdN, get_V), a]
end

"""
use the second Friedmann equation to get the error of the solution
assume the sizes of the vectors: n-2, n, n, n
"""
function get_err(app::Vector, a::Vector, ϕ::Vector, dϕ::Vector, get_V::Function)
    V = get_V.(ϕ)
    return abs.(app./a[1:end-2] - (4*a[1:end-2].^2 .* V[1:end-2] - dϕ[1:end-2].^2)/6)
end

"""
calculate other quantities
"""
function get_others(τ::Vector, ϕ::Vector, dϕ::Vector, a::Vector, get_V::Function)
    ap = diff(a) ./ diff(τ) 
    app = diff(ap) ./ diff(τ[1:end-1]) 
    app_a = app ./ a[1:end-2]
    # cosmic Hubble
    H = ap ./ (a[1:end-1] .^2)
    err = get_err(app, a, ϕ, dϕ, get_V)
    
    #  @show size(τ) size(ϕ) size(dϕ) size(a) size(ap) size(app) size(app_a) size(H) size(err)
    # trim arrays to have the identical dimension
    # once the step size is small enough, remove the last few elements should be OK
    return τ[1:end-2], ϕ[1:end-2], dϕ[1:end-2], a[1:end-2], ap[1:end-1], app, app_a, H[1:end-1], err
end

"""
solve ODE given the parameters and initial conditions
"""
function solve_ode(u₀::SVector{3, Float64}, 
                   tspan::Tuple{Float64, Float64}, 
                   p::Tuple{Function, Function},
                   dtmax::Real)
    # define and solve ODE
    
    # split up ODE solving to non-oscillatory and oscillatory
    # roughly to ϕₑ
    prob = ODEProblem(friedmann_eq, u₀, [tspan[1], 0.0], p)
    sol = solve(prob, RK4(), reltol=1e-6, abstol=1e-8, dtmax=0.1) 

    
    u₀_new = SA[sol.u[end]...]
    #  @show typeof(u₀_new) typeof(u₀)
    prob2 = ODEProblem(friedmann_eq, u₀_new, [0.0, tspan[2]], p)
    sol2 = solve(prob2, RK4(), maxiters=1e8, reltol=1e-7, abstol=1e-9, save_start=false)

    # somehow, static array has weirdness in indexing...
    τ = vcat(sol.t, sol2.t)
    ϕ = vcat([x[1] for x in sol.u], [x[1] for x in sol2.u])
    dϕ = vcat([x[2] for x in sol.u], [x[2] for x in sol2.u])
    a = vcat([x[3] for x in sol.u], [x[3] for x in sol2.u])
    
    return get_others(τ, ϕ, dϕ, a, p[1])
end


function solve_ode_efold(u₀::SVector{3, Float64}, 
                         p::Tuple{Function, Function})
    prob = ODEProblem(friedmann_eq_efold, u₀, [0.0, 4.2], p)
    sol = solve(prob, RK4(), reltol=1e-6, abstol=1e-8, dtmax=0.001) 

    N = sol.t
    ϕ = [x[1] for x in sol.u]
    dϕdN = [x[2] for x in sol.u]
    a = [x[3] for x in sol.u]
    #  @show ϕ[end-10:end]
    
    H = sqrt.(get_H2.(ϕ, dϕdN, p[1]))
    τ = cumul_integrate(N, 1 ./ (a .* H))
    dϕdτ = a .* H .* dϕdN
    #  @show τ[1:50:end]
    return get_others(τ, ϕ, dϕdτ, a, p[1])
end

end
