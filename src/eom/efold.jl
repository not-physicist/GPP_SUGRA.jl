"""
Solving EOM in number of efolds
"""
module EFolds

using StaticArrays, OrdinaryDiffEq, NumericalIntegration, Logging

#  using ..Helpers: get_others

"""
squared Hubble parameter computed in number of efolds
"""
function get_H2(ϕ::Real, dϕdN::Real, V::Function)
    d = 1.0 - dϕdN^2 / 6.0
    return V(ϕ) / 3.0 / d
end

"""
the friedman equation written in number of efolds

H is the normal Hubble
"""
function friedmann_eq_efold(u::SVector, p::Tuple, t::Real)
    ϕ = u[1]
    dϕdN = u[2]
    a = u[3]

    get_V = p[1]
    get_dV = p[2]

    return SA[dϕdN, + dϕdN ^ 3 / 2.0 - 3 * dϕdN - get_dV(ϕ) / get_H2(ϕ, dϕdN, get_V), a]
end

"""
assume u₀ in efold units
"""
function solve_eom(u₀::SVector{3, Float64}, 
                   p::Tuple{Function, Function})    
    # defines when to terminate integrator (at ϵ1 = 0.1)
    condition(u, t, integrator) = u[2]^2 / (2) <= 2.0
    affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(condition, affect!)

    # need at most 30 efolds
    prob = ODEProblem(friedmann_eq_efold, u₀, [0.0, 30.0], p)
    # dtmax setting is required to ensure the following differentiation behaves well enough
    sol = solve(prob, Vern9(), reltol=1e-15, callback=cb, dtmax=1e-4, maxiters=1e9)
     
    N = sol.t
    ϕ = sol[1, :]
    dϕdN = sol[2, :]
    # rescale scale factor directly (aₑ = 1 always)
    a = sol[3, :] / sol[3, end]

    # the last two element seems to be duplicate; something to do with the termination
    N, ϕ, dϕdN, a = N[1:end-1], ϕ[1:end-1], dϕdN[1:end-1], a[1:end-1]
    @info "Number of efolds in inflation: $(N[end] - N[1])"
    @info "End of inflation: ϕ = $(ϕ[end]), ϵ₁=$(dϕdN[end]^2/2.0), dϕdN=$(dϕdN[end])"
    # @info "Field velocity according to slow roll approx.: dϕdN=$(-p[2](ϕ[end])/p[1](ϕ[end]))"
    
    # due to numerical nature, very small negative number can be produced for H^2
    # H = sqrt.(max.(0, get_H2.(ϕ, dϕdN, p[1])))
    H = sqrt.(get_H2.(ϕ, dϕdN, p[1]))
    τ = cumul_integrate(N, @. 1 / (a * H))
    dϕdτ = @. a * H * dϕdN
    
    return τ, ϕ, dϕdτ, a, H, a[end], H[end]
end
end
