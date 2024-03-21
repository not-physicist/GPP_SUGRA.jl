"""
Solve equation of motion for inflaton field
"""
module EOMs

using StaticArrays

include("conf.jl")
using .Conformals

include("efold.jl")
using .EFolds

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
Solving EOM using efolds and conformal time
"""
function solve_eom(u₀, p)
    τ1, ϕ1, dϕdτ1, a1 = EFolds.solve_eom(u₀, p)
    τ1 = τ1 .- τ1[end]
    #  @show τ1[end-10:end] ϕ1[end-10:end]

    u₁ = SA[ϕ1[end], dϕdτ1[end], a1[end]]
    #  @show u₁
    tspan = (τ1[end], - τ1[1])
    τ2, ϕ2, dϕdτ2, a2 = Conformals.solve_eom(u₁, tspan, p)
    #  @show τ2[end-10:end] ϕ2[end-10:end]

    τ = vcat(τ1, τ2)
    ϕ = vcat(ϕ1, ϕ2)
    dϕdτ = vcat(dϕdτ1, dϕdτ2)
    a = vcat(a1, a2)

    return get_others(τ, ϕ, dϕdτ, a, p[1])
end

end