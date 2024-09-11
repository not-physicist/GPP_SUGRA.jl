"""
Solve equation of motion for inflaton field
"""
module EOMs

using StaticArrays, Logging, OrdinaryDiffEq

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
function get_others(τ::Vector, ϕ::Vector, dϕ::Vector, a::Vector, H::Vector, get_V::Function)
    ap = diff(a) ./ diff(τ) 
    app = diff(ap) ./ diff(τ[1:end-1]) 
    # app_a = app ./ a[1:end-2]
    app_a = @. 2.0/3.0 * a^2 * get_V(ϕ) - 1.0/6.0 * dϕ^2
    # cosmic Hubble
    # H = ap ./ (a[1:end-1] .^2)
    err = get_err(app, a, ϕ, dϕ, get_V)
    
    #  @show size(τ) size(ϕ) size(dϕ) size(a) size(ap) size(app) size(app_a) size(H) size(err)
    # trim arrays to have the identical dimension
    # once the step size is small enough, remove the last few elements should be OK
    # return τ[1:end-2], ϕ[1:end-2], dϕ[1:end-2], a[1:end-2], ap[1:end-1], app, app_a, H[1:end-1], err
    return τ[1:end-2], ϕ[1:end-2], dϕ[1:end-2], a[1:end-2], app_a[1:end-2], H[1:end-2], err
end

"""
Solving EOM using efolds and conformal time
"""
function solve_eom(u₀::SVector, p::Tuple)
    τ1, ϕ1, dϕdτ1, a1, H1, aₑ, Hₑ = EFolds.solve_eom(u₀, p)
    τ1 = τ1 .- τ1[end]
    # @show τ1[end-10:end] ϕ1[end-10:end]
    # @show size(get_others(τ1, ϕ1, dϕdτ1, a1, p[1])[6])
    # return (get_others(τ1, ϕ1, dϕdτ1, a1, H1, p[1])..., aₑ, Hₑ)

    # start with τ = -1/H is ok
    # @info τ1[1], -1/Hₑ

    dϕdτ_SR = -a1[end]*p[2](ϕ1[end])/sqrt(3*p[1](ϕ1[end]))
    u₁ = SA[ϕ1[end], dϕdτ1[end], a1[end]]
    # u₁ = SA[ϕ1[end], dϕdτ_SR, a1[end]]
    @info "Initial conditions for the second part of EOM" u₁
    @info "Field velocity according to slow roll approx.: dϕdτ=$(dϕdτ_SR)"

    # this is maximal time span, not accounting for the callback
    tspan = (τ1[end], - τ1[1])
    condition(u, t, integrator) = u[3] / u₀[3] <= 1e4
    affect!(integrator) = terminate!(integrator)
    cb2 = ContinuousCallback(condition, affect!)
    τ2, ϕ2, dϕdτ2, a2, H2 = Conformals.solve_eom(u₁, tspan, p, cb2)
    
    # combine two parts
    τ = vcat(τ1, τ2)
    ϕ = vcat(ϕ1, ϕ2)
    dϕdτ = vcat(dϕdτ1, dϕdτ2)
    a = vcat(a1, a2)
    H = vcat(H1, H2)

    return (get_others(τ, ϕ, dϕdτ, a, H, p[1])..., aₑ, Hₑ)
end

#=
"""
DEPRECATED
only using conformal time for solving background
seems bad if need large numbers of efolds of inflation 
"""
function solve_eom_conf_only(u₀::SVector, p::Tuple, tspan::Tuple)
    affect!(integrator) = terminate!(integrator)
    condition1(u, t, integrator) =  get_ϵ1(u...) <= 0.1
    get_Hubble(ϕ, dϕ, V, a) = sqrt(1 / 3 * (dϕ^2 / 2 + a^2 * V(ϕ)))/a
    get_ϵ1(ϕ, dϕ, a) = (dϕ / (a*get_Hubble(ϕ, dϕ, p[1], a)))^2/2
    cb1 = ContinuousCallback(condition1, affect!)

    condition2(u, t, integrator) = u[3] / u₀[3] <= 1e4
    cb2 = ContinuousCallback(condition2, affect!)
    
    @info "Initial condition" u₀
    τ1, ϕ1, dϕdτ1, a1, H1 = Conformals.solve_eom(u₀, tspan, p, cb1)
    # τ1, ϕ1, dϕdτ1, a1, app_a1, H1, err1 = get_others(τ1, ϕ1, dϕdτ1, a1, H1, p[1])
    @info "End of inflation: ϕ = $(ϕ1[end]), ϵ₁=$(get_ϵ1(ϕ1[end], dϕdτ1[end], a1[end])), dϕdN=$(dϕdτ1[end] / (a1[end] * H1[end]))"
    @info tspan

    #=
    # first slow roll
    # @show size(H), size(τ), size(a)
    ϵ₁ = - diff(H) ./ diff(τ) ./ (a .* H.^2)[1:end-1]
    # @show ϵ₁[1:10], ϵ₁[end-10:end]
    end_index = findfirst(x -> abs(x) > 0.05, ϵ₁)
    # @show a, end_index

    aₑ = a[1]
    Hₑ = H[1]
    try
        aₑ = a[end_index]
        Hₑ = H[end_index]
        @info aₑ, Hₑ, τ[end_index]
        @info "$(log(aₑ)) of efolds of inflation"
    catch e
        @warn e 
    end
    =#
    a1 = a1 ./ a1[end]
    aₑ = a1[end]
    Hₑ = H1[end]

    tspan = (τ1[end], - τ1[1])
    u₁ = SA[ϕ1[end], dϕdτ1[end], a1[end]]
    τ2, ϕ2, dϕdτ2, a2, H2 = Conformals.solve_eom(u₁, tspan, p, cb2)
    # @show get_others(τ2, ϕ2, dϕdτ2, a2, p[1])[6][1:10]

    τ = vcat(τ1, τ2)
    ϕ = vcat(ϕ1, ϕ2)
    dϕdτ = vcat(dϕdτ1, dϕdτ2)
    a = vcat(a1, a2)
    H = vcat(H1, H2)

    return (get_others(τ, ϕ, dϕdτ, a, H, p[1])..., aₑ, Hₑ)
end
=#
end
