"""
Solve equation of motion for inflaton field
"""
module EOMs

using StaticArrays, Logging

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
    τ1, ϕ1, dϕdτ1, a1, aₑ, Hₑ = EFolds.solve_eom(u₀, p)
    @info "End of inflation: ϕ = $(ϕ1[end])"
    τ1 = τ1 .- τ1[end]
    # @show τ1[end-10:end] ϕ1[end-10:end]
    # @show size(get_others(τ1, ϕ1, dϕdτ1, a1, p[1])[6])

    # start with τ = -1/H is ok
    @show τ1[1], -1/Hₑ
    
    u₁ = SA[ϕ1[end], dϕdτ1[end], a1[end]]
    @info "Initial conditions for the second part of EOM" u₁
    tspan = (τ1[end], - τ1[1])
    τ2, ϕ2, dϕdτ2, a2 = Conformals.solve_eom(u₁, tspan, p)
    # @show get_others(τ2, ϕ2, dϕdτ2, a2, p[1])[6][1:10]

    τ = vcat(τ1, τ2)
    ϕ = vcat(ϕ1, ϕ2)
    dϕdτ = vcat(dϕdτ1, dϕdτ2)
    a = vcat(a1, a2)

    # to return a flattened tuple
    # @show get_others(τ, ϕ, dϕdτ, a, p[1])[6][3720:3750]
    return (get_others(τ, ϕ, dϕdτ, a, p[1])..., aₑ, Hₑ)
end

function solve_eom_conf_only(u₀::SVector, p::Tuple, τᵢ::Float64)
    tspan = (τᵢ, -τᵢ)
    τ, ϕ, dϕdτ, a, H = Conformals.solve_eom(u₀, tspan, p)
    τ, ϕ, dϕdτ, a, app_a, H, err = get_others(τ, ϕ, dϕdτ, a, H, p[1])
    # @show size(τ), size(H), size(a)
    # @show size(diff(H)), size(diff(τ))

    
    # first slow roll
    @show size(H), size(τ), size(a)
    ϵ₁ = - diff(H) ./ diff(τ) ./ (a .* H.^2)[1:end-1]
    end_index = findfirst(x -> abs(x) < 0.1, ϵ₁)
    aₑ = a[end_index]
    Hₑ = H[end_index]

    return (τ, ϕ, dϕdτ, a, app_a, H, err, aₑ, Hₑ)
end

end
