"""
Solve equation of motion for inflaton field
"""
module EOMs

using StaticArrays

include("helper.jl")
using .Helpers

include("conf.jl")
using .Conformals

include("efold.jl")
using .EFolds

"""
Solving EOM using efolds and conformal time
"""
function solve_ode(u₀, p)
    τ1, ϕ1, dϕdτ1, a1 = EFolds.solve_ode(u₀, p)
    τ1 = τ1 .- τ1[end]
    #  @show τ1[end-50:end] ϕ1[end-50:end]

    u₁ = SA[ϕ1[end], dϕdτ1[end], a1[end]]
    #  @show u₁
    tspan = (τ1[end], - τ1[1])
    τ2, ϕ2, dϕdτ2, a2 = Conformals.solve_ode(u₁, tspan, p)
    #  @show τ2[end-10:end] ϕ2[end-10:end]

    τ = vcat(τ1, τ2)
    ϕ = vcat(ϕ1, ϕ2)
    dϕdτ = vcat(dϕdτ1, dϕdτ2)
    a = vcat(a1, a2)

    return Helpers.get_others(τ, ϕ, dϕdτ, a, p[1])
end

end
