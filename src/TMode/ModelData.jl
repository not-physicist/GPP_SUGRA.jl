"""
module for generating struct storing model parameters (from observables)
"""
module ModelDatas

using ...Commons

export TMode, save_model_data

struct TMode{T<:Real, N<:Int}
    n::N
    nₛ::T
    r::T

    # derivated quantities
    V₀::T
    α::T
    ϕ_cmb::T  # field value at cmb pivot scale
    ϕₑ::T  # field value at the end of slow-roll inflation
    mᵩ::T

    # parameter for ODE solver 
    ϕᵢ::T
end
TMode(n, nₛ, r, ϕᵢ) = TMode(n, nₛ, r, get_derived(n, nₛ, r)..., ϕᵢ)

"""
compute field value corresponding to the CMB pivot scale 
"""
function get_ϕ_cmb(n::Int, nₛ::Real, α::Real)
    RHS = 1 / (2*(1-nₛ)) * (1 - nₛ + 4n/(3α) + sqrt((1-nₛ)^2 + 8n^2/(3α)*(1-nₛ) + 16n^2/(9α^2)))
    RHS = sqrt(RHS)
    return acosh(RHS) * sqrt(6*α)
end

"""
first slow roll parameter
"""
function get_ϵV(ϕ::Real, n::Int, α::Real)
    x = ϕ / (sqrt(6α))
    dV_V = 2 * n / (cosh(x) * sinh(x)) / sqrt(6α)
    return dV_V^2 / 2
end

function get_ηV(ϕ::Real, n::Int, α::Real)
    t = tanh(ϕ/sqrt(6*α))
    if n == 1
        return (2-8t^2 + 6*t^4) / t^2 / (6*α)
    else
        throw(ErrorException("Not implemented!"))
    end
end

function get_V₀(n::Int, ϕ_cmb::Real, α::Real)
    A = 2.1e-9
    ϵ = get_ϵV(ϕ_cmb, n, α)
    #  @show ϵ
    x = ϕ_cmb / (sqrt(6α))
    V₀ = A * 24 * pi^2 * ϵ / tanh(x)^(2*n)
    return V₀
end

function get_α(n::Int, nₛ::Real, r::Real)
    return 64/3 * n^2 * r / (n^2*(8*(1-nₛ) - r)^2 - r^2)
end

"""
compute all the derived quantities: V₀, α, ϕ_cmb and etc
"""
function get_derived(n::Int, nₛ::Real, r::Real)
    α = get_α(n, nₛ, r)
    ϕ_cmb = get_ϕ_cmb(n, nₛ, α)
    V₀ = get_V₀(n, ϕ_cmb, α)
    ϕₑ = get_ϕₑ(α, n)
    mᵩ = sqrt(V₀ / (3α))
    return V₀, α, ϕ_cmb, ϕₑ, mᵩ
end

function get_αₗ(n::Int)
    return 2n / (2n + sqrt(4n^2 - 1)) / 3
end

"""
get field value at the end of slow-roll inflation;
first implement r > r_l case (λ < λ_l, and ϵ=1 first)
"""
function get_ϕₑ(α::Real, n::Int)
    if α > get_αₗ(n)
        RHS = (1 + sqrt(1 + 4/3*n^2/α)) / 2
    else
        RHS = (1 + 2*n/(3*α) + sqrt(1 + 8*n^2 / (3*α) * (1/(6*α) - 1))) / 2
    end
    RHS = sqrt(RHS)
    ϕₑ = acosh(RHS) * sqrt(6*α)
    return ϕₑ
end

"""
test if one of the slow roll parameters is close to unity at ϕₑ
for a range of r; consistency check for ϕₑ
"""
function test_ϕₑ()
    r = logspace(-5, -1, 100)
    
    models = [TMode(1, 0.965, x) for x in r]
    ϵV = [get_ϵV(model.ϕₑ, model.n, model.α) for model in models]
    ηV = [get_ηV(model.ϕₑ, model.n, model.α) for model in models]
    
    # difference between 1 and the SR parameter closest to 1
    diff = (1 .- ϵV) .* (1 .- abs.(ηV))

    if isapprox(abs.(diff), fill(0.0, size(diff)), atol=1e-5)
        return true
    else
        return false
    end
end

"""
save n, ns, r into plain text; just for reference, accuracy not that important
"""
function save_model_data(model::TMode, fn::String)
    io = open(fn, "w")
    write(io, "n=$(model.n)\nn_s=$(model.nₛ)\nr=$(model.r)\nphi_i=$(model.ϕᵢ)")
    close(io)
end
end
