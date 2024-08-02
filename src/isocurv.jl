"""
module used to compute the isocurvature power spectrum of the DM
"""
module Isocurvature 
using LinearInterpolations, MultiQuad, NumericalIntegration
using ..Commons

export get_Δ2, get_Δ2_χ

# const INTERP_TYPE = LinearInterpolations.Interpolate{typeof(LinearInterpolations.combine), Tuple{Vector{Float64}}, Vector{Float64}, Symbol}
# const INTERP_TYPE_COMP = LinearInterpolations.Interpolate{typeof(LinearInterpolations.combine), Tuple{Vector{Float64}}, Vector{ComplexF64}, Symbol}

"""
integrand of isocurvature power spectrum integral
only care the spectrum at final time (no time-dependence!)
"""
function _get_integrand(kPrime::Float64, cosθ::Float64, k::Float64, α::T, β::T, Ω::N) where {T, N <: Interpolate}
    kPrimeMinusk = sqrt(kPrime^2 + k^2 - 2 * k * kPrime * cosθ) 
    int = abs2(β(kPrime)) * abs2(α(kPrimeMinusk))
    int += real( conj(α(kPrime)) * β(kPrime) * α(kPrimeMinusk) * conj(β(kPrimeMinusk)) * exp(2.0im * Ω(kPrime) - 2.0im * Ω(kPrimeMinusk)) )
    int *= kPrime^2
    return int
end

"""
compute the integrals over k' and cosθ using interpolation and quadrature
"""
function _compute_int(k::Float64, α::T, β::T, Ω::N, kIR::Float64, kUV::Float64) where {T, N <: Interpolate}
    res = dblquad((x, y)->_get_integrand(x, y, k, α, β, Ω), -1.0, 1.0, kIR, kUV, rtol=1e-3)
    # @show k, res
    return res[1]
end

"""
compute the isocurvature power spectrum 
k is both the momenta of α, β but also the output (can be easily changed)
"""
function get_Δ2(k::Vector, α::Vector, β::Vector, Ω::Vector, a4ρ::Float64, a::Float64, m2::Float64)
    get_α = LinearInterpolations.Interpolate(k, α, extrapolate=LinearInterpolations.Constant(0.0))
    get_β = LinearInterpolations.Interpolate(k, β, extrapolate=LinearInterpolations.Constant(0.0))
    get_Ω = LinearInterpolations.Interpolate(k, Ω, extrapolate=LinearInterpolations.Constant(0.0))
    # @show typeof(k) typeof(α) typeof(β) typeof(Ω)
    # @show typeof(get_α) typeof(get_β) typeof(get_Ω)
    
    Δ2 = Array{Float64, 1}(undef, size(k))
    @inbounds Threads.@threads for i in eachindex(k)
        @inbounds Δ2[i] = _compute_int(k[i], get_α, get_β, get_Ω, k[1], k[end])
    end
    Δ2 = Δ2 .* @. m2 / (a^2 * (a4ρ)^2) * k^3 / (2*π^2)

    return Δ2
end

###################################################################################################3
# try discrete integration
###################################################################################################3
#= 
"""
mode functions
"""
function get_χ(α::ComplexF64, β::ComplexF64, Ω::Real, ω::ComplexF64)
    return (α*exp(-1.0im*Ω) + β*exp(+1.0im*Ω) ) / sqrt(2*ω)
end
=#

"""
b, c, d are indices for k, p, q
TODO: use Integrals.jl for the numerical integration
"""
function get_Δ2_χ(k::Vector, α::Vector, β::Vector, Ω::Vector, ω::Vector,
                  a4ρ::Float64, aₑ::Float64, m2::Float64, mᵩ::Float64)
    function get_inner_integrand(c::Int64, d::Int64)
        # return abs2(get_χ(α[c], β[c], Ω[c], ω[c])) * abs2(get_χ(α[d], β[d], Ω[d], ω[d]))
        return k[c] * k[d] * (abs2(β[c]) * abs2(α[d]) + real( conj(α[c])*β[c]*α[d]*conj(β[d])*exp(2.0im*Ω[c] - 2.0im*Ω[d])) )
    end
    
    res = zeros(size(k))
    Threads.@threads for x in eachindex(k)  # k[x] = k
        # pref = aₑ^4 * m2^2 / a4ρ^2 * k[x]^2 / (2*π)^4
        pref = aₑ^2 * m2 / a4ρ^2 * k[x]^2 / (2*π)^4
        # res[x] = pref * integrate(k, [get_inner_int(x, y) for y in eachindex(k)])
        res[x] = pref * double_trap(get_inner_integrand, 0, Inf, z->abs(k[x] - z), z->k[x] + z, k, k)
    end
    return res
end

end
