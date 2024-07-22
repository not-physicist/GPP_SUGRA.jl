"""
module used to compute the isocurvature power spectrum of the DM
"""
module Isocurvature 
using LinearInterpolations, MultiQuad

export get_Δ2

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
compute the integrals over k' and cosθ
"""
function _compute_int(k::Float64, α::T, β::T, Ω::N, kIR::Float64, kUV::Float64) where {T, N <: Interpolate}
    res = dblquad((x, y)->_get_integrand(x, y, k, α, β, Ω), -1.0, 1.0, kIR, kUV, rtol=1e-5)
    @show res
    return res[1]
end

"""
compute the isocurvature power spectrum 
k is both the momenta of α, β but also the output (can be easily changed)
"""
function get_Δ2(k::Vector, α::Vector, β::Vector, Ω::Vector{T}, a4ρ::Float64, a::Float64, m2::Float64) where {T <: Interpolate}
    get_α = LinearInterpolations.Interpolate(k, α, extrapolate=LinearInterpolations.Constant(0.0))
    get_β = LinearInterpolations.Interpolate(k, β, extrapolate=LinearInterpolations.Constant(0.0))
    
    Δ2 = @. ones(size(k)) * m2 / (a^2 * (a4ρ)^2) * k^3 / (2*π^2)
    @inbounds for i in eachindex(k)
        @inbounds Δ2[i] *= _compute_int(k[i], get_α, get_β, Ω[i], k[1], k[end])
    end

    return Δ2
end

end
