"""
module used to compute the isocurvature power spectrum of the DM
"""
module Isocurvature 
using LinearInterpolations, MultiQuad, NumericalIntegration

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
    @show k, res
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
function get_χ(α::ComplexF64, β::ComplexF64, Ω::Real, ω::ComplexF64)
    return (α*exp(-1.0im*Ω) + β*exp(+1.0im*Ω) ) / sqrt(2*ω)
end


function get_Δ2_χ(k::Vector, α::Vector, β::Vector, Ω::Vector, ω::Vector,
                  a4ρ::Float64, a::Float64, m2::Float64)
    """
    first integral (over q).
    """
    function get_inner_integrand(b::Int64, c::Int64)
        return k[c] * abs2(get_χ(α[b], β[b], Ω[b], ω[b])) * abs(get_χ(α[c], β[c], Ω[c], ω[c]))
    end
    
    function get_inner_int(a::Int64, b::Int64)
        # indices for the inner integral 
        # @show k[a], k[b]
        i_start = findfirst(x -> x > abs(k[a] - k[b]), k)
        i_end = findlast(x -> x < k[a] + k[b], k)
        # @show i_start, i_end
        
        return integrate(k[i_start:i_end], [get_inner_integrand(a, x) for x in i_start:i_end])
        #=
        # cound change this to 
        inner_int = (get_inner_integrand(b, i_start) + get_inner_integrand(b, i_end)) / 2.0
        for c in eachindex(k[i_start+1:i_end-1])  # k[c] = q
            # LIMIT WRONG!
            inner_int += get_inner_integrand(b, c)
        end
        inner_int *= Δk
        =#
    end

    function get_outer_int(a::Int64)
        i_start = 1
        i_end = size(k)[1]
        # @show i_start, i_end

        return integrate(k[i_start:i_end], [get_inner_int(a, x) for x in i_start:i_end])
    end

    res = zeros(size(k))
    
    @inbounds Threads.@threads for x in eachindex(k)  # k[x] = k
        pref = a^2 * m2^2 / a4ρ^2 * k[x]^2 / (2*π)^4
        @inbounds res[x] = pref * get_outer_int(x)
    end
    return res
end

end
