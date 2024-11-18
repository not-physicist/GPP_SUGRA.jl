"""
module used to compute the isocurvature power spectrum of the DM
"""
module Isocurvature 
using LinearInterpolations, MultiQuad, NumericalIntegration
using ..Commons

export get_Δ2_quad
export get_Δ2_disc, get_Δ2_disc_only_β
export get_Δ2_χ, get_Δ2_χ_only_β

"""
implement beta^4 first
"""
function _get_integrand(q::Float64, p::Float64, α, β, Ω)
    # return q * p * (abs2(β(p)) * abs2(β(q)) + real(conj(α(p)) * β(p) * α(q) * conj(β(q)) * exp(2.0im * Ω(p) - 2.0im * Ω(q)) ))
    return q * p * abs2(β(p)) * abs2(β(q))
end

"""
compute the isocurvature power spectrum 
k is both the momenta of α, β but also the output (can be easily changed)
"""
function get_Δ2_quad(k::Vector, α::Vector, β::Vector, Ω::Vector, a4ρ::Float64, a::Float64, m2::Float64)
    # α, β, Ω at the IR and UV side should be negligible
    get_α = LinearInterpolations.Interpolate(k, α, extrapolate=LinearInterpolations.Constant(0.0))
    get_β = LinearInterpolations.Interpolate(k, β, extrapolate=LinearInterpolations.Constant(0.0))
    get_Ω = LinearInterpolations.Interpolate(k, Ω, extrapolate=LinearInterpolations.Constant(0.0))
    # get_α = LinearInterpolations.Interpolate(k, α, extrapolate=LinearInterpolations.Replicate())
    # get_β = LinearInterpolations.Interpolate(k, β, extrapolate=LinearInterpolations.Replicate())
    # get_Ω = LinearInterpolations.Interpolate(k, Ω, extrapolate=LinearInterpolations.Replicate())

    # @show typeof(k) typeof(α) typeof(β) typeof(Ω)
    # @show typeof(get_α) typeof(get_β) typeof(get_Ω)
    
    Δ2 = Array{Float64, 1}(undef, size(k))
    @inbounds Threads.@threads for i in eachindex(k)
        _f(y, x) = _get_integrand(y, x, get_α, get_β, get_Ω)
        # setting the outer integral boundaries to 0, Inf leads to NaN results
        res = dblquad(_f, k[1], k[end], z->abs(k[i] - z), z->k[i] + z, rtol=1e-5)
        # @show res
        @inbounds Δ2[i] = res[1]
    end
    # lose one factor of k doing the momentum relabeling
    Δ2 = Δ2 .* @. a^2 * m2 / ((a4ρ)^2) * 2*k^2 / (2*π)^4

    return Δ2
end

###################################################################################################3
# try discrete integration
###################################################################################################3
"""
b, c, d are indices for k, p, q
"""
function get_Δ2_disc(k::Vector, α::Vector, β::Vector, Ω::Vector, a4ρ::Float64, aₑ::Float64, m2::Float64)
    function get_inner_integrand(c::Int64, d::Int64)
        # return abs2(get_χ(α[c], β[c], Ω[c], ω[c])) * abs2(get_χ(α[d], β[d], Ω[d], ω[d]))
        return k[c] * k[d] * (abs2(β[c]) * abs2(α[d]) + real( conj(α[c])*β[c]*α[d]*conj(β[d])*exp(2.0im*Ω[c] - 2.0im*Ω[d])) )
    end
    
    res = zeros(size(k))
    Threads.@threads for x in eachindex(k)  # k[x] = k
        pref = aₑ^2 * m2 / a4ρ^2 * 2*k[x]^2 / (2*π)^4
        res[x] = pref * double_trap(get_inner_integrand, 0, Inf, z->abs(k[x] - z), z->k[x] + z, k, k)
    end
    # @show res
    return res
end


function get_Δ2_disc_only_β(k::Vector, α::Vector, β::Vector, Ω::Vector, a4ρ::Float64, aₑ::Float64, m2::Float64)
    # @show k, β
    function get_inner_integrand(p, q)
        # return k[c] * k[d] * (abs2(β[c]) * abs2(β[d]))
        return p * q * (abs2(β(p)) * abs2(β(q)))
    end
    
    res = zeros(size(k))
    Threads.@threads for x in eachindex(k)  # k[x] = k
        pref = aₑ^2 * m2 / a4ρ^2 * k[x]^2 / (2*π)^4
        res[x] = pref * double_trap(get_inner_integrand, 0, Inf, z->abs(k[x] - z), z->k[x] + z, k, k)
    end
    return res
end


#########################################################################################################
# use formula with mode functions
#########################################################################################################

"""
a, b, c are indices
"""
function get_Δ2_χ(k::Vector, χ::Vector, ∂χ::Vector, a4ρ::Float64, aₑ::Float64, m::Float64)
    function get_inner_int(b::Int64, c::Int64)
        return (k[b] * k[c] * ( abs2(∂χ[b])*abs2(∂χ[c]) + aₑ^4*m^4*abs2(χ[b])*abs2(χ[c]) + aₑ^2*m^2*(χ[b]*conj(∂χ[b])*χ[c]*conj(∂χ[c]) + conj(χ[b])*∂χ[b]*conj(χ[c])*∂χ[c])))
    end
    res = zeros(size(k))
    Threads.@threads for x in eachindex(k)  # k[x] = k
        pref = 1 / a4ρ^2 * k[x]^2 /((2π)^4)
        # @show a4ρ
        res[x] = pref * double_trap(get_inner_int, 0, Inf, z->abs(k[x] - z), z->k[x] + z, k, k)
        # res[x] = pref * dblquad(get_inner_int, 0, Inf, z->abs(k[x] - z), z->k[x] + z)
    end
    return res
end

"""
a, b, c are indices 
only consider β^2 β^2 term
"""
function get_Δ2_only_β(k::Vector, f::Vector, a4ρ::Float64, aₑ::Float64, m::Float64)
    function get_inner_int(b::Int64, c::Int64)
        return k[b] * k[c] * (f[b]*f[c])
    end
    res = zeros(size(k))
    Threads.@threads for x in eachindex(k)  # k[x] = k
        pref = 2*aₑ^2 / a4ρ^2 * k[x]^2 /((2π)^4)
        # @show a4ρ
        res[x] = pref * double_trap(get_inner_int, 0, Inf, z->abs(k[x] - z), z->k[x] + z, k, k)
    end
    return res
end

end
