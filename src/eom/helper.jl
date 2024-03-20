module Helpers
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

end
