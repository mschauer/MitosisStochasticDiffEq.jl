# from linearalgebra.jl

lchol(Σ) = cholesky(Symmetric(Σ)).U'
_logdet(Σ, d::Integer) = LinearAlgebra.logdet(Σ)

# from wgaussian.jl

import Statistics: mean, cov
import Random.rand
struct WGaussian{P,T} # <: AbstractMeasure
    par::NamedTuple{P,T}
end
WGaussian{P}(nt::NamedTuple{P,T}) where {P,T} = WGaussian{P,T}(nt)
WGaussian{P}(args...) where {P} = WGaussian(NamedTuple{P}(args))
WGaussian(; args...) = WGaussian(args.data)
WGaussian{P}(; args...) where {P} = WGaussian{P}(args.data)

# the following propagates uncertainty if `μ` is `Gaussian`
WGaussian(par::NamedTuple{(:μ, :Σ, :c),Tuple{T,S,U}}) where {T<:WGaussian,S,U} = WGaussian((; μ=mean(par.μ), Σ=par.Σ + cov(par.μ), c=par.c + par.μ.c))
WGaussian{P}(par::NamedTuple{(:μ, :Σ, :c),Tuple{T,S,U}}) where {P,T<:WGaussian,S,U} = WGaussian{P}(mean(par.μ), par.Σ + cov(par.μ), par.c + par.μ.c)

Base.getproperty(p::WGaussian, s::Symbol) = getproperty(getfield(p, :par), s)

const WGaussianOrNdTuple{P} = Union{WGaussian{P},NamedTuple{P}}

Base.keys(p::WGaussian{P}) where {P} = P
params(p::WGaussian) = getfield(p, :par)

dim(p::WGaussian{(:F, :Γ, :c)}) = length(p.F)

mean(p::WGaussian{(:μ, :Σ, :c)}) = p.μ
cov(p::WGaussian{(:μ, :Σ, :c)}) = p.Σ

mean(p::WGaussian{(:F, :Γ, :c)}) = p.Γ \ p.F
cov(p::WGaussian{(:F, :Γ, :c)}) = inv(p.Γ)
moment1(p::WGaussian{(:F, :Γ, :c)}) = p.c * p.Γ \ p.F

Base.isapprox(p1::WGaussian, p2::WGaussian; kwargs...) =
    all(isapprox.(Tuple(params(p1)), Tuple(params(p2)); kwargs...))

function logdensityof(p::WGaussian{(:F, :Γ, :c)}, x)
    C = cholesky_(sym(p.Γ))
    p.c - x' * p.Γ * x / 2 + x' * p.F - p.F' * (C \ p.F) / 2 + logdet(C) / 2 - dim(p) * log(2pi) / 2
end
densityof(p::WGaussian, x) = exp(logdensityof(p, x))

StatsBase.rand(p::WGaussian) = rand(Random.GLOBAL_RNG, p)
StatsBase.rand(RNG::AbstractRNG, p::WGaussian{(:μ, :Σ, :c)}) = weighted(unwhiten(Gaussian{(:μ, :Σ)}(p.μ, p.Σ), randn!(RNG, zero(mean(p)))), p.c)
weighted(p::WGaussian{(:μ, :Σ, :c)}, ll) = WGaussian{(:μ, :Σ, :c)}(p.μ, p.Σ, p.c + ll)

function Base.convert(::Type{WGaussian{(:F, :Γ, :c)}}, p::WGaussian{(:μ, :Σ, :c)})
    Γ = inv(p.Σ)
    return WGaussian{(:F, :Γ, :c)}(Γ * p.μ, Γ, p.c)
end

function Base.convert(::Type{WGaussian{(:μ, :Σ, :c)}}, p::WGaussian{(:F, :Γ, :c)})
    Σ = inv(p.Γ)
    return WGaussian{(:μ, :Σ, :c)}(Σ * p.F, Σ, p.c)
end

struct Leaf{T}
    y::T
end
Base.getindex(y::Leaf) = y.y

Base.convert(::Type{WGaussian{(:μ, :Σ, :c)}}, p::Leaf) = convert(WGaussian{(:μ, :Σ, :c)}, p[])
Base.convert(::Type{WGaussian{(:F, :Γ, :c)}}, p::Leaf) = convert(WGaussian{(:F, :Γ, :c)}, p[])


# from gauss.jl
import LinearAlgebra.logdet

"""
    Gaussian{(:μ,:Σ)}
    Gaussian{(:F,:Γ)}

Mitosis provides the measure `Gaussian` based on MeasureTheory.jl,
with a mean `μ` and covariance `Σ` parametrization,
or parametrised by natural parameters `F = Γ μ`, `Γ = Σ⁻¹`.

# Usage:

    Gaussian(μ=m, Σ=C)
    p = Gaussian{(:μ,:Σ)}(m, C)
    Gaussian(F=C\\m, Γ=inv(C))

    convert(Gaussian{(:F,:Γ)}, p)

    rand(rng, p)
"""
struct Gaussian{P,T} #<: AbstractMeasure
    par::NamedTuple{P,T}
end
Gaussian{P}(nt::NamedTuple{P,T}) where {P,T} = Gaussian{P,T}(nt)
Gaussian{P}(args...) where {P} = Gaussian(NamedTuple{P}(args))
Gaussian(; args...) = Gaussian(args.data)
Gaussian{P}(; args...) where {P} = Gaussian{P}(args.data)

# the following propagates uncertainty if `μ` is `Gaussian`
Gaussian(par::NamedTuple{(:μ, :Σ),Tuple{T,S}}) where {T<:Gaussian,S} = Gaussian((; μ=mean(par.μ), Σ=par.Σ + cov(par.μ)))
Gaussian{P}(par::NamedTuple{(:μ, :Σ),Tuple{T,S}}) where {P,T<:Gaussian,S} = Gaussian{P}((; μ=mean(par.μ), Σ=par.Σ + cov(par.μ)))

Base.getproperty(p::Gaussian, s::Symbol) = getproperty(getfield(p, :par), s)

const GaussianOrNdTuple{P} = Union{Gaussian{P},NamedTuple{P}}

Base.keys(p::Gaussian{P}) where {P} = P
params(p::Gaussian) = getfield(p, :par)
## Basics

Base.:(==)(p1::Gaussian, p2::Gaussian) = mean(p1) == mean(p2) && cov(p1) == cov(p2)
Base.isapprox(p1::Gaussian, p2::Gaussian; kwargs...) =
    isapprox(mean(p1), mean(p2); kwargs...) && isapprox(cov(p1), cov(p2); kwargs...)

mean(p::Gaussian{(:μ, :Σ)}) = p.μ
mean(p::Gaussian{(:Σ,)}) = Zero()
cov(p::Gaussian{(:μ, :Σ)}) = p.Σ
meancov(p) = mean(p), cov(p)

precision(p::Gaussian{(:μ, :Σ)}) = inv(p.Σ)

mean(p::Gaussian{(:F, :Γ)}) = p.Γ \ p.F
cov(p::Gaussian{(:F, :Γ)}) = inv(p.Γ)
precision(p::Gaussian{(:F, :Γ)}) = p.Γ
norm_sqr(x) = dot(x, x)
dim(p::Gaussian{(:F, :Γ)}) = length(p.F)
dim(p::Gaussian) = length(mean(p))
dim(p::Gaussian{(:Σ,)}) = size(p.Σ, 1)
whiten(p::Gaussian{(:μ, :Σ)}, x) = lchol(p.Σ) \ (x - p.μ)
unwhiten(p::Gaussian{(:μ, :Σ)}, z) = lchol(p.Σ) * z + p.μ
function whiten(p::Gaussian{(:F, :Γ)}, x)
    L = lchol(p.Γ)
    L * x - L' \ p.F
end
function unwhiten(p::Gaussian{(:F, :Γ)}, z)
    L = lchol(p.Γ)
    L \ (z + L' \ p.F)
end
whiten(p::Gaussian{(:Σ,)}, x) = lchol(p.Σ) \ x
unwhiten(p::Gaussian{(:Σ,)}, z) = lchol(p.Σ) * z
sqmahal(p::Gaussian, x) = norm_sqr(whiten(p, x))

rand(p::Gaussian) = rand(Random.GLOBAL_RNG, p)
randwn(rng::AbstractRNG, x::Vector) = randn!(rng, zero(x))
function randwn(rng::AbstractRNG, x)
    map(xi -> randn(rng, typeof(xi)), x)
end
rand(rng::AbstractRNG, p::Gaussian) = unwhiten(p, randwn(rng, mean(p)))

_logdet(p::Gaussian{(:μ, :Σ)}) = _logdet(p.Σ, dim(p))
_logdet(p::Gaussian{(:Σ,)}) = logdet(p.Σ)
logdensityof(p::Gaussian, x) = -(sqmahal(p, x) + _logdet(p) + dim(p) * log(2pi)) / 2
densityof(p::Gaussian, x) = exp(logdensityof(p, x))
function logdensityof(p::Gaussian{(:F, :Γ)}, x)
    C = cholesky(Symmetric(p.Γ))
    -x' * p.Γ * x / 2 + x' * p.F - p.F' * (C \ p.F) / 2 + logdet(C) / 2 - dim(p) * log(2pi) / 2
end

function logdensity0(p::Gaussian{(:F, :Γ)})
    C = cholesky(Symmetric(p.Γ))
    -p.F' * (C \ p.F) / 2 + logdet(C) / 2 - dim(p) * log(2pi) / 2
end


function Base.convert(::Type{Gaussian{(:F, :Γ)}}, p::Gaussian{(:μ, :Σ)})
    Γ = inv(p.Σ)
    return Gaussian{(:F, :Γ)}(Γ * p.μ, Γ)
end
function Base.convert(::Type{Gaussian{(:μ, :Σ)}}, p::Gaussian{(:F, :Γ)})
    Σ = inv(p.Γ)
    return Gaussian{(:μ, :Σ)}(Σ * p.F, Σ)
end

## Algebra

Base.:+(p::Gaussian{P}, x) where {P} = Gaussian{P}(mean(p) + x, p.par[2])
Base.:+(x, p::Gaussian) = p + x

Base.:-(p::Gaussian, x) = p + (-x)
Base.:*(M, p::Gaussian{P}) where {P} = Gaussian{P}(M * mean(p), Σ=M * cov(p) * M')

⊕(p1::Gaussian{(:μ, :Σ)}, p2::Gaussian{(:μ, :Σ)}) = Gaussian{(:μ, :Σ)}(p1.μ + p2.μ, p1.Σ + p2.Σ)
⊕(x, p::Gaussian) = x + p
⊕(p::Gaussian, x) = p + x

## Conditionals and filtering

"""
    conditional(p::Gaussian, A, B, xB)

Conditional distribution of `X[i for i in A]` given
`X[i for i in B] == xB` if ``X ~ P``.
"""
function conditional(p::Gaussian{(:μ, :Σ)}, A, B, xB)
    Z = p.Σ[A, B] * inv(p.Σ[B, B])
    Gaussian{(:μ, :Σ)}(p.μ[A] + Z * (xB - p.μ[B]), p.Σ[A, A] - Z * p.Σ[B, A])
end

# from Markov.jl

"""
    AffineMap(B, β)

Represents a function `f = AffineMap(B, β)`
such that `f(x) == B*x + β`.
"""
struct AffineMap{S,T}
    B::S
    β::T
end
(a::AffineMap)(x) = a.B * x + a.β
(a::AffineMap)(p::Gaussian) = Gaussian(μ=a.B * mean(p) + a.β, Σ=a.B * cov(p) * a.B')
(a::AffineMap)(p::WGaussian) = WGaussian(μ=a.B * mean(p) + a.β, Σ=a.B * cov(p) * a.B', c=p.c)


"""
    ConstantMap(β)

Represents a function `f = ConstantMap(β)`
such that `f(x) == β`.
"""
struct ConstantMap{T}
    x::T
end
(a::ConstantMap)(x) = a.x
(a::ConstantMap)() = a.x


# fusion

abstract type Context end

"""
    BFFG()

Backward filter forward guiding context for non-linear Gaussian
systems with `h` parametrized by `WGaussian{(:F,:Γ,:c)}`` (see Theorem 7.1 [Automatic BFFG].)
"""
struct BFFG <: Context
end

struct Copy{N}
end
(a::Copy{2})(x) = (x, x)

# from rules.jl

# fuse(a; kargs...) = backward(BFFG(), Copy{1}(), a; kargs...)
fuse(a, b; kargs...) = backward(BFFG(), Copy{2}(), a, b; kargs...)
# fuse(a, b, c; kargs...) = backward(BFFG(), Copy{3}(), a, b, c; kargs...)



function backward(::BFFG, ::Copy, args::Union{Leaf{<:WGaussian{(:μ, :Σ, :c)}},WGaussian{(:μ, :Σ, :c)}}...; unfused=true)
    unfused = false
    F, H, c = params(convert(WGaussian{(:F, :Γ, :c)}, args[1]))
    args[1] isa Leaf || (c += logdensity0(Gaussian{(:F, :Γ)}(F, H)))
    for b in args[2:end]
        F2, H2, c2 = params(convert(WGaussian{(:F, :Γ, :c)}, b))
        F += F2
        H += H2
        c += c2
        b isa Leaf || (c += logdensity0(Gaussian{(:F, :Γ)}(F2, H2)))
    end
    Δ = -logdensityof(Gaussian{(:F, :Γ)}(F, H), 0F)
    # message() = nothing
    nothing, convert(WGaussian{(:μ, :Σ, :c)}, WGaussian{(:F, :Γ, :c)}(F, H, Δ + c))
end


# function backward(::BFFG, ::Copy, a::Gaussian{(:F, :Γ)}, args...)
#     F, H = params(a)
#     for b in args
#         F2, H2 = params(b::Gaussian{(:F, :Γ)})
#         F += F2
#         H += H2
#     end
#     message(), Gaussian{(:F, :Γ)}(F, H)
# end

# function backward(::BFFG, ::Copy, a::Union{Leaf{<:WGaussian{(:F, :Γ, :c)}},WGaussian{(:F, :Γ, :c)}}, args...; unfused=true)
#     unfused = false
#     F, H, c = params(convert(WGaussian{(:F, :Γ, :c)}, a))
#     a isa Leaf || (c += logdensityof(Gaussian{(:F, :Γ)}(F, H), 0F))
#     for b in args
#         F2, H2, c2 = params(convert(WGaussian{(:F, :Γ, :c)}, b))
#         F += F2
#         H += H2
#         c += c2
#         b isa Leaf || (c += logdensityof(Gaussian{(:F, :Γ)}(F2, H2), 0F2))
#     end
#     Δ = -logdensityof(Gaussian{(:F, :Γ)}(F, H), 0F)
#     message(), WGaussian{(:F, :Γ, :c)}(F, H, Δ + c)
# end
