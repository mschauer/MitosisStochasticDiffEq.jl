using Pkg
path = @__DIR__
cd(path)
Pkg.activate(path)

using Mitosis
using MitosisStochasticDiffEq
using LinearAlgebra, Statistics, Random, StatsBase
using DelimitedFiles
using StaticArrays
using NewickTree
using DifferentialEquations
using DiffEqNoiseProcess
using Test

seed = 10
Random.seed!(seed)

const d = 2
const 𝕏 = SVector{d,Float64}
# needed for StochasticDiffEq package where we simulate d+1 states in forwardguiding
# to compute the likelihood.
#const 𝕏_ = SVector{d+1,Float64}
const MS = Mitosis
const MSDE = MitosisStochasticDiffEq

# using MeasureTheory
# import MeasureTheory.logdensity
MS.dim(p::WGaussian{(:μ,:Σ,:c)}) = length(p.μ)
MS._logdet(p::WGaussian{(:μ,:Σ,:c)}) = MS._logdet(p.Σ, MS.dim(p))
MS.whiten(p::WGaussian{(:μ,:Σ,:c)}, x) = MS.lchol(p.Σ)\(x - p.μ)
MS.sqmahal(p::WGaussian, x) = MS.norm_sqr(MS.whiten(p, x))
MS.logdensity(p::WGaussian{(:μ,:Σ,:c)}, x) = p.c - (MS.sqmahal(p,x) + MS._logdet(p) + MS.dim(p)*log(2pi))/2

include("tree.jl")
include("sdetree.jl")


## Read tree
tree = [Tree(S20coal), Tree(S)][1]
#print_tree(tree.newick)

## define model (two traits, as in presentation)
bθ = @SVector [.5, 1.2]
σ0 = @SVector [log(0.2), log(1.05)]
θ0 = (bθ, σ0)

#const M = [-1.0 1.0; 1.0 -1.0]
const M = SMatrix{d,d}(-1.0, 1.0, 1.0, -1.0)
fns(u,θ,t) = (tanh.(Diagonal(θ[1]) * M * u) )
f(u,θ,t) = 𝕏(tanh.(Diagonal(θ[1]) * M * u) )  # f(u,θ,t) = Diagonal(θ[1]) * M * u
gns(u,θ,t) = Diagonal((exp.(θ[2])))
dt0 = 0.001
u0ns = zeros(2)
u0 = zero(𝕏)  # value at root node

## forward sample no tree non-static arrays
ts = 0.0:dt0:1.0
Wsns = cumsum([[zero(u0ns)];[sqrt(ts[i+1]-ti)*randn(size(u0ns))
  for (i,ti) in enumerate(ts[1:end-1])]])
NGns = NoiseGrid(ts,Wsns)
κns = MSDE.SDEKernel(fns, gns, ts, θ0, zeros(d,d))
x1ns, xT1ns = MSDE.sample(κns, u0ns, MSDE.EulerMaruyama!(), NGns)
x2ns, xT2ns = MSDE.sample(κns, u0ns, EM(false), NGns)
@test xT1ns[end] ≈ xT2ns

## forward sample no tree static arrays
ts = 0.0:dt0:1.0
Ws = 𝕏.(Wsns)
NG = NoiseGrid(ts,Ws)
κ = MSDE.SDEKernel(f, g, ts, θ0, zeros(d,d))
x1, xT1 = MSDE.sample(κ, u0, MSDE.EulerMaruyama!(), NG)
x2, xT2 = MSDE.sample(κ, u0, EM(false), NGns) # needs oop
@test xT1[end] ≈ xT1ns[end]
@test xT1[end] ≈ xT2

NG = innov(ts, 𝕏)
x1, xT1 = MSDE.sample(κ, u0, MSDE.EulerMaruyama!(), NG)
x2, xT2 = MSDE.sample(κ, u0, EM(false), NG)
@test xT1[end] ≈ xT2
# dpesn't have same randomness?
# Random.seed!(seed)
# x1, xT1 = MSDE.sample(κ, u0, MSDE.EulerMaruyama!())
# Random.seed!(seed)
# x2, xT2 = MSDE.sample(κ, u0, EM(false); save_noise=true)
# @test xT1[end] ≈ xT1ns[end]
# @test xT1[end] ≈ xT2

## forward sample on the tree to simulate the observations at the tips

Random.seed!(10)
Xd1, segs1 = forwardsample(tree, u0, θ0, dt0, f, g, MSDE.EulerMaruyama!())
Random.seed!(10)
Xd2, segs2 = forwardsample(tree, u0, θ0, dt0, f, g, EM(false))

@test getindex.(segs1[2],3) ≈ segs2[2].u
@test getindex.(segs1[3],2) == segs2[3].t
@show getindex.(segs1[3],3)[258:260] - segs2[3].u[258:260]
