using MitosisStochasticDiffEq
using Mitosis
using Test, Random
using LinearAlgebra

# set true/estimated model parameters
# p = [-0.1,0.2,0.9]

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3]

# time span
tstart = 0.0
tend = 1.0
dt = 0.02
trange = tstart:dt:tend

# intial condition
u0 = 1.1

# set of linear parameters Eq.~(2.2)
B, β, σ̃ = -0.1, 0.2, 1.3
plin = [B, β, σ̃]
kernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,plin)


# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, 0.0, 10.0];
NT = NamedTuple{mynames}(myvalues)

message, solend = MitosisStochasticDiffEq.backwardfilter(kernel, NT)

kernel2 = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel2, NT)

@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)

"""
    backwardfilter() -> ps, p0, c
Backward filtering using the Euler method, starting with `N(ν, P)` prior
and integration scale c2 between observations

"""
function backwardfilter((c, ν, P)::NamedTuple{(:logscale, :μ, :Σ)}, p, s)
    ps = [[ν, P, c]]
    B, β, σ̃ = p
    for i in eachindex(s)[end-1:-1:1]
        dt = s[i+1] - s[i]
        H = inv(P)
        F = H*ν
        P = P - dt*(B*P + P*B' .- σ̃*σ̃')
        ν = ν - dt*(B*ν .+ β)
        c = c - dt*tr(B)
        push!(ps, [ν, P, c])
    end
    ps, [ν, P, c]
end


message2, solend2 = backwardfilter(NT, plin, message.ts)

@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), reduce(hcat, message2), rtol=1e-15)


# multivariate tests
dim = 5
Random.seed!(123)
logscale = randn()
μ = randn(dim)
Σ = randn(dim,dim)
myvalues = [logscale, μ, Σ];
NT = NamedTuple{mynames}(myvalues)

kernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,plin)
message, solend = MitosisStochasticDiffEq.backwardfilter(kernel, NT)
message2, solend2 = backwardfilter(NT, plin, message.ts)

@test isapprox(solend.x[1], solend2[1], rtol=1e-15)
@test isapprox(solend.x[2], solend2[2], rtol=1e-15)
@test isapprox(solend.x[3][1], solend2[3], rtol=1e-15)

kernel2 = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel2, NT)
@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)

# test inplace version
message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT, inplace=true)
@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)

message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel2, NT, inplace=true)
@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)

m = 3 # some number of Brownian processes
plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil

kernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,plin)
message, solend = MitosisStochasticDiffEq.backwardfilter(kernel, NT)
message2, solend2 = backwardfilter(NT, plin, message.ts)

@test isapprox(solend.x[1], solend2[1], rtol=1e-15)
@test isapprox(solend.x[2], solend2[2], rtol=1e-14)
@test isapprox(solend.x[3][1], solend2[3], rtol=1e-14)

kernel2 = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel2, NT)
@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)

# test inplace version
message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT, inplace=true)
@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)


# test symmetric matrix
plin = [Symmetric(randn(dim,dim)), randn(dim), randn(dim,m)] # B, β, σtil
kernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,plin)
kernel2 = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
message, solend  = MitosisStochasticDiffEq.backwardfilter(kernel, NT)
message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel2, NT)
@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)

plin = [Array(plin[1]), plin[2], plin[3]] # B, β, σtil
kernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,plin)
message2, solend2  = MitosisStochasticDiffEq.backwardfilter(kernel, NT)

@test isapprox(solend, solend2, rtol=1e-15)
@test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)
