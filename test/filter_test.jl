import MitosisStochasticDiffEq as MSDE
using Mitosis
using Test, Random
using LinearAlgebra

# set true/estimated model parameters
# p = [-0.1,0.2,0.9]

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

        P = P - dt*(B*P + P*B' - σ̃*σ̃'*I)
        ν = ν - dt*(B*ν .+ β)
        c = c - dt*tr(B)

        push!(ps, [ν, P, c])
    end
    ps, [ν, P, c]
end

@testset "backward filtering tests" begin

  # define SDE function
  f(u,p,t) = p[1]*u .+ p[2]
  g(u,p,t) = p[3]

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.02
  trange = tstart:dt:tend

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]
  kernel = MSDE.SDEKernel(f,g,trange,plin)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  message, solend = MSDE.backwardfilter(kernel, NT)

  kernel2 = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message2, solend2 = MSDE.backwardfilter(kernel2, NT)

  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), Array(message2.sol.u), rtol=1e-15)

  message2, solend2 = backwardfilter(NT, plin, message.ts)

  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), message2, rtol=1e-15)

  # multivariate tests
  dim = 5
  Random.seed!(123)
  logscale = randn()
  μ = randn(dim)
  Σ = randn(dim,dim)
  myvalues = [logscale, μ, Σ];
  NT = NamedTuple{mynames}(myvalues)

  kernel = MSDE.SDEKernel(f,g,trange,plin)
  message, solend = MSDE.backwardfilter(kernel, NT)
  message2, solend2 = backwardfilter(NT, plin, message.ts)

  @test isapprox(solend.x[1], solend2[1], rtol=1e-15)
  @test isapprox(solend.x[2], solend2[2], rtol=1e-15)
  @test isapprox(solend.x[3][1], solend2[3], rtol=1e-15)

  kernel2 = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message2, solend2 = MSDE.backwardfilter(kernel2, NT)
  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), Array(message2.sol.u), rtol=1e-15)

  # test inplace version
  message2, solend2 = MSDE.backwardfilter(kernel, NT, inplace=true)
  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), Array(message2.sol.u), rtol=1e-15)

  message2, solend2 = MSDE.backwardfilter(kernel2, NT, inplace=true)
  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), Array(message2.sol.u), rtol=1e-15)

  m = 3 # some number of Brownian processes
  plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil

  kernel = MSDE.SDEKernel(f,g,trange,plin)
  message, solend = MSDE.backwardfilter(kernel, NT)
  message2, solend2 = backwardfilter(NT, plin, message.ts)

  @test isapprox(solend.x[1], solend2[1], rtol=1e-14)
  @test isapprox(solend.x[2], solend2[2], rtol=1e-14)
  @test isapprox(solend.x[3][1], solend2[3], rtol=1e-14)

  kernel2 = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message2, solend2 = MSDE.backwardfilter(kernel2, NT)
  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), Array(message2.sol.u), rtol=1e-15)

  # test inplace version
  message2, solend2 = MSDE.backwardfilter(kernel, NT, inplace=true)
  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), Array(message2.sol.u), rtol=1e-15)

  # test symmetric matrix
  plin = [Symmetric(randn(dim,dim)), randn(dim), randn(dim,m)] # B, β, σtil
  kernel = MSDE.SDEKernel(f,g,trange,plin)
  kernel2 = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, solend  = MSDE.backwardfilter(kernel, NT)
  message2, solend2 = MSDE.backwardfilter(kernel2, NT)
  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), Array(message2.sol.u), rtol=1e-15)

  plin = [Array(plin[1]), plin[2], plin[3]] # B, β, σtil
  kernel = MSDE.SDEKernel(f,g,trange,plin)
  message2, solend2  = MSDE.backwardfilter(kernel, NT)

  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), Array(message2.sol.u), rtol=1e-15)
end

@testset "backward filtering timechange tests" begin
  # define SDE function
  f(u,p,t) = p[1]*u .+ p[2]
  g(u,p,t) = p[3]

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.02
  trange = tstart:dt:tend

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  kernel = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, solend = MSDE.backwardfilter(kernel, NT, apply_timechange=true)

  @test length(message.ts) == length(trange)
  @test message.ts != trange
  @test message.ts == MSDE.timechange(trange)

  message2, solend2 = backwardfilter(NT, plin, message.ts)

  @test isapprox(solend, solend2, rtol=1e-15)
  @test isapprox(Array(message.sol.u), message2, rtol=1e-15)
end


@testset "backward filtering adaptive tests" begin
  # define SDE function
  using OrdinaryDiffEq

  f(u,p,t) = p[1]*u .+ p[2]
  g(u,p,t) = p[3]

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.05 # more coarse grained dt
  trange = tstart:dt:tend

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  tildekernel = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, solend = MSDE.backwardfilter(tildekernel, NT, alg=Tsit5(), apply_timechange=true)

  # adaptive solution with large tolerances follows tstops
  @test length(message.ts) == length(trange)
  @test message.ts != trange
  @test message.ts == MSDE.timechange(trange)

  message2, solend2 = MSDE.backwardfilter(tildekernel, NT, alg=Tsit5(),
    abstol=1e-12, reltol=1e-12, apply_timechange=true)

  # adaptive solution with small tolerances steps to tstops but also takes additional substeps
  @test length(message.ts) != length(message2.ts)
  @test minimum(message.ts .∈ [message2.ts])
end
