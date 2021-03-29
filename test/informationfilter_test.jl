using MitosisStochasticDiffEq
using Mitosis
using Test, Random
using LinearAlgebra

@testset "backward information filtering tests" begin
  # define SDE function
  f(u,p,t) = p[1]*u .+ p[2]
  g(u,p,t) = p[3]

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.001
  trange = tstart:dt:tend

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  c = 0.0
  ν = 0.0
  P = 10.0
  myvalues = [c, ν, P]
  NT = NamedTuple{mynames}(myvalues)

  # covariance filter
  kernel = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, solend = MitosisStochasticDiffEq.backwardfilter(kernel, NT)

  # information filter
  H = inv(P)
  F = H*ν
  myvalues2 = [c, F, H]
  NT2 = NamedTuple{mynames}(myvalues2)
  message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT2, filter=MitosisStochasticDiffEq.InformationFilter())

  @test !isapprox(solend, solend2, rtol=1e-1)
  invH = inv(solend2[2])
  invHF = invH*solend2[1]
  @test isapprox(solend[1], invHF, rtol=1e-3)
  @test isapprox(solend[2], invH, rtol=1e-3)
  @test isapprox(solend[3], solend2[3], rtol=1e-10)


  # multivariate tests
  dim = 5
  Random.seed!(123)
  c = randn()
  ν = randn(dim)
  P = randn(dim,dim)
  myvalues = [c, ν, P]
  NT = NamedTuple{mynames}(myvalues)
  H = inv(P)
  F = H*ν
  myvalues2 = [c, F, H]
  NT2 = NamedTuple{mynames}(myvalues2)

  # covariance filter
  kernel = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, solend = MitosisStochasticDiffEq.backwardfilter(kernel, NT)

  # information filter
  message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT2,
    filter=MitosisStochasticDiffEq.InformationFilter())

  @test !isapprox(solend, solend2, rtol=1e-1)
  invH = inv(solend2.x[2])
  invHF = invH*solend2.x[1]
  @test isapprox(solend.x[1], invHF, rtol=1e-3)
  @test isapprox(solend.x[2], invH, rtol=1e-2)
  @test isapprox(solend.x[3], solend2.x[3], rtol=1e-10)

  # test inplace version
  message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT2,
    filter=MitosisStochasticDiffEq.InformationFilter(), inplace=true)

  @test !isapprox(solend, solend2, rtol=1e-1)
  invH = inv(solend2.x[2])
  invHF = invH*solend2.x[1]
  @test isapprox(solend.x[1], invHF, rtol=1e-3)
  @test isapprox(solend.x[2], invH, rtol=1e-2)
  @test isapprox(solend.x[3], solend2.x[3], rtol=1e-10)
end
#
# # define SDE function
# f(u,p,t) = p[1]*u .+ p[2]
# g(u,p,t) = p[3]
#
# # time span
# tstart = 0.0
# tend = 1.0
# dt = 0.001
# trange = tstart:dt:tend
#
# m = 3 # some number of Brownian processes
# Random.seed!(123)
# plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil
#
# # multivariate tests
# dim = 5
# Random.seed!(123)
# c = randn()
# ν = randn(dim)
# P = randn(dim,dim)
# myvalues = [c, ν, P]
# NT = NamedTuple{mynames}(myvalues)
# H = inv(P)
# F = H*ν
# myvalues2 = [c, F, H]
# NT2 = NamedTuple{mynames}(myvalues2)
#
# # covariance filter
# kernel = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
# message, solend = MitosisStochasticDiffEq.backwardfilter(kernel, NT)
#
# # information filter
# message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT2,
#   filter=MitosisStochasticDiffEq.InformationFilter())
#
#
# @test !isapprox(solend, solend2, rtol=1e-1)
# invH = inv(solend2.x[2])
# invHF = invH*solend2.x[1]
# @test isapprox(solend.x[1], invHF, rtol=1e-3)
# @test isapprox(solend.x[2], invH, rtol=1e-2)
# @test isapprox(solend.x[3], solend2.x[3], rtol=1e-10)

# test inplace version
# message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT2, inplace=true,
#   filter=MitosisStochasticDiffEq.InformationFilter())
#
# @test !isapprox(solend, solend2, rtol=1e-1)
# invH = inv(solend2.x[2])
# invHF = invH*solend2.x[1]
# @test isapprox(solend.x[1], invHF, rtol=1e-3)
# @test isapprox(solend.x[2], invH, rtol=1e-2)
# @test isapprox(solend.x[3], solend2.x[3], rtol=1e-10)
