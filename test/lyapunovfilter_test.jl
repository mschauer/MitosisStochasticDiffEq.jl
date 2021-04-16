using MitosisStochasticDiffEq
using Mitosis
using Test, Random
using LinearAlgebra
using OrdinaryDiffEq
@testset "backward Lyapunov filtering tests" begin
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
  message, solend = MitosisStochasticDiffEq.backwardfilter(kernel, NT, alg=OrdinaryDiffEq.Tsit5())

  # Lyapunov filter
  message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT,
    filter=MitosisStochasticDiffEq.LyapunovFilter())

  @test message.ts == message2.ts
  @test P == message2.sol.PT
  @test ν == message2.sol.νT
  @test c == message2.sol.cT

  @test message.soldis[1,:] ≈ message2.soldis[1,:] rtol=1e-10
  @test message.soldis[2,:] ≈ message2.soldis[2,:] rtol=1e-10
  @test message.soldis[3,:] ≈ message2.soldis[3,:] rtol=1e-10
  @test isapprox(solend, solend2, rtol=1e-10)

  dim = 5
  m = 3
  B, β, σ̃ = [Symmetric(randn(dim,dim)), randn(dim), randn(dim,m)] # B, β, σtil
  kernel = MitosisStochasticDiffEq.SDEKernel(
      Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, [B, β, σ̃]
  )

  Random.seed!(12345)
  c = randn()
  ν = randn(dim)
  P = randn(dim,dim)
  NT = NamedTuple{mynames}([c, ν, P])

  message, solend  = MitosisStochasticDiffEq.backwardfilter(kernel, NT)
  # Lyapunov filter
  message2, solend2 = MitosisStochasticDiffEq.backwardfilter(kernel, NT,
    filter=MitosisStochasticDiffEq.LyapunovFilter())

  @test message.soldis[1:dim,:] ≈ message2.soldis[1:dim,:] rtol=1e-2
  @test message.soldis[dim+1:dim+dim*dim,:] ≈ message2.soldis[dim+1:dim+dim*dim,:] rtol=1e-1
  @test message.soldis[end,:] ≈ message2.soldis[end,:] rtol=1e-10
  @test solend2 ≈ solend rtol=1e-1

end
