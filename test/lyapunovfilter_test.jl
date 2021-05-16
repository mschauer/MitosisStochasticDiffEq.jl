import MitosisStochasticDiffEq as MSDE
using Mitosis
using Test, Random
using LinearAlgebra
using OrdinaryDiffEq
Random.seed!(12345)
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
  c = randn()
  ν = randn()
  P = 10*rand()
  myvalues = [c, ν, P]
  NT = NamedTuple{mynames}(myvalues)

  # covariance filter
  kernel = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, solend = MSDE.backwardfilter(kernel, NT, alg=OrdinaryDiffEq.Tsit5())

  # Lyapunov filter
  message2, solend2 = MSDE.backwardfilter(kernel, NT,
    filter=MSDE.LyapunovFilter())

  @test message.ts == message2.ts
  @test P == message2.sol.PT
  @test ν == message2.sol.νT
  @test c == message2.sol.cT

  @test getindex.(message.soldis,1) ≈ getindex.(message2.soldis,1) rtol=1e-10
  @test getindex.(message.soldis,2) ≈ getindex.(message2.soldis,2) rtol=1e-10
  @test getindex.(message.soldis,3) ≈ getindex.(message2.soldis,3) rtol=1e-10
  @test isapprox(solend, solend2, rtol=1e-10)

  dim = 1
  m = 1
  B, β, σ̃ = fill(B, 1, 1), fill(β, 1), fill(σ̃, 1, 1)
  kernel = MSDE.SDEKernel(
      Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, [B, β, σ̃]
  )

  ν = fill(ν, 1)
  P = fill(P, 1, 1)
  NT = NamedTuple{mynames}([c, ν, P])

  message, solend  = MSDE.backwardfilter(kernel, NT, alg=OrdinaryDiffEq.Tsit5())
  # Lyapunov filter
  message2, solend2 = MSDE.backwardfilter(kernel, NT,
    filter=MSDE.LyapunovFilter())

  @test getindex.(message.soldis,1) ≈ getindex.(message2.soldis,1) rtol=1e-10
  @test getindex.(message.soldis,2) ≈ getindex.(message2.soldis,2) rtol=1e-10
  @test getindex.(message.soldis,3) ≈ getindex.(message2.soldis,3) rtol=1e-10
  @test isapprox(solend, solend2, rtol=1e-10)

  dim = 4
  m = 3
  B, β, σ̃ = [randn(dim,dim)-I, randn(dim), randn(dim,m)] # B, β, σtil
  kernel = MSDE.SDEKernel(
      Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, [B, β, σ̃]
  )

  c = randn()
  ν = randn(dim)
  P = randn(dim,dim)
  P = I + 0.1*P*P'
  NT = NamedTuple{mynames}([c, ν, P])

  message, solend  = MSDE.backwardfilter(kernel, NT,  alg=OrdinaryDiffEq.Tsit5())
  # Lyapunov filter
  message2, solend2 = MSDE.backwardfilter(kernel, NT,
    filter=MSDE.LyapunovFilter())

  @test getindex.(message.soldis,1) ≈ getindex.(message2.soldis,1) rtol=1e-9
  @test getindex.(message.soldis,2) ≈ getindex.(message2.soldis,2) rtol=1e-9
  @test getindex.(message.soldis,3) ≈ getindex.(message2.soldis,3) rtol=1e-9
  @test solend2 ≈ solend rtol=1e-9
end
