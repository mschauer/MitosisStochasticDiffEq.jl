import MitosisStochasticDiffEq as MSDE
using Test, Random
using LinearAlgebra

@testset "Internal solver tests" begin
  @testset "EulerMaruyama == EM test" begin
    @testset "n=1, m=1 (univariate, scalar) tests" begin
      # define SDE function
      f(u,p,t) = p[1]*u + p[2]
      g(u,p,t) = p[3]*u

      # time span
      tstart = 0.0
      tend = 1.0
      dt = 0.02
      trange = tstart:dt:tend

      B, β, σ̃ = -0.1, 0.2, 1.3
      p = [B, β, σ̃]

      u0 = rand(1)

      kernel = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, p)
      # pass noise process and compare with EM()
      Ws = cumsum([[zero(u0)];[sqrt(trange[i+1]-ti)*randn(size(u0))
              for (i,ti) in enumerate(trange[1:end-1])]])
      NG = NoiseGrid(trange,Ws)

      solEM, solendEM = MSDE.sample(kernel, u0, EM(false), NG)
      sol1, solend1 = MSDE.sample(kernel, u0, MSDE.EulerMaruyama!(), Ws)

      @test getindex.(sol1,3) ≈ solEM.u rtol=1e-12
      @test solendEM ≈ solend1[3] rtol=1e-12
    end
  end

  @testset "strong convergence tests" begin
    @testset "n=1, m=1 (univariate, scalar) tests" begin
      #..
    end
  end

  @testset "weak convergence tests" begin
    @testset "n=1, m=1 (univariate, scalar) tests" begin
      # ..
    end
  end


end
