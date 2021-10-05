import MitosisStochasticDiffEq as MSDE
using StochasticDiffEq
using Mitosis
using LinearAlgebra
using SparseArrays
using DiffEqNoiseProcess
using Test, Random


"""
forwardsample(f, g, p, s, W, x) using the Euler-Maruyama scheme
on a time-grid s with associated noise values W
"""
function forwardsample(f, g, p, s, Ws, x)
    xs = typeof(x)[]
    for i in eachindex(s)[1:end-1]
        dt = s[i+1] - s[i]
        push!(xs, x)
        x = x + f(x, p, s[i])*dt + g(x, p, s[i])*(Ws[i+1]-Ws[i])
    end
    push!(xs, x)

    return xs
end


@testset "sampling tests" begin
  # define SDE function
  f(u,p,t) = p[1]*u + p[2] - 1.5*sin.(u*2pi)
  g(u,p,t) = p[3] .- 0.2*(1 .-sin.(u))

  # set estimate of model parameters or true model parameters
  p = [-0.1,0.2,0.9]

  # time range
  tstart = 0.0
  tend = 1.0
  dt = 0.02
  trange = tstart:dt:tend

  # intial condition
  u0 = 1.1

  kernel = MSDE.SDEKernel(f,g,trange,p)
  # sample using MSDE and EM default
  sol, solend = MSDE.sample(kernel, u0)

  kernel = MSDE.SDEKernel(f,g,collect(trange),p)
  uend, (ts, u, noise) = MSDE.sample(kernel, u0, save_noise=true)

  @test isapprox(u, forwardsample(f,g,p,ts,noise,u0), atol=1e-12)
end


@testset "multivariate sampling tests" begin
  seed = 12345
  Random.seed!(seed)
  d = 2
  u0 = randn(2)
  θlin = (randn(d,d), randn(d), Diagonal([0.1, 0.1]))

  Σ(θ) = Diagonal(θ[2]) # just to generate the noise_rate_prototype

  f(u,p,t) = p[1]*u + p[2]
  f!(du,u,p,t) = (du .= p[1]*u + p[2])
  gvec(u,p,t) = diag(p[3])
  function gvec!(du,u,p,t)
    du[1] = p[3][1,1]
    du[2] = p[3][2,2]
  end
  g(u,p,t) = p[3]
  # Make `g` write the sparse matrix values
  function g!(du,u,p,t)
    du[1,1] = p[3][1,1]
    du[2,2] = p[3][2,2]
  end

  function gstepvec!(dx, _, u, p, t, dw, _)
    dx .+= diag(p[3]).*dw
  end

  function gstep!(dx, _, u, p, t, dw, _)
    dx .+= p[3]*dw
  end

  # Define a sparse matrix by making a dense matrix and setting some values as not zero
  A = zeros(2,2)
  A[1,1] = 1
  A[2,2] = 1
  A = sparse(A)

  # time range
  tstart = 0.0
  tend = 1.0
  dt = 0.02
  trange = tstart:dt:tend

  # define kernels
  k1 = MSDE.SDEKernel(f,gvec,trange,θlin)
  k2 = MSDE.SDEKernel(f,g,trange,θlin,Σ(θlin))
  k3 = MSDE.SDEKernel(f!,g!,trange,θlin,A)
  k4 = MSDE.SDEKernel(Mitosis.AffineMap(θlin[1], θlin[2]), Mitosis.ConstantMap(θlin[3]), trange, θlin, Σ(θlin))
  k5 = MSDE.SDEKernel!(f!,gvec!,gstepvec!,trange,θlin; ws = copy(u0))
  k6 = MSDE.SDEKernel!(f!,g!,gstep!,trange,θlin,A; ws = copy(A))

  @testset "StochasticDiffEq EM() solver" begin
    uend1, (ts1, u1, noise1) = MSDE.sample(k1, u0, EM(false), save_noise=true)
    NG = NoiseGrid(ts1, noise1)
    Z = pCN(NG, 1.0)
    uend2, (ts2, u2, noise2) = MSDE.sample(k2, u0, EM(false), Z, save_noise=true)
    Z = pCN(NG, 1.0)
    uend3, (ts3, u3, noise3) = MSDE.sample(k3, u0, EM(false), Z)
    Z = pCN(NG, 1.0)
    uend4, (ts4, u4, noise4) = MSDE.sample(k4, u0, EM(false), Z)
    Z = pCN(NG, 1.0)
    uend5, (ts5, u5, noise5) = MSDE.sample(k5, u0, EM(false), Z)
    Z = pCN(NG, 1.0)
    uend6, (ts6, u6, noise6) = MSDE.sample(k6, u0, EM(false), Z)

    #@show solend1
    @test isapprox(u1, u2, atol=1e-12)
    @test isapprox(uend1, uend2, atol=1e-12)
    @test isapprox(u1, u3, atol=1e-12)
    @test isapprox(uend1, uend3, atol=1e-12)
    @test isapprox(u1, u4, atol=1e-12)
    @test isapprox(uend1, uend4, atol=1e-12)
    @test isapprox(u1, u5, atol=1e-12)
    @test isapprox(uend1, uend5, atol=1e-12)
    @test isapprox(u1, u6, atol=1e-12)
    @test isapprox(uend1, uend6, atol=1e-12)
  end

  @testset "internal solver" begin
    @testset "without passing a noise" begin
      Random.seed!(seed)
      uend1, (ts1, u1, noise1) = MSDE.sample(k1, u0, MSDE.EulerMaruyama!(), save=true)
      Random.seed!(seed)
      uend2, (ts2, u2, noise2) = MSDE.sample(k2, u0, MSDE.EulerMaruyama!(), save=true)
      Random.seed!(seed)
      # inplace must be written out manually
      @test_broken uend3, (ts3, u3, noise3) = MSDE.sample(k3, u0, MSDE.EulerMaruyama!(), save=true)
      Random.seed!(seed)
      uend4, (ts4, u4, noise4) = MSDE.sample(k4, u0, MSDE.EulerMaruyama!(), save=true)
      Random.seed!(seed)
      uend5, (ts5, u5, noise5) = MSDE.sample(k5, u0, MSDE.EulerMaruyama!(), save=true)
      Random.seed!(seed)
      uend6, (ts6, u6, noise6) = MSDE.sample(k6, u0, MSDE.EulerMaruyama!(), save=true)

      @test length(ts1) == length(trange)
      @test ts1[end] == trange[end]
      @test uend1 == uend2
      @test_broken uend1 == uend3
      @test uend1 == uend4
      @test uend1 == uend5
      @test uend1 == uend6

      Random.seed!(seed)
      uend7, (ts7, u7, noise7) = MSDE.sample(k1, u0, MSDE.EulerMaruyama!(), save=false)
      @test uend1 == uend7
      @test u7 === nothing
    end

    @testset "passing a noise grid" begin
      # pass noise process and compare with EM()
      Ws = cumsum([[zero(u0)];[sqrt(trange[i+1]-ti)*randn(size(u0))
              for (i,ti) in enumerate(trange[1:end-1])]])
      NG = NoiseGrid(trange,Ws)

      uendEM, (tsEM, uEM, noiseEM) = MSDE.sample(k1, u0, EM(false), NG)
      uend1, (ts1, u1, noise1) = MSDE.sample(k1, u0, MSDE.EulerMaruyama!(), NG)
      uend2, (ts2, u2, noise2) = MSDE.sample(k2, u0, MSDE.EulerMaruyama!(), NG)
      @test_broken uend3, (ts3, u3, noise3) = MSDE.sample(k3, u0, MSDE.EulerMaruyama!(), NG)
      uend4, (ts4, u4, noise4) = MSDE.sample(k4, u0, MSDE.EulerMaruyama!(), NG)
      uend5, (ts5, u5, noise5) = MSDE.sample(k5, u0, MSDE.EulerMaruyama!(), NG)
      uend6, (ts6, u6, noise6) = MSDE.sample(k6, u0, MSDE.EulerMaruyama!(), NG)

      @test u1 ≈ uEM rtol=1e-12
      @test uendEM ≈ uend1 rtol=1e-12
      @test uendEM ≈ uend2 rtol=1e-12
      @test_broken uendEM ≈ uend3 rtol=1e-12
      @test uendEM ≈ uend4 rtol=1e-12
      @test uendEM ≈ uend5 rtol=1e-12
      @test uendEM ≈ uend6 rtol=1e-12
    end

    @testset "passing the noise values" begin
      # pass noise process and compare with EM()
      Ws = cumsum([[zero(u0)];[sqrt(trange[i+1]-ti)*randn(size(u0))
              for (i,ti) in enumerate(trange[1:end-1])]])
      NG = NoiseGrid(trange,Ws)

      uendEM, (tsEM, uEM, noiseEM)  = MSDE.sample(k1, u0, EM(false), NG)
      uend1, (ts1, u1, noise1) = MSDE.sample(k1, u0, MSDE.EulerMaruyama!(), Ws)
      uend2, (ts2, u2, noise2) = MSDE.sample(k2, u0, MSDE.EulerMaruyama!(), Ws)
      @test_broken uend3, (ts3, u3, noise3) = MSDE.sample(k3, u0, MSDE.EulerMaruyama!(), Ws)
      uend4, (ts4, u4, noise4) = MSDE.sample(k4, u0, MSDE.EulerMaruyama!(), Ws)
      uend5, (ts5, u5, noise5) = MSDE.sample(k5, u0, MSDE.EulerMaruyama!(), Ws)
      uend6, (ts6, u6, noise6) = MSDE.sample(k6, u0, MSDE.EulerMaruyama!(), Ws)

      @test u1 ≈ uEM rtol=1e-12
      @test uendEM ≈ uend1 rtol=1e-12
      @test uendEM ≈ uend2 rtol=1e-12
      @test_broken uendEM ≈ uend3 rtol=1e-12
      @test uendEM ≈ uend4 rtol=1e-12
      @test uendEM ≈ uend5 rtol=1e-12
      @test uendEM ≈ uend6 rtol=1e-12
    end

    @testset "custom P" begin
      # checks that defining and passing P manually works

      struct customP{θType}
        θ::θType
      end

      function MSDE.tangent!(du, u, dz, P::customP)
        du[3] .= (P.θ[1]*u[3]+P.θ[2])*dz[2] + P.θ[3]*dz[3]

        (dz[1], dz[2], du[3])
      end

      function MSDE.exponential_map!(u, du, P::customP)
        x = u[3]
        @. x += du[3]
        (u[1] + du[1], u[2] + du[2], x)
      end

      # pass noise process and compare with EM()
      Ws = cumsum([[zero(u0)];[sqrt(trange[i+1]-ti)*randn(size(u0))
              for (i,ti) in enumerate(trange[1:end-1])]])
      NG = NoiseGrid(trange,Ws)

      uendEM, (tsEM, uEM, noiseEM) = MSDE.sample(k1, u0, EM(false), NG)
      uend1, (ts1, u1, noise1) = MSDE.sample(k1, u0, MSDE.EulerMaruyama!(), Ws, P=customP(θlin))
      uend2, (ts2, u2, noise2) = MSDE.sample(k2, u0, MSDE.EulerMaruyama!(), Ws, P=customP(θlin))
      uend3, (ts3, u3, noise3) = MSDE.sample(k3, u0, MSDE.EulerMaruyama!(), Ws, P=customP(θlin))
      uend4, (ts4, u4, noise4)  = MSDE.sample(k4, u0, MSDE.EulerMaruyama!(), Ws)
      uend5, (ts5, u5, noise5) = MSDE.sample(k5, u0, MSDE.EulerMaruyama!(), Ws)
      uend6, (ts6, u6, noise6) = MSDE.sample(k6, u0, MSDE.EulerMaruyama!(), Ws)

      @test u1 ≈ uEM rtol=1e-12
      @test uendEM ≈ uend1 rtol=1e-12
      @test uendEM ≈ uend2 rtol=1e-12
      @test uendEM ≈ uend3 rtol=1e-12
      @test uendEM ≈ uend4 rtol=1e-12
      @test uendEM ≈ uend5 rtol=1e-12
      @test uendEM ≈ uend6 rtol=1e-12
    end

  end
end
