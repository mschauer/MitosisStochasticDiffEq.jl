import MitosisStochasticDiffEq as MSDE
using Mitosis
using DiffEqNoiseProcess
using Test, Random
using LinearAlgebra
using Statistics
using StochasticDiffEq
using SparseArrays
using StaticArrays

# Test outer function
@testset "outer function tests" begin
  exA = rand(10,10)
  @test minimum(MSDE.outer_(exA) .!= 0)
  exB = Diagonal(exA)
  @test sum(MSDE.outer_(exB) .!= 0) == 10
end

"""
    forwardguiding(M, s, x, ps, Z) -> xs, ll
Forward sample a guided trajectory `xs` starting in `x` and compute it's
log-likelihood `ll` with innovations `Z = randn(length(s))`.
"""
function forwardguiding(plin, pest, s, (x, ll), ps, Z=randn(length(s)), noisetype=:scalar)
    # linear approximation of b and constant approximation of σ
    # with parameters B, β, and σ̃
    flinear(u,p,t) = p[1]*u .+ p[2]
    σlinear(u,p,t) = p[3]

    function llstep(x, r, t, P, noisetype)
      tmp = MSDE.outer_(g(x,pest,t)) - MSDE.outer_(σlinear(x,plin,t))
      dll = dot(f(x,pest,t) - flinear(x,plin,t), r) -0.5*tr(tmp*(inv(P) - MSDE.outer_(r)))
    end

    xs = typeof(x)[]
    d = length(x)
    for i in eachindex(s)[1:end-1]
        dt = s[i+1] - s[i]
        t = s[i]
        push!(xs, x)
        # ν = @view ps[:,i][1:d]
        # P = reshape(@view(ps[:,i][d+1:d+d*d]), d, d)
        # r = inv(P)*(ν .- x)
        ν = ps[i][1]
        P = Array(ps[i][2])
        r = inv(P)*(ν .- x)


        ll += llstep(x, r, t, P, noisetype)*dt # accumulate log-likelihood

        if noisetype == :scalar
            noise = g(x,pest,t)*Z[i][] #sqrt(dt)*Z[i]
        elseif noisetype ==:diag
            noise = g(x,pest,t).*Z[i]
        elseif noisetype ==:nondiag
            noise = g(x,pest,t)*Z[:,i]
        else
            error("noisetype not understood.")
        end
        if x isa Number
           tmp = (MSDE.outer_(g(x,pest,t))*r*dt)[1]
       else
           tmp = MSDE.outer_(g(x,pest,t))*r*dt
       end
        x = x + f(x,pest,t)*dt + tmp + noise # evolution guided by observations

    end
    push!(xs, x)
    xs, ll
end

# define SDE function
f(u,p,t) = @. p[1]*u + p[2] - 1.5*sin(u*2pi)
g(u,p,t) = p[3] .- 0.2*(1 .-sin.(u))

@testset "IIP Guiding tests" begin
  # set true model parameters
  p = [-0.1,0.2,0.9]

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]
  pest = [-0.4, 0.5, 1.4] # initial guess of parameter to be estimated

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.001
  trange = tstart:dt:tend

  # intial condition
  u0 = 1.1

  # forward kernel
  sdekernel = MSDE.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)

  x0 = randn()
  ll0 = randn()

  (uend1, ll), (ts1, u1, noise1) = MSDE.forwardguiding(sdekernel, message, (x0, ll0); save_noise=true)


  dWs = (noise1[2:end]-noise1[1:end-1])
  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0),ps,dWs)

  @test isapprox(getindex.(u1,1), solfw2, rtol=1e-12)
  @test isapprox(ll, ll2, rtol=1e-12)

  # multivariate tests with scalar random process
  dim = 7
  Random.seed!(1234)
  logscale = randn()
  μ = randn(dim)
  Σ = randn(dim,dim)
  myvalues = [logscale, μ, Σ];
  NT = NamedTuple{mynames}(myvalues)

  m = 1
  plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil

  # set scalar random process
  t = tstart:dt:tend
  W = sqrt(dt)*randn(length(t))
  W1 = cumsum([zero(dt); W[1:end-1]])
  NG = NoiseGrid(t,W1)

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)

  x0 = randn(dim)
  ll0 = randn()
  (uend1, ll), (ts1, u1, noise1) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), EM(false), NG)

  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0), ps, W)

  @test isapprox(u1, solfw2, rtol=1e-12)
  @test isapprox(ll, ll2, rtol=1e-12)

  # multivariate tests with diagonal noise random process
  dim = 2
  Random.seed!(12345)
  logscale = randn()
  μ = randn(dim)
  Σ = randn(dim,dim)
  myvalues = [logscale, μ, Σ];
  NT = NamedTuple{mynames}(myvalues)

  m = 2
  plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)

  x0 = randn(dim)
  ll0 = randn()

  (uend1, ll), (ts1, u1, noise1) = MSDE.forwardguiding(sdekernel, message, (x0, ll0); save_noise=true)

  dWs = noise1[2:end]-noise1[1:end-1]

  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0),ps,dWs,:diag)

  @test isapprox(u1, solfw2, rtol=1e-12)
  @test isapprox(ll, ll2, rtol=1e-12)
end

@testset "OOP Guiding tests" begin

  # set true model parameters
  p = [-0.1,0.2,0.9]

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]
  pest = [-0.4, 0.5, 1.4] # initial guess of parameter to be estimated

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.001
  trange = tstart:dt:tend

  # intial condition
  u0 = 1.1

  # forward kernel
  sdekernel = MSDE.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)

  x0 = randn()
  ll0 = randn()

  (uend1, ll), (ts1, u1, noise1) = MSDE.forwardguiding(sdekernel, message, (x0, ll0); save_noise=true, inplace=false)

  dWs = (noise1[2:end]-noise1[1:end-1])
  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0),ps,dWs)

  @test isapprox(getindex.(u1,1), solfw2, rtol=1e-12)
  @test isapprox(ll, ll2, rtol=1e-12)

  # multivariate tests with scalar random process
  dim = 7
  Random.seed!(1234)
  logscale = randn()
  μ = randn(dim)
  Σ = randn(dim,dim)
  myvalues = [logscale, μ, Σ];
  NT = NamedTuple{mynames}(myvalues)

  m = 1
  plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil

  # set scalar random process
  t = tstart:dt:tend
  W = sqrt(dt)*randn(length(t))
  W1 = cumsum([zero(dt); W[1:end-1]])
  NG = NoiseGrid(t,W1)

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)

  x0 = randn(dim)
  ll0 = randn()
  (uend1, ll), (ts1, u1, noise1)  = MSDE.forwardguiding(sdekernel, message, (x0, ll0), EM(false), NG, inplace=false)

  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0), ps, W)

  @test isapprox(u1, solfw2, rtol=1e-12)
  @test isapprox(ll, ll2, rtol=1e-12)

  # multivariate tests with diagonal noise random process
  dim = 2
  Random.seed!(12345)
  logscale = randn()
  μ = randn(dim)
  Σ = randn(dim,dim)
  myvalues = [logscale, μ, Σ];
  NT = NamedTuple{mynames}(myvalues)

  m = 2
  plin = [randn(dim,dim), randn(dim), randn(dim,m)] # B, β, σtil

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)

  x0 = randn(dim)
  ll0 = randn()

  (uend1, ll), (ts1, u1, noise1)  = MSDE.forwardguiding(sdekernel, message, (x0, ll0); save_noise=true, inplace=false)

  Ws = Array(noise1)
  dWs = Ws[2:end]-Ws[1:end-1]

  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0),ps,dWs,:diag)

  @test isapprox(u1, solfw2, rtol=1e-12)
  @test isapprox(ll, ll2, rtol=1e-12)
end

@testset "Adaptive Guiding tests" begin
  Random.seed!(12345)
  using StochasticDiffEq, DiffEqNoiseProcess
  # set true model parameters
  p = [-0.1,0.2,0.9]

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]
  pest = [-0.4, 0.5, 1.4] # initial guess of parameter to be estimated

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.001
  trange = tstart:dt:tend

  # intial condition
  u0 = 1.1

  # forward kernel
  sdekernel = MSDE.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)

  # define NoiseGrid
  brownian_values = cumsum([[zeros(2)];[sqrt(dt)*randn(2) for i in 1:length(trange)-1]])
  brownian_values2 = cumsum([[zeros(2)];[sqrt(dt)*randn(2) for i in 1:length(trange)-1]])
  W = NoiseGrid(collect(trange),brownian_values,brownian_values2)

  x0 = randn()
  ll0 = randn()

  (uend1, ll), (ts1, u1, noise1) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), LambaEM(),
            W; dt=dt, isadaptive=false)
  (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), LambaEM(),
            W; dt=dt, isadaptive=true)
  (uend3, ll3), (ts3, u3, noise3) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), SOSRI(),
            W; dt=dt, isadaptive=true)

  @test isapprox(ll, ll2, rtol=1e-1)
  @test isapprox(ll, ll3, rtol=1e-1)
  @test isapprox(ll2, ll2, rtol=1e-1)

  @test length(ts1) == length(trange)
  @test length(ts2) < length(ts1)
  @test length(ts3) < length(ts1)

  @show length(ts1), length(ts2), length(ts3)
end

@testset "timechange Guiding tests" begin
  Random.seed!(12345)
  # set true model parameters
  p = [-0.1,0.2,0.9]

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]
  pest = [-0.4, 0.5, 1.4] # initial guess of parameter to be estimated

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.001
  trange = tstart:dt:tend

  # intial condition
  u0 = 1.1

  # forward kernel
  sdekernel = MSDE.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT, apply_timechange=true)

  x0 = randn()
  ll0 = randn()

  (uend, ll), (ts, u, noise) = MSDE.forwardguiding(sdekernel, message, (x0, ll0);
    isadaptive=false)

  @test isapprox(ts, message.ts, rtol=1e-10)
  @test isapprox(ts, MSDE.timechange(trange), rtol=1e-10)
  @test length(ts) == length(trange)
end

@testset "Reuse of noise values tests" begin
  Random.seed!(12345)
  using StochasticDiffEq, DiffEqNoiseProcess
  # set true model parameters
  p = [-0.1,0.2,0.9]

  # set of linear parameters Eq.~(2.2)
  B, β, σ̃ = -0.1, 0.2, 1.3
  plin = [B, β, σ̃]
  pest = [-0.4, 0.5, 1.4] # initial guess of parameter to be estimated

  # time span
  tstart = 0.0
  tend = 1.0
  dt = 0.001
  trange = tstart:dt:tend

  # intial condition
  u0 = 1.1

  # forward kernel
  sdekernel = MSDE.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)


  # define NoiseGrid
  brownian_values = cumsum([[zeros(2)];[sqrt(dt)*randn(2) for i in 1:length(trange)-1]])
  W = NoiseGrid(collect(trange),brownian_values)

  x0 = randn()
  ll0 = randn()

  # test two subsequent evaluations with same Brownian motion given by NoiseGrid
  (uend1, ll1), (ts1, u1, noise1) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), EM(), W; dt=dt)
  (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), EM(), W; dt=dt)

  @test isapprox(ll1, ll2, rtol=1e-14)
  @test isapprox(u1, u2, rtol=1e-14)
  @test isapprox(noise1, noise2, rtol=1e-14)
  @test isapprox(noise1, W.W, rtol=1e-14)

  # test pCN with \rho = 1
  (uend1, ll1), (ts1, u1, noise1) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), EM(); dt=dt, save_noise=true)
  Z = pCN(ts1, noise1, 1.0)

  (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), EM(), Z; dt=dt)

  @test isapprox(ll1, ll2, rtol=1e-14)
  @test isapprox(u1, u2, rtol=1e-14)
  @test isapprox(noise1, noise1, rtol=1e-14)
  @test isapprox(noise1, Z.W, rtol=1e-14)

  # test pCN with ρ = 0.2 (decrease dt for test)
  ρ = 0.2
  dt = 0.0001
  trange = tstart:dt:tend
  # backward kernel
  kerneltilde = MSDE.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MSDE.backwardfilter(kerneltilde, NT)
  (uend1, ll1), (ts1, u1, noise1) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), EM(); dt=dt, save_noise=true)

  Z = pCN(ts1, noise1, ρ)

  (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(sdekernel, message, (x0, ll0), EM(), Z; dt=dt)

  computedW(W, indx, dt) = (W[indx+1][1]-W[indx][1])/sqrt(dt) # likelihood part can be ignored
  dWnew = []
  dWold = []
  for i in 1:length(ts1[1:end-1])
    push!(dWnew,computedW(noise1,i,dt))
    push!(dWold,computedW(noise2,i,dt))
  end
  @show cor(dWnew,dWold)
  @test ≈(cor(dWnew,dWold),ρ,rtol=1e-1)
end

@testset "flag-constant and matrix-valued diffusivity tests" begin

  Random.seed!(12345)

  ## define model (two traits, as in phylo)
  d = 2
  bθ = [.5, 0.9]
  σ0 = [1.25, 1.35]
  θ0 = (bθ, σ0)

  θ = ([-1.0, -0.2], [0.1, 0.1])
  M = -0.05I + [-1.0 1.0; 1.0 -1.0]

  B(θ) = Diagonal(θ[1]) * M
  Σ(θ) = Diagonal(θ[2])

  f(u,θ,t) = tanh.(Diagonal(θ[1]) * M * u)
  g(u,θ,t) = θ[2]

  u0 = zeros(2)

  θlin = (B(θ), zeros(d), Σ(θ))

  # time span
  tstart = 0.0
  tend = 0.1
  dt = 0.001
  trange = tstart:dt:tend

  # forward kernels
  κ1 = MSDE.SDEKernel(f, g, trange, θ0)
  κ2 = MSDE.SDEKernel(f, g, trange, θ0, nothing, true)
  # backward kernel
  κ̃ = MSDE.SDEKernel(Mitosis.AffineMap(θlin[1], θlin[2]), Mitosis.ConstantMap(θlin[3]), trange, θlin)

  # forward sample
  xT, (ts, x, noise) = MSDE.sample(κ1, u0; save_noise=true)
  Z = pCN(ts, noise, 1)
  xT2, (_, x2, _) = MSDE.sample(κ2, u0, EM(false), Z)

  @test x ≈ x2
  @test xT ≈ xT2

  # backward filter
  logscale = randn()
  ν = randn(d)
  P = randn(d,d)
  gaussian = WGaussian{(:μ,:Σ,:c)}(ν, P, logscale)

  message, backward = MSDE.backwardfilter(κ̃, gaussian)

  # forward guiding
  κg1 = MSDE.SDEKernel(f, g, trange, θ)
  κg2 = MSDE.SDEKernel(f, g, trange, θ, nothing, true)

  x0 = randn(d)
  ll0 = randn()

  (uend1, ll1), (ts1, u1, noise1) = MSDE.forwardguiding(κg1, message, (x0, ll0); save_noise=true)
  Z = pCN(ts1, noise1, 1.0)
  (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(κg2, message, (x0, ll0), EM(false), Z)

  @test ll1 ≈ ll2
  @test isapprox(u1, u2, rtol=1e-14)
  @test isapprox(noise1, noise2, rtol=1e-14)


  # check guiding with matrix-valued diffusion
  gmat(u,θ,t) = Diagonal(θ[2])
  kg3 = MSDE.SDEKernel(f, gmat, trange, θ, Σ(θ), true)

  # inplace=true
  Z = pCN(ts1, noise1, 1.0)
  (uend3, ll3), (ts3, u3, noise3) = MSDE.forwardguiding(kg3, message, (x0, ll0), EM(false), Z)
  @test ll1 ≈ ll3
  @test isapprox(u1, u3, rtol=1e-14)

  # inplace=false
  Z = pCN(ts1, noise1, 1.0)
  (uend3, ll3), (ts3, u3, noise3) = MSDE.forwardguiding(kg3, message, (x0, ll0), EM(false), Z, inplace=false)
  @test ll1 ≈ ll3
  @test isapprox(u1, u3, rtol=1e-14)
end

@testset "multivariate forward guiding tests" begin
  seed = 12345
  Random.seed!(seed)
  d = 2
  u0 = randn(2)
  θlin = (randn(d,d), randn(d), Diagonal([0.1, 0.1]))

  Σ(θ) = Diagonal(θ[2]) # just to generate the noise_rate_prototype

  f(u,p,t) = p[1]*u + p[2]
  gvec(u,p,t) = diag(p[3])
  g(u,p,t) = p[3]
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
  k3 = MSDE.SDEKernel(Mitosis.AffineMap(θlin[1], θlin[2]), Mitosis.ConstantMap(θlin[3]), trange, θlin, Σ(θlin))


  solend, (_, sol, _) = MSDE.sample(k1, u0, EM(false), save_noise=true)
  v = solend
  c = randn()
  Pmat = randn(d,d)

  message1, backward1 = MSDE.backwardfilter(k1, WGaussian{(:μ, :Σ, :c)}(v, Pmat, c))
  message2, backward2 = MSDE.backwardfilter(k2, WGaussian{(:μ, :Σ, :c)}(v, Pmat, c))
  message3, backward3 = MSDE.backwardfilter(k2, WGaussian{(:μ, :Σ, :c)}(v, Pmat, c))
  message4, backward4 = MSDE.backwardfilter(k3, WGaussian{(:μ, :Σ, :c)}(SVector{length(v)}(v), SMatrix{2,2,eltype(Pmat)}(Pmat), c))

  @test message1.soldis == message2.soldis
  @test message1.soldis == message3.soldis
  @test getindex.(message1.soldis,1) ≈ getindex.(message4.soldis,1) rtol=1e-14
  @test getindex.(message1.soldis,2) ≈ getindex.(message4.soldis,2) rtol=1e-14
  @test getindex.(message1.soldis,3) ≈ getindex.(message4.soldis,3) rtol=1e-14

  ll0 = randn()

  @testset "StochasticDiffEq EM() solver" begin
    (uend1, ll1), (ts1, u1, noise1) = MSDE.forwardguiding(k1, message1, (u0, ll0), EM(false); save_noise=true)
    Z = pCN(ts1, noise1, 1.0)
    (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(k2, message2, (u0, ll0), EM(false), Z; save_noise=true)
    Z = pCN(ts1, noise1, 1.0)
    (uend3, ll3), (ts3, u3, noise3) = MSDE.forwardguiding(k3, message3, (u0, ll0), EM(false), Z; save_noise=true)
    Z = pCN(ts1, noise1, 1.0)
    (uend4, ll4), (ts4, u4, noise4) = MSDE.forwardguiding(k3, message4, (u0, ll0), EM(false), Z; save_noise=true)

    @test u1 ≈ u2 rtol=1e-14
    @test u1 ≈ u3 rtol=1e-14
    @test u1 ≈ u4 rtol=1e-14
    @test ll1 == ll2
    @test ll1 == ll3
    @test ll1 == ll4
  end

  @testset "internal solver" begin
    @testset "without passing a noise" begin
      Random.seed!(seed)
      (uend1, ll1), (ts1, u1, noise1) = MSDE.forwardguiding(k1, message1, (u0, ll0), MSDE.EulerMaruyama!(), inplace=false)
      Random.seed!(seed)
      (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(k2, message2, (u0, ll0), MSDE.EulerMaruyama!(), inplace=false)
      Random.seed!(seed)
      (uend3, ll3), (ts3, u3, noise3) = MSDE.forwardguiding(k3, message3, (u0, ll0), MSDE.EulerMaruyama!(), inplace=false)
      Random.seed!(seed)
      (uend4, ll4), (ts4, u4, noise4) = MSDE.forwardguiding(k3, message4, (u0, ll0), MSDE.EulerMaruyama!(), inplace=false)
      @test isapprox(uend1,uend2,rtol=1e-14)
      @test isapprox(uend1,uend3,rtol=1e-14)
      @test isapprox(uend1,uend4,rtol=1e-14)
      @test ll1 == ll2
      @test ll1 == ll3
      @test ll1 == ll4
    end

    @testset "passing a noise grid" begin
      Ws = cumsum([[zero(u0)];[sqrt(trange[i+1]-ti)*randn(size(u0))
                for (i,ti) in enumerate(trange[1:end-1])]])
      NG = NoiseGrid(trange,Ws)


      (uendEM, llEM), (tsEM, uEM, noiseEM) = MSDE.forwardguiding(k3, message4, (u0, ll0), EM(false), NG, inplace=false)
      (uend1, ll1), (ts1, u1, noise1) = MSDE.forwardguiding(k1, message1, (u0, ll0), MSDE.EulerMaruyama!(), NG, inplace=false)
      (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(k2, message2, (u0, ll0), MSDE.EulerMaruyama!(), NG, inplace=false)
      (uend3, ll3), (ts3, u3, noise3) = MSDE.forwardguiding(k3, message3, (u0, ll0), MSDE.EulerMaruyama!(), NG, inplace=false)
      (uend4, ll4), (ts4, u4, noise4) = MSDE.forwardguiding(k3, message4, (u0, ll0), MSDE.EulerMaruyama!(), NG, inplace=false)

      @test u1 ≈ uEM rtol=1e-12
      @test llEM ≈ ll1 rtol=1e-12

      @test isapprox(uend1,uend2,rtol=1e-14)
      @test isapprox(uend1,uend3,rtol=1e-14)
      @test isapprox(uend1,uend4,rtol=1e-14)
      @test ll1 == ll2
      @test ll1 == ll3
      @test ll1 == ll4
    end

    @testset "passing the noise values" begin
      # pass noise process and compare with EM()
      Ws = cumsum([[zero(u0)];[sqrt(trange[i+1]-ti)*randn(size(u0))
              for (i,ti) in enumerate(trange[1:end-1])]])
      NG = NoiseGrid(trange,Ws)

      (uendEM, llEM), (tsEM, uEM, noiseEM) = MSDE.forwardguiding(k3, message4, (u0, ll0), EM(false), NG, inplace=false)
      (uend1, ll1), (ts1, u1, noise1)  = MSDE.forwardguiding(k1, message1, (u0, ll0), MSDE.EulerMaruyama!(), Ws, inplace=false)
      (uend2, ll2), (ts2, u2, noise2) = MSDE.forwardguiding(k2, message2, (u0, ll0), MSDE.EulerMaruyama!(), Ws, inplace=false)
      (uend3, ll3), (ts3, u3, noise3) = MSDE.forwardguiding(k3, message3, (u0, ll0), MSDE.EulerMaruyama!(), Ws, inplace=false)
      (uend4, ll4), (ts4, u4, noise4) = MSDE.forwardguiding(k3, message4, (u0, ll0), MSDE.EulerMaruyama!(), Ws, inplace=false)

      @test u1 ≈ uEM rtol=1e-12
      @test llEM ≈ ll1 rtol=1e-12

      @test isapprox(uend1,uend2,rtol=1e-14)
      @test isapprox(uend1,uend3,rtol=1e-14)
      @test isapprox(uend1,uend4,rtol=1e-14)
      @test ll1 == ll2
      @test ll1 == ll3
      @test ll1 == ll4
    end
  end
end
