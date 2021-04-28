using MitosisStochasticDiffEq
using Mitosis
using DiffEqNoiseProcess
using Test, Random
using LinearAlgebra
using Statistics

# Test outer function
@testset "outer function tests" begin
  exA = rand(10,10)
  @test minimum(MitosisStochasticDiffEq.outer_(exA) .!= 0)
  exB = Diagonal(exA)
  @test sum(MitosisStochasticDiffEq.outer_(exB) .!= 0) == 10
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
      tmp = MitosisStochasticDiffEq.outer_(g(x,pest,t)) - MitosisStochasticDiffEq.outer_(σlinear(x,plin,t))
      dll = dot(f(x,pest,t) - flinear(x,plin,t), r) -0.5*tr(tmp*(inv(P) - MitosisStochasticDiffEq.outer_(r)))
    end

    xs = typeof(x)[]
    d = length(x)
    for i in eachindex(s)[1:end-1]
        dt = s[i+1] - s[i]
        t = s[i]
        push!(xs, x)
        ν = @view ps[:,i][1:d]
        P = reshape(@view(ps[:,i][d+1:d+d*d]), d, d)
        r = inv(P)*(ν .- x)

        ll += llstep(x, r, t, P, noisetype)*dt # accumulate log-likelihood

        if noisetype == :scalar
            noise = g(x,pest,t)*Z[i] #sqrt(dt)*Z[i]
        elseif noisetype ==:diag
            noise = g(x,pest,t).*Z[:,i]
        elseif noisetype ==:nondiag
            noise = g(x,pest,t)*Z[:,i]
        else
            error("noisetype not understood.")
        end
        if x isa Number
           tmp = (MitosisStochasticDiffEq.outer_(g(x,pest,t))*r*dt)[1]
       else
           tmp = MitosisStochasticDiffEq.outer_(g(x,pest,t))*r*dt
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
  sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)

  x0 = randn()
  ll0 = randn()

  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            Z=nothing; save_noise=true)


  dWs = (solfw.W[1,2:end]-solfw.W[1,1:end-1])
  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0),ps,dWs)

  @test isapprox(solfw[1,:], solfw2, rtol=1e-12)
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
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)

  x0 = randn(dim)
  ll0 = randn()
  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0), NG)

  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0), ps, W)

  @test isapprox(Array(solfw)[1:dim,:], hcat(solfw2 ...), rtol=1e-12)
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
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)

  x0 = randn(dim)
  ll0 = randn()

  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0); save_noise=true)

  Ws = Array(solfw.W)
  dWs = Ws[1:dim,2:end]-Ws[1:dim,1:end-1]

  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0),ps,dWs,:diag)

  @test isapprox(Array(solfw)[1:dim,:], hcat(solfw2 ...), rtol=1e-12)
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
  sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)

  x0 = randn()
  ll0 = randn()

  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            Z=nothing; save_noise=true, inplace=false)


  dWs = (solfw.W[1,2:end]-solfw.W[1,1:end-1])
  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0),ps,dWs)

  @test isapprox(solfw[1,:], solfw2, rtol=1e-12)
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
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)

  x0 = randn(dim)
  ll0 = randn()
  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0), NG, inplace=false)

  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0), ps, W)

  @test isapprox(Array(solfw)[1:dim,:], hcat(solfw2 ...), rtol=1e-12)
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
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)

  x0 = randn(dim)
  ll0 = randn()

  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0); save_noise=true, inplace=false)

  Ws = Array(solfw.W)
  dWs = Ws[1:dim,2:end]-Ws[1:dim,1:end-1]

  ps = message.soldis
  solfw2, ll2 = forwardguiding(plin, pest, message.ts, (x0, ll0),ps,dWs,:diag)

  @test isapprox(Array(solfw)[1:dim,:], hcat(solfw2 ...), rtol=1e-12)
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
  sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)


  # define NoiseGrid
  brownian_values = cumsum([[zeros(2)];[sqrt(dt)*randn(2) for i in 1:length(trange)-1]])
  brownian_values2 = cumsum([[zeros(2)];[sqrt(dt)*randn(2) for i in 1:length(trange)-1]])
  W = NoiseGrid(collect(trange),brownian_values,brownian_values2)

  x0 = randn()
  ll0 = randn()

  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            W; alg=LambaEM(), dt=dt, isadaptive=false)
  solfw2, ll2 = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            W; alg=LambaEM(), dt=dt, isadaptive=true)
  solfw3, ll3 = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            W; alg=SOSRI(), dt=dt, isadaptive=true)

  @test isapprox(ll, ll2, rtol=1e-1)
  @test isapprox(ll, ll3, rtol=1e-1)
  @test isapprox(ll2, ll2, rtol=1e-1)
  @test isapprox(solfw(solfw2.t).u, solfw2.u, rtol=1e-1)
  @test isapprox(solfw(solfw3.t).u, solfw3.u, rtol=1e-1)

  @show length(solfw.t), length(solfw2.t), length(solfw3.t)

# using Plots
# pl = plot(solfw)
# plot!(solfw2)
# plot!(solfw3)
# savefig(pl,"adaptive_guiding.png")

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
  sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT, apply_timechange=true)

  x0 = randn()
  ll0 = randn()

  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0);
    isadaptive=false)

  @test isapprox(solfw.t, message.ts, rtol=1e-10)
  @test isapprox(solfw.t, MitosisStochasticDiffEq.timechange(trange), rtol=1e-10)
  @test length(solfw.t) == length(trange)

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
  sdekernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,pest)

  # initial values for ODE
  mynames = (:logscale, :μ, :Σ);
  myvalues = [0.0, 0.0, 10.0];
  NT = NamedTuple{mynames}(myvalues)

  # backward kernel
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)


  # define NoiseGrid
  brownian_values = cumsum([[zeros(2)];[sqrt(dt)*randn(2) for i in 1:length(trange)-1]])
  W = NoiseGrid(collect(trange),brownian_values)

  x0 = randn()
  ll0 = randn()

  # test two subsequent evaluations with same Brownian motion given by NoiseGrid
  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            W; alg=EM(), dt=dt)
  solfw2, ll2 = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            W; alg=EM(), dt=dt)

  @test isapprox(ll, ll2, rtol=1e-14)
  @test isapprox(solfw.u, solfw2.u, rtol=1e-14)
  @test isapprox(solfw.W.W, solfw2.W.W, rtol=1e-14)
  @test isapprox(solfw.W.W, W.W, rtol=1e-14)

  # test pCN with \rho = 1
  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0)
             ; alg=EM(), dt=dt, save_noise=true)

  Z = pCN(solfw.W, 1.0)

  solfw2, ll2 = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            Z; alg=EM(), dt=dt)

  @test isapprox(ll, ll2, rtol=1e-14)
  @test isapprox(solfw.u, solfw2.u, rtol=1e-14)
  @test isapprox(solfw.W.W, solfw2.W.W, rtol=1e-14)
  @test isapprox(solfw.W.W, Z.W, rtol=1e-14)
  @test W.W != Z.W


  # test pCN with ρ = 0.2 (decrease dt for test)
  ρ = 0.2
  dt = 0.0001
  trange = tstart:dt:tend
  # backward kernel
  kerneltilde = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(B, β), Mitosis.ConstantMap(σ̃), trange, plin)
  message, backward = MitosisStochasticDiffEq.backwardfilter(kerneltilde, NT)
  solfw, ll = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0)
             ; alg=EM(), dt=dt, save_noise=true)

  Z = pCN(solfw.W, ρ)

  solfw2, ll2 = MitosisStochasticDiffEq.forwardguiding(sdekernel, message, (x0, ll0),
            Z; alg=EM(), dt=dt)

  computedW(W, indx, dt) = (W[indx+1][1]-W[indx][1])/sqrt(dt) # likelihood part can be ignored
  dWnew = []
  dWold = []
  for i in 1:length(solfw.t[1:end-1])
    push!(dWnew,computedW(solfw.W.W,i,dt))
    push!(dWold,computedW(solfw2.W.W,i,dt))
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
  κ1 = MitosisStochasticDiffEq.SDEKernel(f, g, trange, θ0)
  κ2 = MitosisStochasticDiffEq.SDEKernel(f, g, trange, θ0, nothing, true)
  # backward kernel
  κ̃ = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(θlin[1], θlin[2]), Mitosis.ConstantMap(θlin[3]), trange, θlin)

  # forward sample
  x, xT = MitosisStochasticDiffEq.sample(κ1, u0; save_noise=true)
  Z = NoiseWrapper(x.W)
  x2, xT2 = MitosisStochasticDiffEq.sample(κ2, u0; Z=Z)

  @test x.u ≈ x2.u
  @test xT ≈ xT2

  # backward filter
  logscale = randn()
  ν = randn(d)
  P = randn(d,d)
  gaussian = WGaussian{(:μ,:Σ,:c)}(ν, P, logscale)

  message, backward = MitosisStochasticDiffEq.backwardfilter(κ̃, gaussian)

  # forward guiding
  κg1 = MitosisStochasticDiffEq.SDEKernel(f, g, trange, θ)
  κg2 = MitosisStochasticDiffEq.SDEKernel(f, g, trange, θ, nothing, true)

  x0 = randn(d)
  ll0 = randn()

  solfw1, ll1 = MitosisStochasticDiffEq.forwardguiding(κg1, message, (x0, ll0); save_noise=true)
  Z = pCN(solfw1.W, 1.0)
  solfw2, ll2 = MitosisStochasticDiffEq.forwardguiding(κg2, message, (x0, ll0), Z)

  @test ll1 ≈ ll2
  @test isapprox(solfw1.u, solfw2.u, rtol=1e-14)
  @test isapprox(solfw1.W.W, solfw2.W.W, rtol=1e-14)


  # check guiding with matrix-valued diffusion
  gmat(u,θ,t) = Diagonal(θ[2])
  kg3 = MitosisStochasticDiffEq.SDEKernel(f, gmat, trange, θ, Σ(θ), true)

  # inplace=true
  Z = pCN(solfw1.W, 1.0)
  solfw3, ll3 = MitosisStochasticDiffEq.forwardguiding(kg3, message, (x0, ll0), Z)
  @test ll1 ≈ ll3
  @test isapprox(solfw1.u, solfw3.u, rtol=1e-14)

  # inplace=false
  Z = pCN(solfw1.W, 1.0)
  solfw3, ll3 = MitosisStochasticDiffEq.forwardguiding(kg3, message, (x0, ll0), Z, inplace=false)
  @test ll1 ≈ ll3
  @test isapprox(solfw1.u, solfw3.u, rtol=1e-14)

end
