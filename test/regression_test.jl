import MitosisStochasticDiffEq as MSDE
using Mitosis
using StochasticDiffEq
using Test, Random
using Statistics
using LinearAlgebra

"""
    conjugate_posterior(Y, Ξ)

Sample the posterior distribution of the conjugate drift parameters from path `Y`,
prior precision matrix `Ξ` with non-conjugate parameters fixed in the model.
Adjusted from http://www.math.chalmers.se/~smoritz/journal/2018/01/19/parameter-inference-for-a-simple-sir-model/
for Bridge.jl
"""
function conjugate_posterior(Y, Ξ)
    paramgrad(t, u) = [u, 1]
    paramintercept(t, u) = 0
    t, y = Y.t[1], Y.u[1]
    ϕ = paramgrad(t, y)
    mu = zero(ϕ)
    G = zero(mu*mu')

    #mulist = [mu]

    for i in 1:length(Y)-1
        ϕ = paramgrad(t, y)'
        Gϕ = pinv(Y.prob.g(y, Y.prob.p, t)*Y.prob.g(y, Y.prob.p, t)')*ϕ # a is sigma*sigma'. Todo: smoothing like this is very slow
        zi = ϕ'*Gϕ
        t2, y2 = Y.t[i + 1], Y.u[i + 1]
        dy = y2 - y
        ds = t2 - t
        #@show size(mu), size(Gϕ'), (dy - paramintercept(t, y)*ds)
        mu = mu + Gϕ'*(dy - paramintercept(t, y)*ds)
        t, y = t2, y2
        G = G + zi*ds
        # @show ϕ, mu, G+Ξ, zi, ds
        # if i==2
        #   error()
        # end
        # push!(mulist,mu)
    end
    Mitosis.Gaussian{(:F,:Γ)}(mu, G + Ξ)#, mulist
end

@testset "regression tests" begin

  K = 1000

  # define SDE function
  f(du,u,p,t) = du .= p[1]*u .+ p[2]
  g(du,u,p,t) = du .= p[3]
  foop(u,p,t) = p[1]*u .+ p[2]
  goop(u,p,t) = p[3]

  # paramjac
  function f_jac(J,u,p,t)
    J[1,1] = u[1]
    J[1,2] = true
    nothing
  end

  function f_jacoop(u,p,t)
    [u[1]  true]
  end

  # intercept
  function ϕ0(du,u,p,t)
    du .= false
  end

  function ϕ0oop(u,p,t)
    [zero(eltype(u))]
  end

  ϕprototype = zeros((1,2))
  yprototype = zeros(1)

  # time span
  tstart = 0.0
  tend = 100.0
  dt = 0.001
  trange = tstart:dt:tend

  # initial condition
  u0 = 1.1

  # set true model parameters
  par = [-0.3, 0.2, 0.5]

  # define SDE kernel
  sdekernel = MSDE.SDEKernel(foop,goop,trange,par)
  sdekernel2 = MSDE.SDEKernel(f,g,trange,par)

  # sample using MSDE and EM default
  Random.seed!(100)
  sol, solend = MSDE.sample(sdekernel, u0, save_noise=true)
  Random.seed!(100)
  # for later AD check
  sol2, solend2 = MSDE.sample(sdekernel, [u0], save_noise=true)
  @show solend
  @show length(sol)

  R = MSDE.Regression!(sdekernel,yprototype,
      paramjac_prototype=ϕprototype,paramjac=f_jac,intercept=ϕ0)
  R2 = MSDE.Regression(sdekernel,paramjac=f_jacoop,intercept=ϕ0oop)
  R3 = MSDE.Regression!(sdekernel2,yprototype,
      paramjac_prototype=ϕprototype,paramjac=f_jac,intercept=ϕ0)
  R4 = MSDE.Regression(sdekernel2,
      dyprototype=yprototype,paramjac=f_jacoop,intercept=ϕ0oop)

  G = MSDE.conjugate(R, sol, 0.1*I(2))
  G2 = MSDE.conjugate(R2, sol, 0.1*I(2))
  G3 = MSDE.conjugate(R3, sol, 0.1*I(2))
  G4 = MSDE.conjugate(R4, sol, 0.1*I(2))
  Gtest = conjugate_posterior(sol, 0.1*I(2))

  @testset "Regression! oop tests" begin
    @test G ≈ Gtest rtol=1e-10
    @test G.F ≈ Gtest.F rtol=1e-10
    @test G.Γ ≈ Gtest.Γ rtol=1e-10
  end
  @testset "Regression oop tests" begin
    @test G2 ≈ Gtest rtol=1e-10
    @test G2.F ≈ Gtest.F rtol=1e-10
    @test G2.Γ ≈ Gtest.Γ rtol=1e-10
  end
  @testset "Regression! iip tests" begin
    @test G3 ≈ Gtest rtol=1e-10
    @test G3.F ≈ Gtest.F rtol=1e-10
    @test G3.Γ ≈ Gtest.Γ rtol=1e-10
  end
  @testset "Regression iip tests" begin
    @test G4 ≈ Gtest rtol=1e-10
    @test G4.F ≈ Gtest.F rtol=1e-10
    @test G4.Γ ≈ Gtest.Γ rtol=1e-10
  end
  @info Gtest.F
  @info Gtest.Γ

  # test samples
  mu = G.F
  Gamma = G.Γ
  WL = (cholesky(Hermitian(Gamma)).U)'

  Random.seed!(1)
  Π = []
  Π2 = []
  for i=1:K
    th° = WL'\(randn(size(mu))+WL\mu)
    push!(Π,th°)
    th° = WL'\(randn(size(mu))+WL\mu)
    push!(Π2,th°)
  end

  mu = G2.F
  Gamma = G2.Γ
  WL = (cholesky(Hermitian(Gamma)).U)'

  Random.seed!(1)
  for i=1:K
    th° = WL'\(randn(size(mu))+WL\mu)
    push!(Π2,th°)
  end
  @testset "regression samples tests" begin
    @test par[1:2] ≈ mean(Π) rtol=0.2
    @test par[1:2] ≈ mean(Π2) rtol=0.2
    @test mean(Π) ≈ mean(Π2) atol=0.1
  end
# using Plots
# pl = scatter(first.(Π), last.(Π), markersize=1, c=:blue, label="posterior samples")
# scatter!(first.(Π2), last.(Π2), markersize=1, c=:green, label="posterior samples")
# scatter!([par[1]], [par[2]], color="red", label="truth")
# savefig(pl, "regression.png")

  # test with ForwardDiff
  @testset "AD for Jacobian tests" begin
    using ForwardDiff
    pf = MSDE.ParamJacobianWrapper(sdekernel.f,first(sdekernel.trange),[u0])
    pf2 = MSDE.ParamJacobianWrapper2(sdekernel2.f,first(sdekernel2.trange),[u0])
    f_jac(ϕprototype,[u0],par,first(sdekernel.trange))
    @test f_jacoop(u0,par,first(sdekernel.trange)) == ϕprototype
    @test ForwardDiff.jacobian(pf, par[1:2]) == ϕprototype
    @test ForwardDiff.jacobian(pf2, par[1:2]) == ϕprototype

    # check θ function
    RAD = MSDE.Regression!(sdekernel,yprototype,
      paramjac_prototype=ϕprototype,intercept=ϕ0,θ=par[1:2])
    GAD = MSDE.conjugate(RAD, sol, 0.1*I(2))
    @test G ≈ GAD rtol=1e-10

    #check with intercept = nothing
    RAD = MSDE.Regression!(sdekernel,yprototype,
      paramjac_prototype=ϕprototype,θ=par[1:2])
    RAD2 = MSDE.Regression(sdekernel,θ=par[1:2],yprototype=yprototype)
    GAD = MSDE.conjugate(RAD, sol, 0.1*I(2))
    GAD2 = MSDE.conjugate(RAD2, sol2, 0.1*I(2))
    @test G ≈ GAD rtol=1e-10
    @test G ≈ GAD2 rtol=1e-10

    #check AD for inplace function
    RAD3 = MSDE.Regression!(sdekernel2,yprototype,
        paramjac_prototype=ϕprototype,θ=par[1:2])
    RAD4 = MSDE.Regression(sdekernel2,
        dyprototype=yprototype,yprototype=yprototype,θ=par[1:2])
    GAD3 = MSDE.conjugate(RAD3, sol, 0.1*I(2))
    GAD4 = MSDE.conjugate(RAD4, sol2, 0.1*I(2))

    @test G ≈ GAD3 rtol=1e-10
    @test G ≈ GAD4 rtol=1e-10
  end
end


@testset "regression SDE Problem tests" begin
  # define SDE function
  foop(u,p,t) = p[1]*u .+ p[2]
  goop(u,p,t) = p[3]

  # paramjac
  function f_jac(J,u,p,t)
    J[1,1] = u[1]
    J[1,2] = true
    nothing
  end

  function f_jacoop(u,p,t)
    [u[1]  true]
  end

  # intercept
  function ϕ0(du,u,p,t)
    du .= false
  end

  function ϕ0oop(u,p,t)
    [zero(eltype(u))]
  end

  ϕprototype = zeros((1,2))
  yprototype = zeros(1)

  # time span
  tstart = 0.0
  tend = 100.0
  dt = 0.01
  trange = tstart:dt:tend

  # initial condition
  u0 = 1.1

  # set true model parameters
  par = [-0.3, 0.2, 0.5]

  # define SDE Problem and sample using EM
  Random.seed!(100)
  prob = SDEProblem(foop, goop, u0, (tstart,tend), par)
  sol = solve(prob, EM(), dt = dt)

  @show sol[end]

  R = MSDE.Regression!(prob,yprototype,
     paramjac_prototype=ϕprototype,paramjac=f_jac,intercept=ϕ0)
  R2 = MSDE.Regression(prob,paramjac=f_jacoop,intercept=ϕ0oop)

  G = MSDE.conjugate(R, sol, 0.1*I(2))
  G2 = MSDE.conjugate(R2, sol, 0.1*I(2))
  G3 = conjugate_posterior(sol, 0.1*I(2))

  @testset "iip tests" begin
    @test G ≈ G3 rtol=1e-10
    @test G.F ≈ G3.F rtol=1e-10
    @test G.Γ ≈ G3.Γ rtol=1e-10
  end
  @testset "oop tests" begin
    @test G2 ≈ G3 rtol=1e-10
    @test G2.F ≈ G3.F rtol=1e-10
    @test G2.Γ ≈ G3.Γ rtol=1e-10
  end
  @info G3.F
  @info G3.Γ
end


@testset "scalar Brusselator" begin
  # define SDE function for Brusselator, test inplace version and AD
  using DiffEqNoiseProcess, Zygote, DiffEqSensitivity
  function brusselator_f!(du,u,p,t)
    @inbounds begin
      du[1] = (p[1]-1)*u[1]+p[1]*u[1]^2+(u[1]+1)^2*u[2]
      du[2] = -p[1]*u[1]-p[1]*u[1]^2-(u[1]+1)^2*u[2]
    end
    nothing
  end

  function brusselator_f(u,p,t)
    dx1 = (p[1]-1)*u[1]+p[1]*u[1]^2+(u[1]+1)^2*u[2]
    dx2 = -p[1]*u[1]-p[1]*u[1]^2-(u[1]+1)^2*u[2]

    return [dx1,dx2]
  end

  function scalar_noise!(du,u,p,t)
    @inbounds begin
      du[1] = p[2]*u[1]*(1+u[1])
      du[2] = -p[2]*u[1]*(1+u[1])
     end
     nothing
  end

  function scalar_noise(u,p,t)
    dx1 = p[2]*u[1]*(1+u[1])
    dx2 = -p[2]*u[1]*(1+u[1])

    return [dx1,dx2]
  end

  # matrix form
  function scalar_noise_matrix(u,p,t)
    dx1 = p[2]*u[1]*(1+u[1])
    dx2 = -p[2]*u[1]*(1+u[1])

    reshape([ dx1, dx2],2,1)
  end

  # paramjac
  function f_jac!(J,u,p,t)
    J[1,1] = u[1]+u[1]^2
    J[2,1] = -(u[1]+u[1]^2)
    nothing
  end

  function f_jac(u,p,t)
    [ u[1]+u[1]^2
      -(u[1]+u[1]^2)]
  end

  # intercept
  function ϕ0!(du,u,p,t)
    du[1] = -u[1]+(u[1]+1)^2*u[2]
    du[2] = -(u[1]+1)^2*u[2]
  end

  function ϕ0(u,p,t)
    dx1 = -u[1]+(u[1]+1)^2*u[2]
    dx2 = -(u[1]+1)^2*u[2]
    [dx1, dx2]
  end

  yprototype = zeros(2)
  ϕprototype = zeros((2,1))

  # fix seeds
  seed = 100
  Random.seed!(seed)
  W = WienerProcess(0.0,0.0,nothing)

  u0 = [-0.1,0.0]
  tspan = (0.0,100.0)
  p = [1.9,0.1] # p[1] is only in the drift

  prob = SDEProblem(brusselator_f!,scalar_noise!,u0,tspan,p,noise=W)
  sol = solve(prob, EM(), dt = 0.01)

  @test size(scalar_noise_matrix(sol.u[1], p, 0.0)) == (2,1)

  testvec = randn(2)
  o1 = MSDE.outer_(scalar_noise_matrix(testvec, p, 0.0))
  o2 = MSDE.outer_(reshape(scalar_noise(testvec,p,0.0),2,1))
  @test o1==o2

  # oop regression with matrix form for diffusion to test
  sdekernel1 = MSDE.SDEKernel(brusselator_f,scalar_noise_matrix,tspan,p)
  R1 = MSDE.Regression(sdekernel1,paramjac=f_jac,intercept=ϕ0)
  # oop regression with vector form as in StochasticDiffEq
  sdekernel2 = MSDE.SDEKernel(brusselator_f,scalar_noise,tspan,p)
  R2 = MSDE.Regression(sdekernel2,paramjac=f_jac,intercept=ϕ0, m=1)
  # inplace regression with vector form as in StochasticDiffEq
  sdekernel3 = MSDE.SDEKernel(brusselator_f!,scalar_noise!,tspan,p)
  R3 = MSDE.Regression!(sdekernel3,yprototype,
     paramjac_prototype=ϕprototype,paramjac=f_jac!,intercept=ϕ0!,m=1)


  G1 = MSDE.conjugate(R1, sol, 0.1*I(1))
  G2 = MSDE.conjugate(R2, sol, 0.1*I(1))
  G3 = MSDE.conjugate(R3, sol, 0.1*I(1))

  @test G1.F ≈ G2.F rtol=1e-10
  @test G1.Γ ≈ G2.Γ rtol=1e-10
  @test G1.F ≈ G3.F rtol=1e-10
  @test G1.Γ ≈ G3.Γ rtol=1e-10
  @test G2.F ≈ G3.F rtol=1e-10
  @test G2.Γ ≈ G3.Γ rtol=1e-10

  # Estimate
  p̂ = mean(G1)
  se = sqrt.(diag(cov(G1)))
  display(map((p̂, se, p) -> "$(round(p̂, digits=3)) ± $(round(se, digits=3)) (true: $p)", p̂, se, p))
  @test p̂[1] ≈ p[1] rtol=se[1]

  # use NoiseWrapper to reproduce same trajectory
  # Wrep = NoiseWrapper(sol.W)
  ts = sol.t
  gaussian = Mitosis.Gaussian{(:F,:Γ)}(zeros(eltype(Matrix(0.1*I(1))), size(Matrix(0.1*I(1)), 1)), Matrix(0.1*I(1)))

  function loss(p,u0,R)
    _prob = remake(prob,p = p, u0 = u0, noise=W)
    _sol = (solve(_prob, EM(), dt = 0.01)).u
    G = MSDE.conjugate(R, _sol, gaussian, ts)
    p̂ = mean(G)[1]
    se = sqrt.(diag(cov(G)))[1]
    se + (p̂-p[1])^2
  end
  @test loss(p,u0,R1) != zero(loss(p,u0,R1))
  @test loss(p,u0,R2) != zero(loss(p,u0,R2))
  @test loss(p,u0,R3) != zero(loss(p,u0,R3))

  @test loss(p,u0,R1) ≈ loss(p,u0,R2) atol=1e-3
  @test loss(p,u0,R2) ≈ loss(p,u0,R3) atol=1e-3

  du01,dp1 = Zygote.gradient((u0,p)->loss(p,u0,R1),u0,p)
  @test du01 != zero(du01)
  @test dp1 != zero(dp1)
end
