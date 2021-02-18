using MitosisStochasticDiffEq
using Mitosis
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

  # define SDE kernel
  sdekernel = MitosisStochasticDiffEq.SDEKernel(foop,goop,trange,par)

  # sample using MitosisStochasticDiffEq and EM default
  Random.seed!(100)
  sol, solend = MitosisStochasticDiffEq.sample(sdekernel, u0, save_noise=true)
  @show solend
  @show length(sol)

  R = MitosisStochasticDiffEq.Regression!(sdekernel,yprototype,
     paramjac_prototype=ϕprototype,paramjac=f_jac,intercept=ϕ0)
  R2 = MitosisStochasticDiffEq.Regression(sdekernel,paramjac=f_jacoop,intercept=ϕ0oop)

  G = MitosisStochasticDiffEq.conjugate(R, sol, 0.1*I(2))
  G2 = MitosisStochasticDiffEq.conjugate(R2, sol, 0.1*I(2))
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
  @testset "AD tests" begin
    using ForwardDiff
    pf = MitosisStochasticDiffEq.ParamJacobianWrapper(sdekernel.f,first(sdekernel.trange),[u0])
    f_jac(ϕprototype,[u0],par,first(sdekernel.trange))
    @test f_jacoop(u0,par,first(sdekernel.trange)) == ϕprototype
    @test ForwardDiff.jacobian(pf, par[1:2]) == ϕprototype

    # check θ function
    RAD = MitosisStochasticDiffEq.Regression!(sdekernel,yprototype,
      paramjac_prototype=ϕprototype,intercept=ϕ0,θ=par[1:2])
    GAD = MitosisStochasticDiffEq.conjugate(RAD, sol, 0.1*I(2))
    @test G ≈ GAD rtol=1e-10

    #check with intercept = nothing
    RAD = MitosisStochasticDiffEq.Regression!(sdekernel,yprototype,
      paramjac_prototype=ϕprototype,θ=par[1:2])
    GAD = MitosisStochasticDiffEq.conjugate(RAD, sol, 0.1*I(2))
    @test G ≈ GAD rtol=1e-10
  end
end
