using MitosisStochasticDiffEq
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

  kernel = MitosisStochasticDiffEq.SDEKernel(f,g,trange,p)
  # sample using MitosisStochasticDiffEq and EM default
  sol, solend = MitosisStochasticDiffEq.sample(kernel, u0)

  kernel = MitosisStochasticDiffEq.SDEKernel(f,g,collect(trange),p)
  sol, solend = MitosisStochasticDiffEq.sample(kernel, u0, save_noise=true)

  @test isapprox(sol.u, forwardsample(f,g,p,sol.t,sol.W.W,sol.prob.u0), atol=1e-12)
end


@testset "multivariate sampling tests" begin
  Random.seed!(12345)
  d = 2
  u0 = randn(2)
  θlin = (randn(d,d), randn(d), Diagonal([0.1, 0.1]))

  Σ(θ) = Diagonal(θ[2]) # just to generate the noise_rate_prototype

  f(u,p,t) = p[1]*u + p[2]
  f!(du,u,p,t) = (du .= p[1]*u + p[2])
  gvec(u,p,t) = diag(p[3])
  g(u,p,t) = p[3]
  # Make `g` write the sparse matrix values
  function g!(du,u,p,t)
    du[1,1] = p[3][1,1]
    du[2,2] = p[3][2,2]
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
  k1 = MitosisStochasticDiffEq.SDEKernel(f,gvec,trange,θlin)
  k2 = MitosisStochasticDiffEq.SDEKernel(f,g,trange,θlin,Σ(θlin))
  k3 = MitosisStochasticDiffEq.SDEKernel(f!,g!,trange,θlin,A)
  k4 = MitosisStochasticDiffEq.SDEKernel(Mitosis.AffineMap(θlin[1], θlin[2]), Mitosis.ConstantMap(θlin[3]), trange, θlin, Σ(θlin))

  sol1, solend1 = MitosisStochasticDiffEq.sample(k1, u0, save_noise=true)
  Z = pCN(sol1.W, 1.0)
  sol2, solend2 = MitosisStochasticDiffEq.sample(k2, u0, Z=Z, save_noise=true)
  Z = pCN(sol1.W, 1.0)
  sol3, solend3 = MitosisStochasticDiffEq.sample(k3, u0, Z=Z)
  Z = pCN(sol1.W, 1.0)
  sol4, solend4 = MitosisStochasticDiffEq.sample(k4, u0, Z=Z)

  @show solend1
  @test isapprox(sol1.u, sol2.u, atol=1e-12)
  @test isapprox(solend1, solend2, atol=1e-12)
  @test isapprox(sol1.u, sol3.u, atol=1e-12)
  @test isapprox(solend1, solend3, atol=1e-12)
  @test isapprox(sol1.u, sol4.u, atol=1e-12)
  @test isapprox(solend1, solend4, atol=1e-12)
end
