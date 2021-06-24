import MitosisStochasticDiffEq as MSDE
using Test, Random
using StaticArrays
using Statistics
using LinearAlgebra

@testset "static array tests" begin
seed = 1234
Random.seed!(seed)

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3]

function fstat(u,p,t)
  dx = p[1]*u[1] + p[2]
  @SVector [dx]
end

function gstat(u,p,t)
  @SVector [p[3]]
end

# time span
tstart = 0.0
tend = 1.0
dt = 0.01
trange = tstart:dt:tend

# initial condition
u0 = [1.1]
u0stat = @SVector [1.1]

# set true model parameters
par = [-0.2, 0.1, 0.9]

# set of linear parameters Eq.~(2.2)
plin = copy(par)

sdekernel1 = MSDE.SDEKernel(f,g,trange,par)
sdekernel2 = MSDE.SDEKernel(fstat,gstat,trange,par)

Random.seed!(seed)
samples1 = MSDE.sample(sdekernel1, u0, save_noise=false)
Random.seed!(seed)
samples2 = MSDE.sample(sdekernel2, u0stat, save_noise=false)

@test isapprox(samples1[1], samples2[1], rtol=1e-10)
@test isapprox(samples1[2][3], samples2[2][3], rtol=1e-10)
@test typeof(samples2[2][2][end]) <: SArray


# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues1 =  [0.0, 1.5, 0.1];
NT1 = NamedTuple{mynames}(myvalues1)

#myvalues2 = @SVector [0.0, 1.5, 0.1] # will be just converted to a standard array
myvalues2 = [0.0, (@SVector [1.5]), (@SMatrix[0.1])]
NT2 = NamedTuple{mynames}(myvalues2)

sdetildekernel1 = MSDE.SDEKernel(f,g,trange,plin)
sdetildekernel2 = MSDE.SDEKernel(fstat,gstat,trange,plin)
message1, backward1 = MSDE.backwardfilter(sdetildekernel1, NT1)
message2, backward2 = MSDE.backwardfilter(sdetildekernel2, NT2)

@test isapprox(backward1, backward2, rtol=1e-10)
@test typeof(backward2.x[1]) <: SArray
@test typeof(backward2.x[2]) <: SArray
@test typeof(backward2.x[3]) <: SArray

x0 = [1.34]
x0stat = @SVector [1.34]
ll0 = randn()

Random.seed!(seed)
samples1 = MSDE.forwardguiding(sdekernel1, message1,
  (x0, ll0), inplace=true)
Random.seed!(seed)
samples2 = MSDE.forwardguiding(sdekernel2, message2,
  (x0stat, ll0), inplace=false)

@test isapprox(samples1[2][1], samples2[2][1], rtol=1e-10)
@test isapprox(samples1[2][2], samples2[2][2], rtol=1e-10)
@test_broken typeof(samples2[2][end]) <: SArray
@test_broken typeof(samples2[3]) <: SArray
end

@testset "static tilde parameter tests" begin
  d = 2

  𝕏 = SVector{d,Float64}
  𝕏_ = SVector{d+1,Float64}
  bθ = @SVector [.5, 0.9]
  σ0 = @SVector [1.25, 1.35]
  θ0 = (bθ, σ0)

  M = @SMatrix [-1.0 1.0; 1.0 -1.0]
  f(u,θ,t) = 𝕏(tanh.(Diagonal(θ[1]) * M * u))  # f(u,θ,t) = Diagonal(θ[1]) * M * u
  g(u,θ,t) = θ[2]

  dt0 = 0.001
  trange = 0.0:dt0:1.0

  u0 = zero(𝕏)  # value at root node

  B(θ) = (Diagonal(θ[1]) * M)
  Σ(θ) = SMatrix{2,2}(Diagonal(θ[2]))
  b̃ = @SVector [.1, 0.1]  #θ̃ = θ0
  θ̃ = (b̃, σ0)


  θlinstat = ((B(θ̃), @SVector(zeros(d)), Σ(θ̃)))
  θlin =  Array.(θlinstat)
  @show typeof(θlin), typeof(θlinstat)

  # initial values for ODE
  Random.seed!(123)
  logscale = randn()
  ν = @SVector randn(d)
  P = @SMatrix randn(d,d)
  myvalues = [logscale, ν, P]
  mynames = (:logscale, :μ, :Σ)
  NTstat = NamedTuple{mynames}(myvalues)
  NT = NamedTuple{mynames}([logscale, Array(ν), Array(P)])

  tildekernelstat = MSDE.SDEKernel(f,g,trange,θlinstat)
  messagestat, backwardstat = MSDE.backwardfilter(tildekernelstat, NTstat)

  @test typeof(backwardstat.x[1]) <: SArray
  @test typeof(backwardstat.x[2]) <: SArray
  @test typeof(backwardstat.x[3]) <: SArray

  tildekernel = MSDE.SDEKernel(f,g,trange,θlin)
  message, backward = MSDE.backwardfilter(tildekernel, NT)

  @test !(typeof(backward.x[1]) <: SArray)
  @test !(typeof(backward.x[2]) <: SArray)
  @test !(typeof(backward.x[3]) <: SArray)

  @test isapprox(backwardstat.x[1], backward.x[1], rtol=1e-10)
  @test isapprox(backwardstat.x[2], backward.x[2], rtol=1e-10)
  @test isapprox(backwardstat.x[3], backward.x[3], rtol=1e-10)

  # mixed cases
  message1, backward1 = MSDE.backwardfilter(tildekernel, NTstat)
  @test typeof(backward1.x[1]) <: SArray
  @test typeof(backward1.x[2]) <: SArray
  @test typeof(backward1.x[3]) <: SArray

  @test isapprox(backward1.x[1], backward.x[1], rtol=1e-10)
  @test isapprox(backward1.x[2], backward.x[2], rtol=1e-10)
  @test isapprox(backward1.x[3], backward.x[3], rtol=1e-10)

  message2, backward2 = MSDE.backwardfilter(tildekernelstat, NT)
  @test !(typeof(backward2.x[1]) <: SArray)
  @test !(typeof(backward2.x[2]) <: SArray)
  @test !(typeof(backward2.x[3]) <: SArray)

  @test isapprox(backward2.x[1], backward.x[1], rtol=1e-10)
  @test isapprox(backward2.x[2], backward.x[2], rtol=1e-10)
  @test isapprox(backward2.x[3], backward.x[3], rtol=1e-10)

  # backward
  (c, μ, Pmat) = NT
  p = tildekernel.p
  u0 = MSDE.mypack(μ, Pmat, c)
  MSDE.CovarianceFilterODE(u0, p, 0.0)
  b, β, σtil = p
  MSDE.compute_dν(b,μ,β)
  MSDE.compute_dP(b,Pmat,σtil)
  @test typeof(MSDE.compute_dν(b,μ,β)) == typeof(μ)
  @test typeof(MSDE.compute_dP(b,Pmat,σtil)) == typeof(Pmat)

  # backwardstat
  (c, μ, Pmat) = NTstat
  p = tildekernelstat.p
  u0 = MSDE.mypack(μ, Pmat, c)
  MSDE.CovarianceFilterODE(u0, p, 0.0)
  b, β, σtil = p
  MSDE.compute_dν(b,μ,β)
  MSDE.compute_dP(b,Pmat,σtil)
  @test typeof(MSDE.compute_dν(b,μ,β)) == typeof(μ)
  @test typeof(MSDE.compute_dP(b,Pmat,σtil)) == typeof(Pmat)

  # backward1
  (c, μ, Pmat) = NTstat
  p = tildekernel.p
  u0 = MSDE.mypack(μ, Pmat, c)
  MSDE.CovarianceFilterODE(u0, p, 0.0)
  b, β, σtil = p
  MSDE.compute_dν(b,μ,β)
  MSDE.compute_dP(b,Pmat,σtil)
  @test typeof(MSDE.compute_dν(b,μ,β)) == typeof(μ)
  @test typeof(MSDE.compute_dP(b,Pmat,σtil)) == typeof(Pmat)

  # backward2
  (c, μ, Pmat) = NT
  p = tildekernelstat.p
  u0 = MSDE.mypack(μ, Pmat, c)
  MSDE.CovarianceFilterODE(u0, p, 0.0)
  b, β, σtil = p
  @test typeof(MSDE.compute_dν(b,μ,β)) == typeof(μ)
  @test typeof(MSDE.compute_dP(b,Pmat,σtil)) == typeof(Pmat)
end
