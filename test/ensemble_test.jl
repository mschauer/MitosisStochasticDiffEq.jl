import MitosisStochasticDiffEq as MSDE
using StochasticDiffEq
using Mitosis
using Test, Random
using Statistics

K = 200_000
Random.seed!(1234)

# define SDE function
f(u,p,t) = p[1]*u .+ p[2]
g(u,p,t) = p[3]

# time span
tstart = 0.0
tend = 1.0
dt = 0.01
trange = tstart:dt:tend

# initial condition
u0 = 1.1

# set true model parameters
par = [-0.2, 0.1, 0.9]

# set of linear parameters Eq.~(2.2)
plin = copy(par)
sdekernel = MSDE.SDEKernel(f,g,trange,par)

@testset "ensemble sampling tests" begin
  samples1 = MSDE.sample(sdekernel, u0, K, save_noise=false).u
  samples2 = [MSDE.sample(sdekernel, u0, save_noise=false)[2] for _ in 1:K]

  @test isapprox(mean(samples1), mean(samples2), rtol=1e-2)
  @test isapprox(cov(samples1), cov(samples2), rtol=1e-2)
end

# initial values for ODE
mynames = (:logscale, :μ, :Σ);
myvalues = [0.0, 1.5, 0.1];
NT = NamedTuple{mynames}(myvalues)

sdetildekernel = MSDE.SDEKernel(f,g,trange,plin)
message, backward = MSDE.backwardfilter(sdetildekernel, NT)

sdetildekernel2 = MSDE.SDEKernel(Mitosis.AffineMap(plin[1], plin[2]), Mitosis.ConstantMap(plin[3]), trange, plin)
message2, backward2 = MSDE.backwardfilter(sdetildekernel2, NT)

@testset "ensemble backward tests" begin
  @test isapprox(backward, backward2, rtol=1e-15)
  @test isapprox(Array(message.sol), Array(message2.sol), rtol=1e-15)
end

x0 = 1.34
ll0 = randn()

samples1 = [MSDE.forwardguiding(sdekernel, message, (x0, ll0))[1][1,end] for k in 1:K]
samples2 = MSDE.forwardguiding(sdekernel, message, (x0, ll0), numtraj=K)[1][1,end,:]

@testset "ensemble guiding tests" begin
  @test isapprox(mean(samples1), mean(samples2), rtol=1e-2)
  @test isapprox(cov(samples1), cov(samples2), rtol=1e-2)
end
